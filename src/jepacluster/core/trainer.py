"""
This file contains the training loop.
"""

from pathlib import Path
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.logger import get_logger
from utils.config import ModelConfig, TrainingConfig
from core.architecture import JEPAArchitecture
from core.dataset import LogMaskDataset, MaskSampler, collate_jepa_batches

logger = get_logger(__name__)


class Trainer:
    """
    Trainer class for training the JEPA model.
    """

    def __init__(self, data_dir: str, model_dir: str, model_config: ModelConfig, training_config: TrainingConfig):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model_config = model_config
        self.training_config = training_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = self.load_data()
        self.model = JEPAArchitecture(model_config, vocab_size=self.dataset.vocab_size).to(self.device)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.training_config.use_amp and self.device.type == "cuda")
        self.start_epoch = 0
        self.global_step = 0
        self.optimizer = None
        self.total_iterations = 1
        self.warmup_iterations = 1
        self.base_lr = self.training_config.learning_rate
        self.final_lr = self.training_config.learning_rate * 0.1
        self.base_wd = self.training_config.weight_decay
        self.final_wd = self.training_config.weight_decay
        self.resume_if_available()

    def train(self):
        """
        Full JEPA training loop:
        1. load a batch of windows
        2. hide some tokens with masks
        3. encode the visible context
        4. encode the hidden target with the frozen teacher
        5. predict the hidden target in latent space
        6. update only the context encoder + predictor
        7. move the target encoder with EMA
        """

        dataset = self.load_data()
        logger.info("Dataset ready: approx_samples=%d, vocab_size=%d", getattr(dataset, "sample_count", 0), dataset.vocab_size)

        optimizer = torch.optim.Adam(
            list(self.model.context_encoder.parameters()) + list(self.model.predictor.parameters()),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        self.optimizer = optimizer
        self._build_schedulers(optimizer)

        logger.info("Starting training on %s", self.device)
        self.model.train()

        num_workers = 4 if self.device.type == "cuda" else 2
        logger.info("Using DataLoader workers: %d", num_workers)

        for epoch in range(self.start_epoch, self.training_config.epochs):
            epoch_loss = 0.0
            self.model.train()
            logger.info("Epoch %d/%d starting", epoch + 1, self.training_config.epochs)

            dataloader = DataLoader(
                dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_jepa_batches,
                num_workers=num_workers,
                pin_memory=self.device.type == "cuda",
                persistent_workers=num_workers > 0,
            )
            logger.info("Dataloader ready for epoch %d", epoch + 1)

            for batch in dataloader:
                if self.global_step >= self.training_config.max_steps:
                    logger.info("Reached max_steps=%d, stopping epoch early", self.training_config.max_steps)
                    break
                tokens = batch["tokens"].to(self.device)
                optimizer.zero_grad(set_to_none=True)

                # jepa usually uses more than one mask per batch item; we sample a few masked views so the model learns to predict different hidden regions
                mask_set = MaskSampler(
                    window_size=self.model_config.window_size,
                    num_context_tokens=max(1, self.model_config.window_size - 1),
                    num_target_tokens=1,
                    num_masks=2,
                ).sample()
                batch_losses = []

                for mask_pair in mask_set:
                    context_mask_i = mask_pair["context_mask"].to(self.device)
                    target_mask_i = mask_pair["target_mask"].to(self.device)

                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.scaler.is_enabled()):
                        context_tokens = self.model.encode_context(tokens, mask=context_mask_i)
                        with torch.no_grad():
                            # never backprop through the target encoder - it moves by EMA instead
                            target_tokens = self.model.encode_target(tokens, mask=target_mask_i)

                        # predictor must know which hidden positions it is trying to fill in
                        target_positions = torch.arange(tokens.size(1), device=self.device)[target_mask_i]
                        if target_positions.numel() == 0:
                            continue

                        prediction = self.model.predict(context_tokens, target_positions=target_positions)
                        # jepa compares predicted representations to teacher representations in latent space
                        if prediction.numel() == 0 or target_tokens.numel() == 0:
                            continue
                        batch_losses.append(torch.nn.functional.smooth_l1_loss(prediction, target_tokens))

                if not batch_losses:
                    continue

                loss = torch.stack(batch_losses).mean()
                self._step_schedulers()
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.context_encoder.parameters(), self.training_config.clip_grad_norm)
                    nn.utils.clip_grad_norm_(self.model.predictor.parameters(), self.training_config.clip_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.context_encoder.parameters(), self.training_config.clip_grad_norm)
                    nn.utils.clip_grad_norm_(self.model.predictor.parameters(), self.training_config.clip_grad_norm)
                    optimizer.step()
                self.model.update_target_encoder(self.training_config.ema_decay)

                self._log_collapse_stats(batch, loss)
                epoch_loss += loss.item()
                self.global_step += 1
                if self.global_step == 1 or self.global_step % 10 == 0:
                    logger.info("step=%d running_loss=%.6f", self.global_step, loss.item())

            avg_loss = epoch_loss / max(self.global_step if self.global_step > 0 else 1, 1)
            logger.info("Epoch %d/%d - loss: %.6f", epoch + 1, self.training_config.epochs, avg_loss)
            self.save_model(epoch=epoch + 1)

    def _build_schedulers(self, optimizer):
        # the official jepa repo schedules learning weight and weight decay over epochs
        # we keep the same behavior by treating the training loop as an iteration budget
        self.total_iterations = max(1, self.training_config.epochs)
        self.warmup_iterations = max(1, self.training_config.warmup_epochs)
        self.optimizer = optimizer

    def _step_schedulers(self):
        if self.optimizer is None:
            return

        step = self.global_step
        total = max(1, self.total_iterations)
        warmup = max(1, self.warmup_iterations)

        if step < warmup:
            progress = float(step + 1) / float(warmup)
            lr_scale = progress
        else:
            decay_span = max(1, total - warmup)
            progress = float(step - warmup) / float(decay_span)
            progress = min(1.0, max(0.0, progress))
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        lr = self.final_lr + (self.base_lr - self.final_lr) * lr_scale
        wd = self.final_wd + (self.base_wd - self.final_wd) * lr_scale

        for group in self.optimizer.param_groups:
            group["lr"] = lr
            group["weight_decay"] = wd

    def _log_collapse_stats(self, batch, loss):
        with torch.no_grad():
            tokens = batch["tokens"].to(self.device)
            context_mask = batch["context_mask"].to(self.device)
            encoded = self.model.encode_context(tokens, mask=context_mask)
            var = encoded.var(dim=1).mean().item()
            min_var = encoded.var(dim=1).min().item()
            logger.info("step=%d loss=%.6f latent_var=%.6f latent_min_var=%.6f", self.global_step, loss.item(), var, min_var)

    def load_data(self):
        # turns each log file into many masked prediction examples.
        return LogMaskDataset(
            self.data_dir,
            window_size=self.model_config.window_size,
            min_freq=self.training_config.min_freq,
            max_vocab_size=self.training_config.max_vocab_size,
            tokenizer_path=self.model_config.tokenizer_path,
        )

    def resume_if_available(self):
        checkpoint_path = self.training_config.resume_checkpoint
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            logger.info("Resume checkpoint not found: %s", path)
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.start_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.total_iterations = checkpoint.get("total_iterations", self.total_iterations)
        self.warmup_iterations = checkpoint.get("warmup_iterations", self.warmup_iterations)
        logger.info("Resumed from checkpoint %s at epoch %d", path, self.start_epoch)

    def save_model(self, epoch: int):
        model_dir = Path(self.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{self.model_config.name}.pt"
        torch.save(
            {
                "model_config": self.model_config.model_dump(),
                "training_config": self.training_config.model_dump(),
                "model_state_dict": self.model.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
                "total_iterations": self.total_iterations,
                "warmup_iterations": self.warmup_iterations,
                "vocab_size": self.dataset.vocab_size,
            },
            model_path,
        )
        logger.info("Saved model checkpoint to %s", model_path)

