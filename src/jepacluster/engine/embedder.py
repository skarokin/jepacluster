"""
Inference-time embedding pipeline for JEPA log windows.
"""

from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from core.architecture import JEPAArchitecture
from core.dataset import LogMaskDataset, collate_jepa_batches
from utils.config import ModelConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model_config = ModelConfig.model_validate(checkpoint["model_config"])

        vocab_size = checkpoint.get("vocab_size")
        if vocab_size is None:
            raise ValueError("Checkpoint is missing vocab_size, cannot build inference model.")

        self.model = JEPAArchitecture(self.model_config, vocab_size=vocab_size).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def embed(self, infer_path: str) -> tuple[torch.Tensor, list[str]]:
        infer_path_obj = Path(infer_path)
        if not infer_path_obj.exists():
            raise FileNotFoundError(f"Inference path does not exist: {infer_path}")

        dataset = LogMaskDataset(
            infer_path,
            window_size=self.model_config.window_size,
            min_freq=1,
            max_vocab_size=self.model_config.vocab_size or 50000,
            tokenizer_path=self.model_config.tokenizer_path,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_jepa_batches,
            num_workers=0,
        )

        embeddings: List[torch.Tensor] = []
        samples: list[str] = []
        with torch.no_grad():
            for batch in dataloader:
                tokens = batch["tokens"].to(self.device)
                context_mask = batch["context_mask"].to(self.device)
                encoded = self.model.encode_context(tokens, mask=context_mask)
                pooled = encoded.mean(dim=1).detach().cpu()
                embeddings.append(pooled)
                for window in batch.get("window_text", []):
                    samples.append("\n".join(window))

        if not embeddings:
            return torch.empty(0, self.model_config.latent_dim), []

        result = torch.cat(embeddings, dim=0)
        logger.info("Generated %d embeddings with dim=%d", result.shape[0], result.shape[1])
        return result, samples
