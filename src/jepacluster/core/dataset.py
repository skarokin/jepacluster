"""
This file contains the dataset pipeline: log-to-tensor transformation.
"""

from collections import Counter
import re
from pathlib import Path
from typing import Dict, Iterator, List

import math

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

SentencePieceTokenizer = None
try:
    from torchtext.transforms import SentencePieceTokenizer as _SentencePieceTokenizer  # type: ignore
    SentencePieceTokenizer = _SentencePieceTokenizer
except Exception:  # pragma: no cover - optional dependency
    SentencePieceTokenizer = None


class LogMaskDataset(IterableDataset):
    """
    Loads fixed-length log windows and returns token-level context/target masks.
    Each item is a dict that mirrors the JEPA style of masked prediction.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int,
        num_context_tokens: int | None = None,
        num_target_tokens: int = 3,
        min_freq: int = 2,
        max_vocab_size: int = 50000,
        tokenizer_path: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.num_context_tokens = num_context_tokens or max(1, window_size - num_target_tokens)
        self.num_target_tokens = num_target_tokens
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.tokenizer_path = tokenizer_path
        self.source_patterns = ("*.txt", "*.log")
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.token_to_id: Dict[str, int] = {self.pad_token: 0, self.unk_token: 1}
        self._build_vocab()
        self.sp_model = self._load_sentencepiece()

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _build_vocab(self) -> None:
        # builds a vocabulary from the training logs.
        counter: Counter[str] = Counter()
        if not self.data_dir.exists():
            return

        log_files = list(self._iter_log_files())
        for idx, path in enumerate(log_files, start=1):
            if idx == 1 or idx % 25 == 0 or idx == len(log_files):
                print(f"[dataset] building vocab: {idx}/{len(log_files)} files")
            text = path.read_text(encoding="utf-8")
            for token in self._tokenize(text):
                counter[token] += 1

        self.token_to_id[self.bos_token] = len(self.token_to_id)
        self.token_to_id[self.eos_token] = len(self.token_to_id)

        for token, count in counter.most_common(self.max_vocab_size):
            if count < self.min_freq:
                continue
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)

    def _load_sentencepiece(self):
        # if SentencePiece model is already present, use it.
        # otherwise we fall back to our learned vocabulary.
        if self.tokenizer_path and SentencePieceTokenizer is not None and Path(self.tokenizer_path).exists():
            return SentencePieceTokenizer(self.tokenizer_path)
        return None

    def _tokenize(self, text: str) -> List[str]:
        # splits on words, numbers, and punctuation so logs keep some structure
        return re.findall(r"[A-Za-z_]+|\d+|[^\sA-Za-z_\d]", text)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if not self.data_dir.exists():
            return

        worker_info = get_worker_info()
        log_files = self._iter_log_files()
        if worker_info is None:
            start_idx, end_idx = 0, len(log_files)
        else:
            per_worker = int(math.ceil(len(log_files) / worker_info.num_workers))
            start_idx = worker_info.id * per_worker
            end_idx = min(start_idx + per_worker, len(log_files))

        shard = log_files[start_idx:end_idx]
        for file_idx, path in enumerate(shard, start=1):
            if file_idx == 1 or file_idx % 25 == 0 or file_idx == len(shard):
                print(f"[dataset] streaming samples: file {file_idx}/{len(shard)}")
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) < self.window_size:
                print(f"[dataset] skipping short file: {path.name} lines={len(lines)}")
                continue

            for idx in range(len(lines) - self.window_size + 1):
                window = lines[idx : idx + self.window_size]
                token_tensor, token_mask = self._encode_window(window)
                for context_mask, target_mask in self._build_masks():
                    yield {
                        "tokens": token_tensor,
                        "token_mask": token_mask,
                        "context_mask": context_mask,
                        "target_mask": target_mask,
                    }

    def _iter_log_files(self) -> List[Path]:
        # Accept both .txt and .log so the dataset works with plain text logs and loghub-style files.
        seen = set()
        paths: List[Path] = []
        for pattern in self.source_patterns:
            for path in self.data_dir.rglob(pattern):
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
        return paths

    def _build_masks(self) -> List[tuple[torch.Tensor, torch.Tensor]]:
        # context tokens are the visible part of the window
        # target tokens are the hidden part the predictor must infer
        # we sample several spans so the model does not only learn one fixed prediction pattern
        mask_pairs: List[tuple[torch.Tensor, torch.Tensor]] = []
        span_len = max(1, self.num_target_tokens)
        stride = max(1, (self.window_size - span_len) // 3)

        for start in range(0, self.window_size - span_len + 1, stride):
            context_mask = torch.ones(self.window_size, dtype=torch.bool)
            target_mask = torch.zeros(self.window_size, dtype=torch.bool)
            context_mask[start : start + span_len] = False
            target_mask[start : start + span_len] = True
            mask_pairs.append((context_mask, target_mask))

        if not mask_pairs:
            context_mask = torch.ones(self.window_size, dtype=torch.bool)
            target_mask = torch.zeros(self.window_size, dtype=torch.bool)
            context_mask[-span_len:] = False
            target_mask[-span_len:] = True
            mask_pairs.append((context_mask, target_mask))

        return mask_pairs

    def _encode_window(self, window: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_lines = [self._encode_line(line) for line in window]
        token_tensor = torch.stack(encoded_lines, dim=0)
        token_mask = token_tensor.ne(self.token_to_id[self.pad_token])
        return token_tensor, token_mask

    def _encode_line(self, line: str) -> torch.Tensor:
        # tokenizes a log line into ids
        # if SentencePiece is available and a model path is provided, we use that instead of the fallback regex split
        if self.sp_model is not None:
            pieces = self.sp_model(line)
        else:
            pieces = self._tokenize(line)

        token_ids = [self.token_to_id[self.bos_token]]
        token_ids.extend(self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in pieces[: self.window_size - 2])
        token_ids.append(self.token_to_id[self.eos_token])
        if len(token_ids) < self.window_size:
            token_ids.extend([self.token_to_id[self.pad_token]] * (self.window_size - len(token_ids)))
        return torch.tensor(token_ids[: self.window_size], dtype=torch.long)

    def __len__(self) -> int:
        # IterableDataset does not have a stable finite length, but returning 1 prevents code paths
        # that expect len(dataset) from failing outright.
        return 1


def collate_jepa_batches(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Stack a list of dataset items into one batch.

    We keep this separate because JEPA wants several masks per batch element.
    """

    tokens = torch.stack([item["tokens"] for item in batch], dim=0)
    token_mask = torch.stack([item["token_mask"] for item in batch], dim=0)
    context_mask = torch.stack([item["context_mask"] for item in batch], dim=0)
    target_mask = torch.stack([item["target_mask"] for item in batch], dim=0)
    return {
        "tokens": tokens,
        "token_mask": token_mask,
        "context_mask": context_mask,
        "target_mask": target_mask,
    }


class MaskSampler:
    """
    Simple multi-mask sampler.

    The official JEPA code uses dedicated mask collators. This keeps that idea by allowing one log window
    to produce several masked prediction setups.
    """

    def __init__(self, window_size: int, num_context_tokens: int, num_target_tokens: int, num_masks: int = 4):
        self.window_size = window_size
        self.num_context_tokens = num_context_tokens
        self.num_target_tokens = num_target_tokens
        self.num_masks = num_masks

    def sample(self) -> List[Dict[str, torch.Tensor]]:
        masks: List[Dict[str, torch.Tensor]] = []
        span_lengths = sorted(
            {
                max(1, self.num_target_tokens),
                max(1, self.num_target_tokens + 1),
                max(1, min(self.window_size // 3, self.num_target_tokens + 2)),
            }
        )

        for span_len in span_lengths:
            max_start = max(1, self.window_size - span_len + 1)
            for i in range(self.num_masks):
                target_start = (i * max_start) // self.num_masks
                context_mask = torch.ones(self.window_size, dtype=torch.bool)
                target_mask = torch.zeros(self.window_size, dtype=torch.bool)

                # different masks hide different spans and positions
                context_mask[target_start : target_start + span_len] = False
                target_mask[target_start : target_start + span_len] = True

                # keep at least some context visible
                if not context_mask.any():
                    context_mask[: self.num_context_tokens] = True
                    target_mask[: self.num_target_tokens] = True
                    context_mask[: self.num_target_tokens] = False

                masks.append({"context_mask": context_mask, "target_mask": target_mask})

        return masks
