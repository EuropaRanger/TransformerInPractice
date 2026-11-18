import re
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import Dataset


class ParallelTextDataset(Dataset):
    def __init__(
        self,
        source_path: str,
        target_path: str,
        source_tokenizer: spm.SentencePieceProcessor,
        target_tokenizer: spm.SentencePieceProcessor,
        max_len: int,
    ):
        self.source_lines = self._read_lines(source_path)
        self.target_lines = self._read_lines(target_path)
        if len(self.source_lines) != len(self.target_lines):
            raise ValueError(
                f"Parallel files have different lengths: "
                f"{len(self.source_lines)} != {len(self.target_lines)}"
            )
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.source_lines)

    def _encode(self, text: str, tokenizer: spm.SentencePieceProcessor) -> list[int]:
        ids = tokenizer.EncodeAsIds(text)
        return [tokenizer.bos_id(), *ids[: self.max_len - 2], tokenizer.eos_id()]

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return (
            self._encode(self.source_lines[index], self.source_tokenizer),
            self._encode(self.target_lines[index], self.target_tokenizer),
        )
    @staticmethod
    def _read_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f] 


def collate_parallel_batch(
    batch: list[tuple[list[int], list[int]]], padding_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    source_length = max(len(source) for source, _ in batch)
    target_length = max(len(target) for _, target in batch)
    source = torch.full((len(batch), source_length), padding_idx, dtype=torch.long)
    target = torch.full((len(batch), target_length), padding_idx, dtype=torch.long)
    for row, (source_ids, target_ids) in enumerate(batch):
        source[row, : len(source_ids)] = torch.tensor(source_ids)
        target[row, : len(target_ids)] = torch.tensor(target_ids)
    return source, target


def subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones((1, size, size), dtype=torch.bool, device=device))


def make_masks(
    source: torch.Tensor, target_input: torch.Tensor, padding_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    source_mask = (source != padding_idx).unsqueeze(1)
    target_padding_mask = (target_input != padding_idx).unsqueeze(1)
    target_mask = target_padding_mask & subsequent_mask(
        target_input.size(1), target_input.device
    )
    return source_mask, target_mask


def clean_sgm_line(line: str) -> str:
    return re.sub(r"<[^>]+>", "", line).strip()
