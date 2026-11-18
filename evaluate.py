from pathlib import Path

import sacrebleu
import sentencepiece as spm
import torch

import config
from data import clean_sgm_line
from infer import translate


def read_sgm_segments(path: str) -> list[str]:
    segments = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "<seg " in line:
            text = clean_sgm_line(line)
            if text:
                segments.append(text)
    return segments


@torch.no_grad()
def evaluate_bleu(
    model: torch.nn.Module,
    source_tokenizer: spm.SentencePieceProcessor,
    target_tokenizer: spm.SentencePieceProcessor,
    source_path: str = config.TEST_EN,
    target_path: str = config.TEST_ZH,
    max_len: int = config.max_len,
) -> float:
    sources = read_sgm_segments(source_path)
    references = read_sgm_segments(target_path)
    if len(sources) != len(references):
        raise ValueError(
            f"Evaluation files have different segment counts: "
            f"{len(sources)} != {len(references)}"
        )

    model.eval()
    predictions = [
        translate(model, source, source_tokenizer, target_tokenizer, max_len)
        for source in sources
    ]
    return float(sacrebleu.corpus_bleu(predictions, [references]).score)
