import argparse

import sentencepiece as spm
import torch

import config
from data import clean_sgm_line, make_masks
from model.tf_model import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate text with a trained Transformer.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="One source sentence.")
    group.add_argument("--input", help="A UTF-8 text/SGM file, one sentence per line.")
    parser.add_argument("--checkpoint", default=config.MODEL_DIR)
    parser.add_argument("--max-len", type=int, default=config.max_len)
    group.add_argument(
        "--interactive",
        action="store_true",
        help="Enter source sentences repeatedly; enter exit, quit, or q to stop.",
    )
    return parser.parse_args()


def load_model(path: str) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=config.DEVICE)
    model_args = checkpoint.get("model_args", {})
    model = make_model(**model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def translate(
    model: torch.nn.Module,
    text: str,
    source_tokenizer: spm.SentencePieceProcessor,
    target_tokenizer: spm.SentencePieceProcessor,
    max_len: int,
) -> str:
    source_ids = [source_tokenizer.bos_id(), *source_tokenizer.EncodeAsIds(text)[: max_len - 2], source_tokenizer.eos_id()]
    source = torch.tensor([source_ids], dtype=torch.long, device=config.DEVICE)
    memory = model.encode(source, (source != config.padding_idx).unsqueeze(1))#type: ignore
    generated = [target_tokenizer.bos_id()]
    for _ in range(max_len - 1):
        target = torch.tensor([generated], dtype=torch.long, device=config.DEVICE)
        _, target_mask = make_masks(source, target, config.padding_idx)
        hidden = model.decode(
            memory, (source != config.padding_idx).unsqueeze(1), target, target_mask
        )#type: ignore
        next_id = model.generator.proj(hidden[:, -1]).argmax(dim=-1).item()#type: ignore
        generated.append(next_id)
        if next_id == target_tokenizer.eos_id():
            break
    return target_tokenizer.Decode([token for token in generated[1:] if token != target_tokenizer.eos_id()])


def main() -> None:
    args = parse_args()
    source_tokenizer = spm.SentencePieceProcessor(model_file=config.VOCAB_EN)#type: ignore
    target_tokenizer = spm.SentencePieceProcessor(model_file=config.VOCAB_ZH)#type: ignore
    model = load_model(args.checkpoint)

    if args.interactive:
        print("Enter a sentence to translate. Enter exit, quit, or q to stop.")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if line.lower() in {"exit", "quit", "q", "/exit", "/quit"}:
                break
            if line:
                print(translate(model, line, source_tokenizer, target_tokenizer, args.max_len))
        return

    lines = [args.text] if args.text is not None else open(args.input, encoding="utf-8")
    try:
        for line in lines:
            text = clean_sgm_line(line) if args.input else line.strip()
            if text:
                print(translate(model, text, source_tokenizer, target_tokenizer, args.max_len))
    finally:
        if args.input:
            lines.close()#type: ignore


if __name__ == "__main__":
    main()
