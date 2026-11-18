import argparse
import csv
import os
import random

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data import ParallelTextDataset, collate_parallel_batch, make_masks
from evaluate import evaluate_bleu
from model.tf_model import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Transformer from scratch.")
    parser.add_argument("--epochs", type=int, default=config.epoch_num)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--learning-rate", type=float, default=config.lr)
    parser.add_argument("--max-len", type=int, default=config.max_len)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=config.MODEL_DIR)
    parser.add_argument(
        "--metrics-file",
        default=os.path.join(config.BASE_DIR, "weights/training_metrics.csv"),
        help="CSV file for per-epoch loss and periodic BLEU.",
    )
    parser.add_argument(
        "--bleu-every",
        type=int,
        default=5,
        help="Evaluate BLEU every N epochs; 0 disables BLEU evaluation.",
    )
    return parser.parse_args()


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    path: str,
    source_vocab_size: int,
    target_vocab_size: int,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_args": {
                "src_vocab": source_vocab_size,
                "tgt_vocab": target_vocab_size,
                "N": config.n_layers,
                "d_model": config.d_model,
                "d_ff": config.d_ff,
                "h": config.n_head,
                "dropout": config.dropout,
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source_tokenizer = spm.SentencePieceProcessor(model_file=config.VOCAB_EN)#type: ignore
    target_tokenizer = spm.SentencePieceProcessor(model_file=config.VOCAB_ZH)#type: ignore
    dataset = ParallelTextDataset(
        config.TRAIN_EN,
        config.TRAIN_ZH,
        source_tokenizer,
        target_tokenizer,
        args.max_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_parallel_batch(batch, config.padding_idx),
        pin_memory=config.DEVICE.type == "cuda",
    )

    model = make_model(
        source_tokenizer.vocab_size(),
        target_tokenizer.vocab_size(),
        config.n_layers,
        config.d_model,
        config.d_ff,
        config.n_head,
        config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=config.padding_idx)
    if args.bleu_every < 0:
        raise ValueError("--bleu-every must be non-negative")
    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_file)), exist_ok=True)
    with open(args.metrics_file, "w", newline="", encoding="utf-8") as metrics_file:
        metrics_writer = csv.DictWriter(
            metrics_file, fieldnames=["epoch", "loss", "bleu"]
        )
        metrics_writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            token_count = 0
            progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
            for source, target in progress:
                source = source.to(config.DEVICE)
                target = target.to(config.DEVICE)
                target_input, target_output = target[:, :-1], target[:, 1:]
                source_mask, target_mask = make_masks(
                    source, target_input, config.padding_idx
                )

                optimizer.zero_grad(set_to_none=True)
                hidden = model(source, target_input, source_mask, target_mask)
                logits = model.generator.proj(hidden)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), target_output.reshape(-1)
                )
                loss.backward()
                optimizer.step()

                non_padding = (target_output != config.padding_idx).sum().item()
                total_loss += loss.item() * non_padding
                token_count += non_padding
                progress.set_postfix(loss=f"{loss.item():.4f}")

            epoch_loss = total_loss / max(token_count, 1)
            bleu = None
            if args.bleu_every and epoch % args.bleu_every == 0:
                bleu = evaluate_bleu(
                    model,
                    source_tokenizer,
                    target_tokenizer,
                    max_len=args.max_len,
                )
                model.train()
                print(f"epoch {epoch}: BLEU={bleu:.2f}")

            save_checkpoint(
                model,
                optimizer,
                epoch,
                epoch_loss,
                args.output,
                source_tokenizer.vocab_size(),
                target_tokenizer.vocab_size(),
            )
            metrics_writer.writerow(
                {
                    "epoch": epoch,
                    "loss": f"{epoch_loss:.6f}",
                    "bleu": "" if bleu is None else f"{bleu:.6f}",
                }
            )
            metrics_file.flush()
            print(f"epoch {epoch}: loss={epoch_loss:.4f}, checkpoint={args.output}")


if __name__ == "__main__":
    main()
