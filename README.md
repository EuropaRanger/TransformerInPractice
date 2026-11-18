# TransformerInPractice
Self implemented transformer

## Train

Activate the prepared environment and run training from the repository root:

```bash
conda activate tfp
python train.py
```

The script reads the parallel English/Chinese corpus and SentencePiece models
from `config.py`, trains from randomly initialized weights with token-level
cross entropy (padding tokens are ignored), writes `weights/transformer.pt`
after each epoch, and records metrics in
`weights/training_metrics.csv`. Loss is recorded every epoch; BLEU is
calculated on the configured newstest2018 English-Chinese test set every five
epochs by default. Common settings can be overridden,
for example:

```bash
python train.py --epochs 10 --batch-size 48 --learning-rate 3e-4
```

Change the evaluation interval or disable BLEU evaluation:

```bash
python train.py --epochs 20 --bleu-every 5
python train.py --epochs 3 --bleu-every 0
```

## Inference

Translate one sentence using a trained checkpoint:

```bash
python infer.py --text "Hello world"
```

To translate a UTF-8 text or SGM file one sentence per line:

```bash
python infer.py --input raw_data/test/newstest2018-enzh-src.en.sgm
```

Use `--checkpoint` to load a checkpoint at another path.
