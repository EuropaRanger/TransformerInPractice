import torch
import os

# Hyperparameters
# embedding size
d_model = 512
n_head = 8
n_layers = 6
# key dimension
d_k = d_model // n_head
# value dimension
d_v = d_model // n_head
# feed forward dimension
d_ff = 2048
dropout = 0.1

# vocab setting
src_vocab_size = 32000
tgt_vocab_size = 32000
padding_idx = 0
bos_idx = 2
eos_idx = 3

# training setting
batch_size = 48
# learning rate
lr = 3e-4
# training epochs
epoch_num = 3

# generator setting
max_len = 256
beam_size = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "raw_data/training-parallel-nc-v13")
TEST_DIR = os.path.join(BASE_DIR, "raw_data/test")
TRAIN_EN = os.path.join(DATA_DIR, "news-commentary-v13.zh-en.en")
TRAIN_ZH = os.path.join(DATA_DIR, "news-commentary-v13.zh-en.zh")
TEST_EN = os.path.join(TEST_DIR, "newstest2018-enzh-src.en.sgm")
TEST_ZH = os.path.join(TEST_DIR, "newstest2018-enzh-ref.zh.sgm")
VOCAB_EN = os.path.join(BASE_DIR, "tokenizer/en_spm.model")
VOCAB_ZH = os.path.join(BASE_DIR, "tokenizer/zh_spm.model")
MODEL_DIR = os.path.join(BASE_DIR, "weights/transformer.pt")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")