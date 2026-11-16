import sentencepiece as spm
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EN_IN = os.path.join(BASE_DIR, "../raw_data/training-parallel-nc-v13/news-commentary-v13.zh-en.en")
ZH_IN = os.path.join(BASE_DIR, "../raw_data/training-parallel-nc-v13/news-commentary-v13.zh-en.zh")
EN_CHARACTER_COVERAGE = 1.0
ZH_CHARACTER_COVERAGE = 0.9995
EN_MODEL_NAME = os.path.join(BASE_DIR, "en_spm")
ZH_MODEL_NAME = os.path.join(BASE_DIR, "zh_spm")
MODEL_TYPE = "bpe"
VOCAB_SIZE = 32000

def train(input_file,vocab_size,model_name,model_type,character_coverage):
    input_argument = (
        '--input=%s '
        '--model_prefix=%s '
        '--vocab_size=%s '
        '--model_type=%s '
        '--character_coverage=%s '
        '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
    )

    cmd = input_argument % (input_file,model_name,vocab_size,model_type,character_coverage)

    spm.SentencePieceTrainer.Train(cmd)

def run():
    train(EN_IN,VOCAB_SIZE,EN_MODEL_NAME,MODEL_TYPE,EN_CHARACTER_COVERAGE)
    train(ZH_IN,VOCAB_SIZE,ZH_MODEL_NAME,MODEL_TYPE,ZH_CHARACTER_COVERAGE)

def test():
    sp = spm.SentencePieceProcessor()
    text = "我爱自然语言处理"

    sp.Load(ZH_MODEL_NAME + ".model")
    print(sp.EncodeAsPieces(text))
    print(sp.EncodeAsIds(text))

if __name__ == "__main__":
    if not os.path.exists(EN_MODEL_NAME + ".model") or not os.path.exists(ZH_MODEL_NAME + ".model"):
        run()
    test()