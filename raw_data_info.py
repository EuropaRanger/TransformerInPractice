import os

def get_corpus_info(path):
    if not os.path.exists(path):
        print("No such file!")
        return
    total = 0
    lines_with_fffd = 0
    with open(path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0,len(lines)):
            n = lines[i].count("\ufffd")
            if n:
                total += n
                lines_with_fffd += 1
                # 可选：打印前几行看看
                if lines_with_fffd <= 5:
                    print(i, repr(lines[i][:120]))
    print(f"lines:{len(lines)}")
    print("U+FFFD 总数:", total)
    print("含 U+FFFD 的行数:", lines_with_fffd)

if __name__ =="__main__":
    get_corpus_info("./raw_data/training-parallel-nc-v13/news-commentary-v13.zh-en.zh")
    get_corpus_info("./raw_data/training-parallel-nc-v13/news-commentary-v13.zh-en.en")
