# 创建原始数据集，从文本转换成json
'''
读取文件，每500个为一组，写到一个json中。
{
    "id": "2515499",
    "reviews": [
        {
            "sentences": [
            ],
            "rating": 3
        },
        {
            "sentences": [

            ],
            "rating": 5
        },
    ]
}
'''
import json

from tqdm import tqdm
import re


def create_data(filedir, dest_filedir):
    with open(filedir, 'r') as origin_file:
        lines = origin_file.readlines()
        product_review_list = []
        for i, line in tqdm(enumerate(lines), total=len(lines)):
            sentences = re.split(r'[？！；。.\s]\s*', line)
            # sentences = re.split(r'[;。.]', line)
            sentences_new = []

            curr_sentence = ""
            for sentence in sentences:
                curr_sentence += f" {sentence}"
                if len(curr_sentence) > 20:
                    sentences_new.append(curr_sentence)
                    curr_sentence = ""
            if len(sentences_new) > 25:
                sentences_new = sentences_new[:25]
            if len(sentences_new) < 2:
                print("len(sentences_new) < 2")
                continue

            review = {"sentences": sentences_new}
            if i % 10 == 0:
                if i > 1:
                    with open(dest_filedir, 'a', encoding='utf8') as dest_file:
                        dest_file.write(json.dumps(product_review, ensure_ascii=False))
                        dest_file.write("\n")
                product_review = {}
                product_review["id"] = f"{i}"
                reviews = []
                product_review["reviews"] = reviews
                product_review_list.append(product_review)
            reviews.append(review)
            # if i % 500 == 0 and i>0:

            # if i>2000:
            #     break
        # for product_review in product_review_list:

def deal_aspect():
    with open("../data/park_abae_aspect.txt", 'r') as seed_file:
        lines = seed_file.readlines()
        for line in lines:
            words = line.split("/")
            with open(f"../seeds/park/{words[0]}.txt", 'w') as w_seed_file:
                for word in words:
                    w_seed_file.write(word+"\n")


if __name__ == "__main__":
    # src_filedir = "../data/park/reviews.txt"
    # dest_filedir = "../data/park/train.jsonl"
    # create_data(src_filedir, dest_filedir)
    deal_aspect()
