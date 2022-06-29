import json
import numpy as np
import os
from tqdm import tqdm
import random


def create_data(filedir, dest_filedir=None, seeds_filedir=None):
    print(filedir)

    # get aspects and keywords
    if seeds_filedir == None:
        seeds_filedir = 'seeds/' + filedir[5:]
    files = os.listdir(seeds_filedir)
    keywords_dict = {}
    for file in files:
        f = open(seeds_filedir + '/' + file, 'r')
        keywords = []
        lines = f.readlines()
        for line in lines:
            if len(line.strip().split()) == 0:
                continue
            keyword = line.strip().split()[-1]
            keywords.append(keyword)

        f.close()
        aspect = file.replace('.txt', '')
        keywords_dict[aspect] = keywords

    aspects = list(keywords_dict.keys())  # + ['general']

    instance_dict = {}
    f = open(filedir, 'r')
    for line in tqdm(f):
        inst = json.loads(line.strip())

        domain = 'space'
        if domain not in instance_dict:
            instance_dict[domain] = {}

        reviews = inst['reviews']
        for review in reviews:
            sentences = review['sentences']

            # sanity check
            if len(sentences) > 35 or len(sentences) < 1:
                print("len(sentences) > 35 or len(sentences) < 1")
                continue
            sentences_new = []
            for sentence in sentences:
                if len(sentence.split()) <= 35:
                    sentences_new.append(sentence)
            sentences = sentences_new
            if max([len(sentence.split()) for sentence in sentences]) > 35:
                print("最大字数超过了35 max([len(sentence.split()) for sentence in sentences]) > 35")
                continue

            review2 = ' '.join(sentences).split()
            review = ' '.join(sentences)
            # check whether aspect keywords in review
            class_list = []
            for aspect in aspects:
                keywords = keywords_dict[aspect]
                includes = int(any([keyword in review for keyword in keywords]))
                # includes = 0
                # for keyword in keywords:
                #     # for review in
                #     if keyword in review:
                #         includes = 1
                class_list.append(includes)
            # assert len(class_list) == 3

            # add review to corresponding aspect buckets
            instance_tuple = (sentences, class_list)
            class_list = tuple(class_list)
            # print(f"class_list:{class_list}")
            if class_list not in instance_dict[domain]:
                instance_dict[domain][class_list] = []
            instance_dict[domain][class_list].append(instance_tuple)
            print(instance_tuple)

    f.close()
    # print(f"instance_dict:{instance_dict}")
    for domain in instance_dict:
        print('domain', domain)
        for key in instance_dict[domain]:
            print(f"key:{key}  length:{len(instance_dict[domain][key])}")
        lengths = [len(instance_dict[domain][key]) for key in instance_dict[domain]]
        # print(lengths)
        min_length = sorted(lengths)[1]

        for i in range(len(aspects)):
            c = [0] * len(aspects)
            c[i] = 1
            # print(c)
            if tuple(c) in instance_dict[domain]:
                print(len(instance_dict[domain][tuple(c)]))
            else:
                print(0)

        print('mininum instances per tuple', min_length)

        data = []
        for key in instance_dict[domain]:
            instances = instance_dict[domain][key]
            random.shuffle(instances)
            data += instances[:min_length]

        print('total data', len(data))
        random.shuffle(data)
        max_text_length = 0

        domain_aspects = aspects
        if dest_filedir == None:
            dest_filedir = filedir
        if not os.path.exists(os.path.dirname(dest_filedir)):
            os.makedirs(dest_filedir)
        f = open(dest_filedir, 'w')
        count_dict = {aspect: 0 for aspect in domain_aspects}
        for inst in data:
            new_inst = {}
            new_inst['review'] = inst[0]
            max_text_length = max(max_text_length, len(inst[0]))
            class_dict = {}

            for i, aspect in enumerate(domain_aspects):
                class_dict[aspect] = 'yes' if inst[1][i] else 'no'
                if inst[1][i]:
                    count_dict[aspect] += 1
            new_inst['aspects'] = class_dict
            f.write(json.dumps(new_inst, ensure_ascii=False) + '\n')

        f.close()

        print('max text length', max_text_length)
        # print(count_dict)


import sys
if __name__ == "__main__":
    # filedir = sys.argv[1]
    # seeds_filedir = sys.argv[2]
    # src_filedir = "../data/amusement/train.jsonl"
    # dest_filedir = "../data/amusement/train.mil.jsonl"
    # seeds_filedir = "../seeds/amusement"
    src_filedir = "../data/park/train.jsonl"
    dest_filedir = "../data/park/train.mil.jsonl"
    seeds_filedir = "../seeds/park"
    create_data(src_filedir, dest_filedir, seeds_filedir)

    # create_data('data/oposum/bag')
    # create_data('data/oposum/boots')
    # create_data('data/oposum/bt')
    # create_data('data/oposum/keyboard')
    # create_data('data/oposum/tv')
    # create_data('data/oposum/vacuum')
    # create_data('data/space/')
