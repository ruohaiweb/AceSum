import argparse

import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM as Model
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup

import os
from tqdm import tqdm

from calculate_rouge import calculate
from data_pipeline import SummarizationDataset


def get_data(data_file, len):

    use_keywords = "input"
    use_switch = "input"
    batch_size = 1
    dataset = SummarizationDataset(
        data_file,
        use_keywords=use_keywords, use_switch=use_switch)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    inp_batch_list = []
    for i, (inp_batch, out_batch, switch_batch) in enumerate(tqdm(dataloader)):
        if i == len:
            break
        inp_batch_list.append(inp_batch)


    src_data_list = []
    with open(data_file, 'r') as d_file:
        for i in range(len):
            line = d_file.readline()
            inst = json.loads(line)
            print(inst['summary'])
            sums = [summary.lower() for summary in inst['summary']]
            src_data_list.append(sums)

    return inp_batch_list, src_data_list

def sum_model(data_file):
    print()
    # 参数
    model_type = "t5-small"
    # load_model = "/data/webw3/code/acesum/model/space/sum.best.8000.15.746"
    load_model = "/data/webw3/code/acesum/model/space/sum.490000.3.15"
    num_aspects = 6
    use_switch = "input"

    no_warmup_steps = 20000
    no_train_steps = 500000

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    special_tokens = ['<rev>', '<key>', '<sum>', '<switch>']
    if use_switch != 'none':
        for i in range(num_aspects):
            special_tokens.append('<pos_%d>' % i)

    tokenizer.add_special_tokens(
        {'additional_special_tokens': special_tokens}
    )
    pad_id = tokenizer.pad_token_id

    # 加载预训练模型
    model = Model.from_pretrained(model_type, return_dict=True)
    model.resize_token_embeddings(len(tokenizer))
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, no_warmup_steps, no_train_steps)
    # 加载微调的参数
    best_point = torch.load(load_model)
    model.load_state_dict(best_point['model'])
    optimizer.load_state_dict(best_point['optimizer'])
    scheduler.load_state_dict(best_point['scheduler'])

    # 准备输入
    # 输入输出参数
    max_length = 512
    max_target_length = 128
    model.eval()
    #
    data_len = 22
    inp_batch_list, src_sum_list = get_data(data_file, data_len)
    result_sum_list = []
    rouge_scores = []
    for i, inp_batch in enumerate(inp_batch_list):
        batch_encoding = tokenizer.prepare_seq2seq_batch(
            src_texts=list(inp_batch),
            # tgt_texts=list(out_batch),
            max_length=max_length,
            max_target_length=max_target_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        inp_ids = batch_encoding['input_ids'].cuda()
        inp_mask = batch_encoding['attention_mask'].cuda()

        # 调用模型
        # 模型参数
        min_target_length = 15
        num_beams = 2
        no_repeat_ngram_size = 3
        repetition_penalty = 1
        length_penalty = 1
        preds = model.generate(
            inp_ids,
            decoder_start_token_id=0,
            min_length=min_target_length,
            max_length=max_target_length * 2,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )

        # 获取结果
        result = ""
        for pred in preds:
          result += tokenizer.decode(pred) + '\n'

        asp_pred_sums = result
        # 评价结果
        # f = open(data_file, 'r')
        # lines = f.readlines()
        result_sum_list.append(result)
        asp_scores = calculate(src_sum_list[i], asp_pred_sums)
        print(f"第{i}条：")
        print(f"rouge: {asp_scores}")
        print(f"src: {src_sum_list[i]}")
        print(f"result: {result}")



        rouge_scores += list(asp_scores)

    # rouge = np.power(np.product(rouge_scores), 1.0 / len(rouge_scores))
    # print(rouge)
    rouge_12l = calculate(src_sum_list[:data_len], result_sum_list[:data_len])
    print(rouge_12l)



if __name__ == "__main__":
    dataset_name = "space"
    # data_file = f"../data_backup/{dataset_name}/test.sum.aspect.jsonl" # (14.246443458128265, 1.5005405405405408, 11.169449304279452)
    data_file = f"../data_backup/{dataset_name}/dev.sum.general.jsonl" # (30.937898120903075, 7.134746469885761, 21.068277668782134)
    sum_model(data_file)

