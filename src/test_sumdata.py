import json

from data_pipeline import SummarizationDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

import torch


def test_():
    dataset_name = "space"
    data_file = f"../data_backup/{dataset_name}/train.sum.jsonl"
    use_keywords = "input"
    use_switch = "input"
    batch_size = 16
    dataset = SummarizationDataset(
        data_file,
        use_keywords=use_keywords, use_switch=use_switch)

    for i, item in enumerate(dataset):
        print(item)
        break

    # src_data_list = []
    # with open(data_file, 'r') as d_file:
    #     for i in range(len):
    #         line = d_file.readline()
    #         inst = json.loads(line)
    #         print(inst['summary'])
    #         sums = [summary.lower() for summary in inst['summary']]
    #         src_data_list.append(sums)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    model_type = "t5-small"
    max_length = 512
    max_target_length = 128
    # for _, (inp_batch, out_batch, switch_batch) in enumerate(tqdm(dataloader)):
    #
    #     tokenizer = AutoTokenizer.from_pretrained(model_type)
    #     batch_encoding = tokenizer.prepare_seq2seq_batch(
    #         # src_texts=inp_batch,
    #         # tgt_texts=out_batch,
    #         src_texts=list(inp_batch),
    #         tgt_texts=list(out_batch),
    #         max_length=max_length,
    #         max_target_length=max_target_length,
    #         padding=True,
    #         truncation=True,
    #         return_tensors='pt'
    #     )
    #
    #     inp_ids = batch_encoding['input_ids'].cuda()
    #     inp_mask = batch_encoding['attention_mask'].cuda()
    #     out_ids = batch_encoding['labels'].cuda()
    #     out_mask = torch.where(out_ids == 0, 0, 1).unsqueeze(-1)  # batch_size, out_len
    #     out_ids[out_ids == 0] = -100
    #     # dec_inp_ids = model._shift_right(out_ids)
    #     print("")


if __name__ == "__main__":
    test_()




