import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data_pipeline import AspectDetectionDataset, aspect_detection_collate


def test_mil_dataset():
    # args = get_args()
    # print(args)
    args = {
        "model_type":"distilroberta-base",
        "data_dir":"../data",
        "dataset":"space",
        "train_file":"train.mil.jsonl",
        "batch_size":32
    }
    args = ObjDict(args)
    print('Preparing data...')

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    dataset = AspectDetectionDataset(
        args.data_dir + '/' + args.dataset + '/' + args.train_file, tokenizer)

    with open(dataset.file, "r") as input_file:
        # lines = input_file.readlines()
        result = dataset.process(next(input_file))
        print(result)


    # dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=aspect_detection_collate)
    # for _, (inp_batch, out_batch) in enumerate(tqdm(dataloader)):
    #     # print(inp_batch)
    #     # print(out_batch)
    #     B, S, T = inp_batch.size()
    #     print(inp_batch.size())
    #     print(B)
    #     print(S)
    #     print(T)
    #     x_BSxT = inp_batch.view(B * S, T)
    #     print(x_BSxT  )
    #     break


class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value
