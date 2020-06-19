import pandas as pd
import os
from transformers import BertTokenizer
import numpy as np
import math
import unicodedata
import re
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import json

all_tag = ['調達年度', '都道府県', '入札件名', '施設名', '需要場所(住所)', \
            '調達開始日', '調達終了日', '公告日', '仕様書交付期限', '質問票締切日時', \
            '資格申請締切日時', '入札書締切日時', '開札日時', '質問箇所所属/担当者', '質問箇所TEL/FAX', \
            '資格申請送付先', '資格申請送付先部署/担当者名', '入札書送付先', '入札書送付先部署/担当者名', '開札場所']
tag_to_label = {text: i for i, text in enumerate(all_tag)}
label_to_tag = {i: text for i, text in enumerate(all_tag)}
translate = lambda x : ''.join(i.strip('#') for i in x)

never_split_list = []
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenizer.add_tokens(['～', 'Fax','１','２','３','４','５','６','７','８','９','０', 'ＦＡＸ', 'ＴＥＬ'])

# [CLS] : 101
# [SEP] : 102
# [PAD] : 0

with open("./config.json") as f:
    config = json.load(f)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, data):
        starts = []
        ends = []
        tokens = []
        token_types = []
        tags = []
        masks = []
        pdf_ids = []
        sen_ids = []
        sen_cls = []
        if not data[0].get('start') == None:
            for i in range(len(data)):
                starts.append(data[i]['start'])
                ends.append(data[i]['end'])
                tokens.append(data[i]['token'])
                token_types.append(data[i]['token_type'])
                masks.append(data[i]['mask'])
                sen_cls.append(data[i]['sen_cls'])
            return {
                'token':torch.tensor(tokens),
                'token_type':torch.tensor(token_types),
                'mask':torch.tensor(masks),
                'sen_cls' : sen_cls,
                'start':torch.tensor(starts),
                'end':torch.tensor(ends),
            }
        else:
            for i in range(len(data)):
                tokens.append(data[i]['token'])
                token_types.append(data[i]['token_type'])
                masks.append(data[i]['mask'])
                tags.append(data[i]['tag'])
                pdf_ids.append(data[i]['pdf_id'])
                sen_ids.append(data[i]['sen_id'])
                sen_cls.append(data[i]['sen_cls'])
            return {
                'token':torch.tensor(tokens),
                'token_type':torch.tensor(token_types),
                'mask':torch.tensor(masks),
                'sen_cls' : sen_cls,
                'tag':tags,
                'pdf_id':pdf_ids,
                'sen_id':sen_ids
            }

def find_position(context, value):
    context_len = len(context)
    value_len = len(value)
    for start in range(context_len - value_len + 1):
        for end in range(start + value_len - 10, start + value_len + 10):
            if end <= start or end > context_len:
                continue
            a = [context[j].strip('#') for j in range(start, end)]
            b = [value[j].strip('#') for j in range(value_len)]
            if ''.join(b) == ''.join(a):
                return start, end - 1

def prepare_tags_and_values(tags, values):
    if type(tags) != str:
        return [] , []
    else:
        tags = tags.split(';')
        values = values.split(';')
        if len(tags) > 1 and len(values) == 1:
            for i in range(len(tags) - len(values)):
                values.append(values[0])
        elif len(tags) == 1 and len(values) > 1:
            for i in range(len(values) - len(tags)):
                tags.append(tags[0])
        for i in range(len(tags)):
            tags[i] = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tags[i]))
            tags[i] = tokenizer.tokenize(tags[i])
            values[i] = tokenizer.tokenize(values[i])
        return tags, values

def pad_to_len(seq, padding, to_len):
    padded = seq + [padding] * max(0, to_len - len(seq))
    return padded

def make_data(data_path, mode):
    files = os.listdir(data_path)
    files.sort()
    raw_datas = []
    for file_name in files:
        data = pd.read_excel(data_path+file_name, encoding = 'big5')
        raw_data = data.to_numpy()
        raw_datas.append(raw_data)

    datas = []
    print('start loading data...')

    if mode != 'test':

        for i, data in tqdm(enumerate(raw_datas), total=len(raw_datas)):
            for data_line in data:
                if type(data_line[6]) == str and type(data_line[7]) != str:
                    continue
                texts = tokenizer.tokenize(data_line[1])
                tags, values = prepare_tags_and_values(data_line[6], data_line[7])
                assert len(tags) == len(values)
                if len(datas) == 0 or len(datas[-1]['text']) + len(texts) + 1> config['text_max_len'] - 2 or files[i].split('.')[0] != datas[-1]['pdf_id']:
                    datas.append({'text' : ['[CLS]', '[SEP]', '[CLS]'] + texts + ['[SEP]'],
                                  'tag' : tags, 'value' : values, 'pdf_id' : files[i].split('.')[0], 'sen_cls': [(2, len(texts) + 3)]})
                else:
                    datas[-1]['sen_cls'].append((len(datas[-1]['text']), len(datas[-1]['text']) + len(texts) + 1))
                    datas[-1]['text'] += ['[CLS]'] + texts + ['[SEP]']
                    datas[-1]['tag'] += tags
                    datas[-1]['value'] += values

        tokenized_all_tag = [tokenizer.tokenize(tag) for tag in all_tag]
        dataset = []

        print('start processing data...')
        for data in tqdm(datas):

            #first, handle each tag-value pair in a dictionary: tag -> [(start, end)]
            tag_pos_dict = {i: {} for i in range(20)}
            sen_cls = data['sen_cls']

            for tag, value in zip(data['tag'], data['value']):

                tag_pos = find_position(data['text'], value)
                if tag_pos == None:
                    continue

                #find the sentence of pos
                for i, (sen_start, sen_end) in enumerate(sen_cls):
                    if sen_start <= tag_pos[0] and tag_pos[1] <= sen_end:
                        sen_pos = i
                        break
                    #print(sen_start, sen_end, data['text'][sen_start: sen_end+1])

                tag_pos_dict[tag_to_label[translate(tag)]][i] = tag_pos
                #print(data['text'][pos[0]:pos[1] + 1])


            #first, pad text to text_max_len
            text = pad_to_len(data['text'], '[PAD]', config['text_max_len'])

            #then iterate through each tag type, create a sample per tag type
            for tag_index in range(20):

                len_tag = len(tokenized_all_tag[tag_index]) + 1
                token_type = pad_to_len([0] * config['text_max_len'] + [1] * len_tag, 0, 512)

                # calculate mask, pad (token + tag)
                token = text + tokenized_all_tag[tag_index] + ['[SEP]'] # >= text_max_len: tag part
                token = pad_to_len(tokenizer.convert_tokens_to_ids(token), 0, 512)
                mask = pad_to_len([1] * len(data['text']) + [0] * (config['text_max_len'] - len(data['text'])) + len_tag * [1], 0, 512)

                #for a, b, c in zip(token, mask, token_type):
                #    print(tokenizer.convert_ids_to_tokens(a), b, c)

                start = [0. for _ in range(512)]
                end = [0. for _ in range(512)]
                #find the sentence of pos
                for i, (sen_start, sen_end) in enumerate(sen_cls):
                    #print(sen_start, sen_end, data['text'][sen_start: sen_end+1])
                    if i in tag_pos_dict[tag_index]: #we have answer in this sentence
                        ans_start, ans_end = tag_pos_dict[tag_index][i]
                        start[ans_start] = 1.
                        end[ans_end] = 1.
                    else: #we have no answer in this sentence
                        start[sen_start] = 1.
                        end[sen_start] = 1.

                context = tokenizer.convert_ids_to_tokens(token)

                if len(tag_pos_dict[tag_index]):
                    """
                    print('========================= tag: {} ==============================='.format(label_to_tag[tag_index]))
                    for i in range(0, 512, 10):
                        print(context[i:i+10])
                        print(start[i:i+10])
                        print(end[i: i+10])
                    """
                    dataset.append({
                        'token' : token,
                        'token_type' : token_type,
                        'mask' : mask,
                        'sen_cls' : sen_cls,
                        'start' : start,
                        'end' : end,
                    })
        return dataset

    else: #testing mode
        for i, data in tqdm(enumerate(raw_datas), total=len(raw_datas)):
            for j, data_line in enumerate(data):
                texts = tokenizer.tokenize(data_line[1])

                if len(datas) == 0 or len(datas[-1]['text']) + len(texts) + 1> config['text_max_len'] - 2 or files[i].split('.')[0] != datas[-1]['pdf_id']:
                    datas.append({'text' : ['[CLS]', '[SEP]', '[CLS]'] + texts + ['[SEP]'],
                                  'pdf_id' : files[i].split('.')[0],
                                  'sen_id' : [int(data_line[2])],
                                  'sen_cls': [(2, len(texts) + 3)]})
                else:
                    datas[-1]['sen_cls'].append((len(datas[-1]['text']), len(datas[-1]['text']) + len(texts) + 1))
                    datas[-1]['text'] += ['[CLS]'] + texts + ['[SEP]']
                    datas[-1]['sen_id'].append(int(data_line[2]))


        tokenized_all_tag = [tokenizer.tokenize(tag) for tag in all_tag]
        dataset = []
        print('start precessing data...')
        for data in tqdm(datas):

            text = pad_to_len(data['text'], '[PAD]', config['text_max_len'])
            sen_cls = data['sen_cls']

            #then iterate through each tag type, create a sample per tag type
            for tag_index in range(20):

                len_tag = len(tokenized_all_tag[tag_index]) + 1
                token_type = pad_to_len([0] * config['text_max_len'] + [1] * len_tag, 0, 512)

                # calculate mask, pad (token + tag)
                token = text + tokenized_all_tag[tag_index] + ['[SEP]'] # >= text_max_len: tag part
                token = pad_to_len(tokenizer.convert_tokens_to_ids(token), 0, 512)
                mask = pad_to_len([1] * len(data['text']) + [0] * (config['text_max_len'] - len(data['text'])) + len_tag * [1], 0, 512)

                dataset.append({
                    'token' : token,
                    'token_type' : token_type,
                    'mask' : mask,
                    'sen_cls' : sen_cls,
                    'tag': [tag_index],
                    'pdf_id' : data['pdf_id'],
                    'sen_id' : data['sen_id']
                })

        return dataset

if __name__ == '__main__':
    train_data_path = './release/train/ca_data/'
    dev_data_path = './release/dev/ca_data/'

    train_data = make_data(train_data_path, mode='train')
    train_dataset = MyDataset(train_data)
    train_dataloader = DataLoader(
    					dataset = train_dataset,
    					batch_size = 4,
    					shuffle = False,
                        collate_fn=train_dataset.collate_fn
    				)

    dev_data = make_data(dev_data_path, mode='test')
    dev_dataset = MyDataset(dev_data)
    dev_dataloader = DataLoader(
    					dataset = dev_dataset,
    					batch_size = 4,
    					shuffle = False,
                        collate_fn=dev_dataset.collate_fn
    				)

    for batch in train_dataloader:
        print(batch)
        print('-----')
        break
    for batch in dev_dataloader:
        print(batch)
        break


    with open('./train_dataset_cnn.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('./dev_dataset_cnn.pkl', 'wb') as f:
        pickle.dump(dev_dataset, f)
