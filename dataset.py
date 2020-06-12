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
import json

all_tag = ['調達年度', '都道府県', '入札件名', '施設名', '需要場所(住所)', \
            '調達開始日', '調達終了日', '公告日', '仕様書交付期限', '質問票締切日時', \
            '資格申請締切日時', '入札書締切日時', '開札日時', '質問箇所所属/担当者', '質問箇所TEL/FAX', \
            '資格申請送付先', '資格申請送付先部署/担当者名', '入札書送付先', '入札書送付先部署/担当者名', '開札場所']

never_split_list = []
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenizer.add_tokens(['～', 'Fax','１','２','３','４','５','６','７','８','９','０', 'ＦＡＸ', 'ＴＥＬ'])

# [CLS] : 101
# [SEP] : 102
# [PAD] : 0

with open("./config.json") as f:
    config = json.load(f)

class create_dataset(Dataset):
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
        masks = []
        tags = []
        values = []
        pdf_ids = []
        sen_ids = []
        if not data[0].get('start') == None:
            for i in range(len(data)):
                starts.append(data[i]['start'])
                ends.append(data[i]['end'])
                tokens.append(data[i]['token'])
                token_types.append(data[i]['token_type'])
                masks.append(data[i]['mask'])
                tags.append(data[i]['tag'])
                values.append(data[i]['value'])
            return {
                'start':torch.LongTensor(starts),
                'end':torch.LongTensor(ends),
                'token':torch.tensor(tokens),
                'token_type':torch.tensor(token_types),
                'mask':torch.tensor(masks),
                'tag':tags,
                'value':values,
            }
        else:
            for i in range(len(data)):
                tokens.append(data[i]['token'])
                token_types.append(data[i]['token_type'])
                masks.append(data[i]['mask'])
                tags.append(data[i]['tag'])
                pdf_ids.append(data[i]['pdf_id'])
                sen_ids.append(data[i]['sen_id'])
            return {
                'token':torch.tensor(tokens),
                'token_type':torch.tensor(token_types),
                'mask':torch.tensor(masks),
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
                return start+1, end

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
    if mode != 'test':
        for i, data in enumerate(raw_datas):
            for data_line in data:
                if type(data_line[6]) == str and type(data_line[7]) != str:
                    continue
                texts = tokenizer.tokenize(data_line[1])
                tags, values = prepare_tags_and_values(data_line[6], data_line[7])
                assert len(tags) == len(values)
                if len(datas) == 0 or len(datas[-1]['text']) + len(texts) + 1> config['text_max_len'] - 2 or files[i].split('.')[0] != datas[-1]['pdf_id']:
                    datas.append({'text' : texts, 'tag' : tags, 'value' : values, 'pdf_id' : files[i].split('.')[0]})
                else:
                    datas[len(datas) - 1]['text'] += ['[SEP]'] + texts
                    datas[len(datas) - 1]['tag'] += tags
                    datas[len(datas) - 1]['value'] += values


        tokenized_all_tag = [tokenizer.tokenize(tag) for tag in all_tag]
        dataset = []
        pick = 0
        for data in datas:
            if len(data['tag']):
                for i in range(len(data['tag'])):
                    pos = find_position(data['text'], data['value'][i])
                    if pos == None:
                        continue
                    value = data['value'][i]
                    text = pad_to_len(data['text'], '[PAD]', config['text_max_len'] - 2)
                    # text = data['text']
                    token = ['[CLS]'] + text + ['[SEP'] + data['tag'][i] + ['[SEP']
                    len_text = len(text) + 2
                    len_tag = len(data['tag'][i]) + 1
                    token = pad_to_len(tokenizer.convert_tokens_to_ids(token), 0, 512)
                    token_type = pad_to_len([0] * len_text + [1] * len_tag, 0, 512)
                    mask = pad_to_len([1] * (len_text + len_tag), 0, 512)
                    dataset.append({
                        'token' : token,
                        'token_type' : token_type,
                        'mask' : mask,
                        'tag' : data['tag'][i],
                        'start' : pos[0],
                        'end' : pos[1],
                        'value' : value
                    })
            else:
            # n_tag = [tag for tag in tokenized_all_tag if tag not in data['tag']]

            # for tag in n_tag:
                tag = tokenized_all_tag[pick]
                pick = (pick + 1) % 20
                start = 0
                end = 0
                value = []
                text = pad_to_len(data['text'], '[PAD]', config['text_max_len'] - 2)
                # text = data['text']
                token = ['[CLS]'] + text + ['[SEP'] + tag + ['[SEP']
                len_text = len(text) + 2
                len_tag = len(tag) + 1
                token = pad_to_len(tokenizer.convert_tokens_to_ids(token), 0, 512)
                token_type = pad_to_len([0] * len_text + [1] * len_tag, 0, 512)
                mask = pad_to_len([1] * (len_text + len_tag), 0, 512)
                dataset.append({
                    'token' : token,
                    'token_type' : token_type,
                    'mask' : mask,
                    'tag' : tag,
                    'start' : start,
                    'end' : end,
                    'value' : value
                })
        return dataset
    else:
        for i, data in enumerate(raw_datas):
            for j, data_line in enumerate(data):
                texts = tokenizer.tokenize(data_line[1])
                if len(datas) == 0 or len(datas[-1]['text']) + len(texts) + 1 > config['text_max_len'] - 2 or files[i].split('.')[0] != datas[-1]['pdf_id']:
                    datas.append({
                        'text' : texts, 
                        'pdf_id' : files[i].split('.')[0],
                        'sen_id' : [int(data_line[2])],
                    })
                else:
                    datas[-1]['text'] += ['[SEP]'] + texts
                    datas[-1]['sen_id'].append(int(data_line[2]))


        tokenized_all_tag = [tokenizer.tokenize(tag) for tag in all_tag]
        dataset = []
        for data in datas:
            for i in range(len(tokenized_all_tag)):
                text = pad_to_len(data['text'], '[PAD]', config['text_max_len'] - 2)
                # text = data['text']
                token = ['[CLS]'] + text + ['[SEP'] + tokenized_all_tag[i] + ['[SEP']
                len_text = len(data['text']) + 2
                len_tag = len(tokenized_all_tag[i]) + 1
                token = pad_to_len(tokenizer.convert_tokens_to_ids(token), 0, 512)
                token_type = pad_to_len([0] * len_text + [1] * len_tag, 0, 512)
                mask = pad_to_len([1] * (len_text + len_tag), 0, 512)
                dataset.append({
                    'token' : token,
                    'token_type' : token_type,
                    'mask' : mask,
                    'tag' : tokenized_all_tag[i],
                    'pdf_id' : data['pdf_id'],
                    'sen_id' : data['sen_id']
                })
        return dataset

if __name__ == '__main__':
    train_data_path = './release/train/ca_data/'
    dev_data_path = './release/dev/ca_data/'

    train_data = make_data(train_data_path, 'train')
    # print(train_data)
    print(len(train_data))
    Train_dataset = create_dataset(train_data)

    dev_data = make_data(dev_data_path, 'dev')
    # print(dev_data)
    print(len(dev_data))
    Dev_dataset = create_dataset(dev_data)

    with open('./train_dataset_cnn.pkl', 'wb') as f:
        pickle.dump(Train_dataset, f)
    with open('./dev_dataset_cnn.pkl', 'wb') as f:
        pickle.dump(Dev_dataset, f)
