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
from tqdm import tqdm

all_tag = ['O', '調達年度', '都道府県', '入札件名', '施設名', '需要場所(住所)', \
            '調達開始日', '調達終了日', '公告日', '仕様書交付期限', '質問票締切日時', \
            '資格申請締切日時', '入札書締切日時', '開札日時', '質問箇所所属/担当者', '質問箇所TEL/FAX', \
            '資格申請送付先', '資格申請送付先部署/担当者名', '入札書送付先', '入札書送付先部署/担当者名', '開札場所']
tag_to_label = {text: i for i, text in enumerate(all_tag)}
label_to_tag = {i: text for i, text in enumerate(all_tag + ['I-' + tag for tag in all_tag[1:]])}

never_split_list = []
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case = True, never_split = never_split_list)
full_char = 'Ａ Ｂ Ｃ Ｄ Ｅ Ｆ Ｇ Ｈ Ｉ Ｊ Ｋ Ｌ Ｍ Ｎ Ｏ Ｐ Ｑ Ｒ Ｓ Ｔ Ｕ Ｖ Ｗ Ｘ Ｙ Ｚ'.split()
tokenizer.add_tokens(['～','―', '‐', 'Fax','１','２','３','４','５','６','７','８','９','０'] + full_char)

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
    return None, None

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
            values[i] = tokenizer.tokenize(values[i])
        return tags, values

def pad_to_len(seq,padding):
    to_len = 512
    padded = seq + [padding] * max(0, to_len - len(seq))
    return padded

def make_data(data_path, mode):
    files = os.listdir(data_path)
    files.sort()
    raw_datas = []

    for file_name in files:
        if file_name.startswith('.'): continue

        data = pd.read_excel(data_path+file_name, encoding = 'big5')
        raw_data = data.to_numpy()
        raw_datas.append(raw_data)

    #first extract datas into a sample
    print('loading data...')
    datas = []
    for i, data in enumerate(raw_datas):
        for data_line in data:
            #it's a broken data (has tag but no value)
            if type(data_line[6]) == str and type(data_line[7]) != str:
                continue

            #tokenize context
            texts = tokenizer.tokenize(data_line[1])

            #tags, values: [tokenized tags...], [list of tokenized values]
            tags, values = prepare_tags_and_values(data_line[6], data_line[7])
            assert len(tags) == len(values)

            if mode != 'test': #training mode
                #add these samples into datas, change to next data if full/next file
                if len(datas) == 0 or len(datas[-1]['text']) + len(texts) > config['text_max_len'] or files[i].split('.')[0] != datas[-1]['pdf_id']:
                    datas.append({
                        'text' : texts,
                        'tag' : tags,
                        'value' : values,
                        'mask': [1] * len(texts),
                        'pdf_id' : files[i].split('.')[0]})
                else:
                    datas[-1]['text'] += ['[SEP]'] + texts
                    datas[-1]['mask'] += [0] + [1] * len(texts)
                    datas[-1]['tag'] += tags
                    datas[-1]['value'] += values

            else: #testing mode
                if len(datas) == 0 or len(datas[-1]['text']) + len(texts) > config['text_max_len'] or files[i].split('.')[0] != datas[-1]['pdf_id']:
                    datas.append({
                        'text' : texts,
                        'mask': [1] * len(texts),
                        'pdf_id' : files[i].split('.')[0],
                        'sen_id' : [int(data_line[2])]
                        })
                else:
                    datas[-1]['text'] += ['[SEP]'] + texts
                    datas[-1]['mask'] += [0] + [1] * len(texts)
                    datas[-1]['sen_id'].append(int(data_line[2]))

    #generating dataset (padding, configuring other parameters)
    print('data processing...')
    dataset = []
    for data in tqdm(datas):
        #initialize each sample
        context = ['[CLS]'] + data['text'] + ['[SEP]']
        token = pad_to_len(tokenizer.convert_tokens_to_ids(context), 0)
        mask = pad_to_len(len(context) * [1], 0)
        token_type = pad_to_len([0], 0)
        label = pad_to_len([0], 0)

        if mode != 'test':

            for i in range(len(data['tag'])):
                start, end = find_position(data['text'], data['value'][i])
                if start == None or end == None:
                    print('error text:{}\nvalue: {}'.format(data['text'], data['value']))
                    continue
                label[start] = tag_to_label[data['tag'][i]]
                label[start + 1: end + 1] = [tag_to_label[data['tag'][i]] + 20] * (end - start)
                dataset.append({
                    'token': torch.tensor(token),
                    'token_type': torch.tensor(token_type),
                    'mask': torch.tensor(mask),
                    'label': torch.LongTensor(label)
                    })
                label = pad_to_len([0], 0)

            if len(data['tag']) == 0:
                dataset.append({
                    'token': torch.tensor(token),
                    'token_type': torch.tensor(token_type),
                    'mask': torch.tensor(mask),
                    'label': torch.LongTensor(label)
                    })

        else: #testing mode

            dataset.append({
                'token': torch.tensor(token),
                'token_type': torch.tensor(token_type),
                'mask': torch.tensor(mask),
                'pdf_id' : data['pdf_id'],
                'sen_id' : data['sen_id']})

    return dataset

def create_dataloader(data_path='./release/train/ca_data/', mode='train', batch_size=32, save=True, load=False):
    if load:
        print('./{}_dataloader.pt loaded.'.format(mode), flush=True)
        dataloader = torch.load('./{}_dataloader.pt'.format(mode))
        return dataloader

    print('creating {} dataset...'.format(mode), flush=True)
    dataset = make_data(data_path, mode)
    dataloader = DataLoader(
					dataset = dataset,
					batch_size = batch_size,
					shuffle = True,
				)
    if save:
        torch.save(dataloader, './{}_dataloader.pt'.format(mode))
    return dataloader


if __name__ == '__main__':
    train_data_path = './release/train/ca_data/'
    dev_data_path = './release/dev/ca_data/'

    train_dataloader = create_dataloader(train_data_path, mode='train', batch_size=32)
    dev_dataloader = create_dataloader(dev_data_path, mode='dev', batch_size=32)

    print('------')
    for batch in train_dataloader:
        for key in batch.keys():
            print(key,':', batch[key].shape)
        break
    print('-----')
    for batch in dev_dataloader:
        for key in batch.keys():
            try:
                print(key,':', batch[key].shape)
            except AttributeError:
                print(key, ':', batch[key][0])
        break
