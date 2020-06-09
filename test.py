model_path = './checkpoint.15.pt'
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import json
from statistics import mean
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.nn import CrossEntropyLoss
import math
import time
from tqdm import tqdm
from dataset import create_dataset, make_data
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import os
import csv

parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case = True)
tokenizer.add_tokens(['～', 'Fax','１','２','３','４','５','６','７','８','９','０'])

with open("./config.json") as f:
    config = json.load(f)

print('loading data...')
dev_data = make_data(args.test_data_path, 'test')
dev_dataset = create_dataset(dev_data)
files = os.listdir(args.test_data_path)
files.sort()
files = [x.split('.')[0] for x in files]


all_tag = ['調達年度', '都道府県', '入札件名', '施設名', '需要場所(住所)', \
            '調達開始日', '調達終了日', '公告日', '仕様書交付期限', '質問票締切日時', \
            '資格申請締切日時', '入札書締切日時', '開札日時', '質問箇所所属/担当者', '質問箇所TEL/FAX', \
            '資格申請送付先', '資格申請送付先部署/担当者名', '入札書送付先', '入札書送付先部署/担当者名', '開札場所']


files_per_len_ans = {}
# files_per_tag_ans = {}

for file_name in files:
    data = pd.read_excel(args.test_data_path+file_name+'.pdf.xlsx', encoding = 'big5')
    raw_data = data.to_numpy()
    files_per_len_ans[file_name] = {}
    for i, data_line in enumerate(raw_data):
        files_per_len_ans[file_name][int(data_line[2])] = []
    # files_per_tag_ans[file_name] = [-10000] * 20

dev_dataloader = DataLoader(
                    dataset = dev_dataset,
                    batch_size = 2,
                    shuffle = False,
                    collate_fn=dev_dataset.collate_fn
                )

def build_model():
    model = BertForQuestionAnswering.from_pretrained("bert-base-multilingual-cased")
    model.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], amsgrad=True)
    model.to(device)
    return model, optimizer

def findans(start_score, end_score, token):
    start, indice_start = torch.sort(start_score, descending = True)
    end, indice_end = torch.sort(end_score, descending = True)
    start_best = indice_start[0]
    end_best = indice_end[0]
    if end_best - start_best > 60 or end_best - start_best < 0:
        score_1 = start[1] + end[0]
        score_2 = start[0] + end[1]
        score_3 = start[1] + end[1]
        if score_1 > score_2 and score_1 > score_3 and 0 < indice_end[0] - indice_start[1] < 60:
            start_best = indice_start[1]
            end_best = indice_end[0]
        elif score_2 > score_1 and score_2 > score_3 and 0 < indice_end[1] - indice_start[0] < 60:
            start_best = indice_start[0]
            end_best = indice_end[1]
        elif  0 < indice_end[1] - indice_start[1] < 60:
            start_best =  indice_start[1]
            end_best = indice_end[1]
    predict = token[start_best:end_best+1]
    count_sep = predict.count('[SEP]')
    if count_sep > 1:
        predict = ''
        return start_best, end_best, predict, -1
    elif count_sep == 1:
        if start_score[start_best] > end_score[end_best]:
            end_best = start_best + predict.index('[SEP]') - 1
            predict = token[start_best: end_best+1]
        else:
            start_best = start_best + predict.index('[SEP]') + 1
            predict = token[start_best:end_best+1]
    sen_id = [-1]
    for i in range(len(token)):
        if token[i] == '[SEP]':
            sen_id.append(i)
    sen_id.append(600)
    tag_sen_id = 0

    for i in range(len(sen_id) - 1):
        if start_best >= sen_id[i] and end_best <= sen_id[i+1]:
            tag_sen_id = i
            break
    return start_best, end_best, predict, tag_sen_id

def test(model, devloader):
    model.eval()
    predict = {}
    answerable_acc = 0
    softmax = torch.nn.Softmax(dim = -1)
    now_pdf = 0
    count_to_20 = 0
    now_sen = []
    for batch in tqdm(devloader):
        start_scores, end_scores = model(
                batch['token'].to(device), 
                attention_mask = batch['mask'].to(device), 
                token_type_ids = batch['token_type'].to(device), 
            )
        for i in range(0, len(batch['mask'])):
            unanswerable = ((softmax(start_scores[i])[0] + softmax(end_scores[i])[0]) / 2).item()
            if unanswerable > 2e-7:
                tag = ''.join(batch['tag'][i])
                prediction = ''
                # print('tag : ', tag)
                # print('predict :')
            else:
                all_tokens = tokenizer.convert_ids_to_tokens(batch['token'][i])
                buttom = batch['token_type'][i].tolist().index(1) - 1
                start_best, end_best, predict, sen_id = findans(start_scores[i][1:buttom], end_scores[i][1:buttom], all_tokens[1:buttom])
                
                tag = ''.join(batch['tag'][i])
                count_sep = predict.count('[SEP]')
                
                prediction = ''.join([x.strip('#') for x in predict])
                # print('tag : ', tag)
                # print('predict :', prediction)
            
            
            # print(files_per_len_ans[batch['pdf_id'][i]])
            if len(prediction) != 0:
                files_per_len_ans[batch['pdf_id'][i]][batch['sen_id'][i][sen_id]].append(all_tag[count_to_20]+':'+prediction)
            count_to_20 += 1
            count_to_20 %= 20
        # break
    # output_file = Path(args.output_path)
    # output_file.write_text(json.dumps(predict))
    

model, optimizer = build_model()
print('loading model...')
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['state_dict'])
print('predicting...')
test(model, dev_dataloader)

# print(files_per_len_ans)

with open(args.output_path, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['ID', 'Prediction']
    # print(header)
    csv_writer.writerow(header)
    for pdf_id in files:
        for key in sorted(files_per_len_ans[pdf_id].keys()):
            row_id = pdf_id+'-'+'{0}'.format(key)
            prediction = ''
            for tag_value in files_per_len_ans[pdf_id][key]:
                prediction += tag_value+' '
            if len(prediction) == 0:
                prediction = 'NONE'
            row = [row_id, prediction]
            csv_writer.writerow(row)
print('finish predicting!')
    
