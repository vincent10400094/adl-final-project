model_path = './checkpoint.20.pt'
threshold = 0.005

import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import json
from statistics import mean
from transformers import BertTokenizer, BertForQuestionAnswering
from model import Bert
from torch.nn import CrossEntropyLoss
import math
import time
from tqdm import tqdm
from dataset import MyDataset, make_data, tokenizer, tag_to_label, label_to_tag
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import os
import csv
import sys

translate = lambda x : ''.join(i.strip('#') for i in x)

def build_model():
    model = Bert.from_pretrained("bert-base-multilingual-cased")
    model.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], amsgrad=True)
    model.to(device)
    return model, optimizer

def test(model, devloader):
    model.eval()
    predict = []
    softmax = torch.nn.Softmax(dim = -1)
    for batch in tqdm(devloader):
        _, start_scores, end_scores = model(
                batch['token'].to(device),
                attention_mask = batch['mask'].to(device),
                token_type_ids = batch['token_type'].to(device),
            )

        #print(start_scores.shape, end_scores.shape)
        sen_cls = batch['sen_cls']
        sen_ids = batch['sen_id']
        tags = batch['tag']
        pdf_ids = batch['pdf_id']
        tokens = batch['token']

        for b in range(start_scores.size(0)):
            start_score = start_scores[b]
            end_score = end_scores[b]
            tag_id = tags[b]
            pdf_id = pdf_ids[b]

            token = tokens[b]
            #print('predicting pdf_id: {}, sen_id: [{}, {}]'.format(pdf_id, sen_ids[b][0], sen_ids[b][-1]), flush=True)


            for sen_id, (sen_start, sen_end) in zip(sen_ids[b], sen_cls[b]):
                #print('pdf_id/sen_id: {}/{}, sen_start/end: {}/{}, tag: {}'.format(pdf_id, sen_id, sen_start, sen_end, label_to_tag[tag_id]))

                start_score_per_tag = softmax(start_score[sen_start: sen_end + 1])
                end_score_per_tag = softmax(end_score[sen_start: sen_end + 1])

                start_token_score, start_pos = torch.topk(start_score_per_tag, k=1)
                end_token_score, end_pos = torch.topk(end_score_per_tag, k=1)
                #print(start_token_score + end_token_score, start_token_score + end_token_score > 1)
                if start_score_per_tag[0] + end_score_per_tag[0] > threshold: #there's nothing in this sentence
                    continue

                if (start_pos == 0) ^ (end_pos == 0):
                    print('it is weird...', flush=True)
                    print('pdf_id/sen_id: {}/{}, sen_start/end: {}/{}, tag: {}'.format(pdf_id, sen_id, sen_start, sen_end, label_to_tag[tag_id]))
                    print(tokenizer.convert_ids_to_tokens(token[sen_start: sen_end+1]))
                    #print(start_score_per_tag)
                    #print(end_score_per_tag)
                    print('start_pos', start_pos, 'end_pos', end_pos, 'unanswerable score: ', start_score_per_tag[0] + end_score_per_tag[0])
                    print('this answer score: ', start_token_score + end_token_score)
                    #exit()
                    print('---------')
                    continue

                else: #handling prediction here
                    predicted_text = token[sen_start + start_pos: sen_start + end_pos + 1]
                    predicted_text = tokenizer.convert_ids_to_tokens(predicted_text)
                    predict.append((pdf_id, sen_id, label_to_tag[tag_id], translate(predicted_text)))

                    #print(tokenizer.convert_ids_to_tokens(token[sen_start: sen_end+1]))
                    #print(start_score_per_tag)
                    #print(end_score_per_tag)
                    #print('pdf_id/sen_id: {}/{}, sen_start/end: {}/{}, tag: {}'.format(pdf_id, sen_id, sen_start, sen_end, label_to_tag[tag_id]))
                    print('normal prediction..., tag: {}, prediction: {}, unanswerable score: {}'.format(label_to_tag[tag_id],
                             translate(tokenizer.convert_ids_to_tokens(token[sen_start + start_pos: sen_start + end_pos + 1])), start_score_per_tag[0] + end_score_per_tag[0]))
                    #print('this answer score: ', start_token_score + end_token_score)
                    print('--------')
    return predict



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--test_data_path', default='./release/dev/ca_data/', type=str)
    parser.add_argument('--output_path', default='./predict.csv', type=str)
    args = parser.parse_args()

    with open("./config.json") as f:
        config = json.load(f)

    device = torch.device("cuda:{}".format(config['gpus']) if torch.cuda.is_available() else "cpu")

    print('loading data...')
    dev_data = make_data(args.test_data_path, 'test')
    dev_dataset = MyDataset(dev_data)

    #make files_per_len_ans
    files = os.listdir(args.test_data_path)
    files.sort()
    files = [x.split('.')[0] for x in files if not x.startswith('.')]

    all_tag = ['調達年度', '都道府県', '入札件名', '施設名', '需要場所(住所)', \
                '調達開始日', '調達終了日', '公告日', '仕様書交付期限', '質問票締切日時', \
                '資格申請締切日時', '入札書締切日時', '開札日時', '質問箇所所属/担当者', '質問箇所TEL/FAX', \
                '資格申請送付先', '資格申請送付先部署/担当者名', '入札書送付先', '入札書送付先部署/担当者名', '開札場所']


    files_per_len_ans = {}
    # files_per_tag_ans = {}

    for file_name in files:
        if file_name.startswith('.') or len(file_name) == 0: continue
        data = pd.read_excel(args.test_data_path+file_name+'.pdf.xlsx', encoding = 'big5')
        raw_data = data.to_numpy()
        files_per_len_ans[file_name] = {}
        for i, data_line in enumerate(raw_data):
            files_per_len_ans[file_name][int(data_line[2])] = []
        # files_per_tag_ans[file_name] = [-10000] * 20

    dev_dataloader = DataLoader(
                        dataset = dev_dataset,
                        batch_size = 4,
                        shuffle = False,
                        collate_fn=dev_dataset.collate_fn
                    )

    # model_path = './checkpoint.'+'{0}'.format(i)+'.new.pt'
    model, optimizer = build_model()

    print('loading model...')
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'])

    print('predicting...')
    predictions = test(model, dev_dataloader)
    for (pdf_id, sen_id, tag_name, predicted_text) in predictions:
        try:
            files_per_len_ans[pdf_id][sen_id].append(tag_name + ':' + predicted_text)
        except KeyError:
            print(pdf_id, sen_id, tag_name, predicted_text)
            exit()

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
                prediction = prediction.strip()
                if len(prediction) == 0:
                    prediction = 'NONE'
                row = [row_id, prediction]
                csv_writer.writerow(row)

    ans = os.popen('python release/score.py release/dev/dev_ref.csv ./predict.csv').read()
    print(i, ans)
    sys.stdout.flush()
    print('finish predicting!')
