import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import json
from statistics import mean
from transformers import BertTokenizer, BertForTokenClassification
from torch.nn import CrossEntropyLoss
import math
import time
from tqdm import tqdm
from dataset import make_data, create_dataloader, tokenizer, tag_to_label, label_to_tag
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import os
import csv

device = None

def build_model(model_path = './best.pt'):
    model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=40)
    model.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], amsgrad=True)
    model.to(device)

    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)

    return model

def find_ans(token, score, pdf_id, sen_ids):
    prediction = []

    #first, we find the sentence boundary of each sentence
    context = tokenizer.convert_ids_to_tokens(token)
    context_pos = 0
    sen_to_pos = {}

    for sen_id in sen_ids: #shouldn't consider too much special case...
        if sen_id == -1: break
        while context[context_pos] != '[SEP]':
            context_pos += 1
        sen_to_pos[sen_id] = context_pos

        context_pos += 1

    context_len = context_pos

    #start iterate through each tag
    threshold = 0.95

    for tag_index in range(20): #each tag
        score_per_tag = score[:, tag_index]
        body_score = score[:, tag_index + 20]

        score_pos = 0

        #start seaching for some score > 0.5
        while score_pos < context_len:
            if score_per_tag[score_pos] <= threshold:
                score_pos += 1
                continue
            else:
                body_pos = score_pos + 1
                while body_pos < context_len and body_score[body_pos] > threshold:
                    body_pos += 1

                #[score_pos, body_pos) is the prediction
                if body_pos - score_pos >= 2: #valid answer

                    #find sentence id
                    sen_id = None
                    for s_id, sep_pos in sen_to_pos.items():

                        if sep_pos > score_pos:
                            sen_id = s_id
                            break

                    #find predicted text
                    predicted_text = context[score_pos: body_pos]
                    predicted_text = ''.join([t.strip('#') for t in predicted_text])

                    #print('pdf_id: {}, sen_id: {}, predicted_text: {}, tag:{}, [{},{})'.format(
                    #        pdf_id, sen_id, predicted_text, label_to_tag[tag_index], score_pos, body_pos))
                    #print('score: {}, {}\n------'.format(score_per_tag[score_pos], body_score[score_pos + 1: body_pos]))

                    prediction.append((pdf_id, sen_id, label_to_tag[tag_index], predicted_text))

                score_pos = body_pos
    #print('complete iterating through each tag', flush=True)
    return prediction



def predict(model, dataloader, dev=False):
    model.eval()
    sigmoid = torch.nn.Sigmoid()
    predictions = []

    for i, batch in enumerate(dataloader):

        #calculate score of each token
        token = batch['token'].to(device)
        token_type = batch['token_type'].to(device)
        mask = batch['mask'].to(device)
        if dev:
            label = batch['label']
        else:
            pdf_id = batch['pdf_id']
            sen_id = [id.tolist() for id in batch['sen_id']]
            sen_id = list(zip(*sen_id))


        scores = model(token, attention_mask=mask, token_type_ids=token_type)[0]
        padding_mask = ~mask.bool()
        scores[padding_mask] = -1e3

        scores = sigmoid(scores)
        #score: (B x 512 x 40)

        for b in range(scores.size(0)):
            predicted = find_ans(token[b], scores[b], pdf_id[b], sen_id[b])
            print('batch {}/{}'.format(i, len(dataloader)), predicted, flush=True)
            predictions.extend(predicted)
    return predictions



def main(args, config):

    #creating dataset
    print('loading data...')
    predict_dataloader = create_dataloader(args.test_data_path, mode='test', batch_size=config['batch_size'], load=False)

    files = os.listdir(args.test_data_path)
    files.sort()
    files = [x.split('.')[0] for x in files if not x.startswith('.')]

    all_tag = ['調達年度', '都道府県', '入札件名', '施設名', '需要場所(住所)', \
                '調達開始日', '調達終了日', '公告日', '仕様書交付期限', '質問票締切日時', \
                '資格申請締切日時', '入札書締切日時', '開札日時', '質問箇所所属/担当者', '質問箇所TEL/FAX', \
                '資格申請送付先', '資格申請送付先部署/担当者名', '入札書送付先', '入札書送付先部署/担当者名', '開札場所']

    files_per_len_ans = {}
    for file_name in files:
        data = pd.read_excel(args.test_data_path+file_name+'.pdf.xlsx', encoding = 'big5')
        raw_data = data.to_numpy()
        files_per_len_ans[file_name] = {}
        for i, data_line in enumerate(raw_data):
            files_per_len_ans[file_name][int(data_line[2])] = []

    print('loading model...')
    model = build_model(args.model_path)

    print('predicting...')
    predictions = predict(model, predict_dataloader)
    print(predictions)
    for (pdf_id, sen_id, tag_name, predicted_text) in predictions:
        files_per_len_ans[pdf_id][sen_id].append(tag_name + ':' + predicted_text)

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

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--test_data_path', default='release/dev/ca_data/', type=str)
    parser.add_argument('--output_path', default='output.csv', type=str)
    parser.add_argument('--model_path', default='./best.pt', type=str)
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(config['gpus']) if torch.cuda.is_available() else "cpu")

    with open("./config.json") as f:
        config = json.load(f)

    main(args, config)
