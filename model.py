from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertForQuestionAnswering
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
# import pytorch_lightning as pl
import math
import json

with open("./config.json") as f:
    Config = json.load(f)

class Bert(BertForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, self.config.num_labels)
        # self.is_answeable = nn.Linear(config.hidden_size, 1)
        self.init_weights()
        self.conv1d = nn.Conv1d(config.hidden_size, 768, 5, stride=1, padding=2)
        self.config = config

        self.logsoftmax = nn.LogSoftmax(dim=0)

    def cross_entropy(self, pred, soft_targets):
        return torch.sum(- soft_targets * self.logsoftmax(pred), 0)

    def forward(self, token, attention_mask, token_type_ids, start_positions = None, end_positions = None, sen_cls = None):
        outputs = self.bert(token, attention_mask = attention_mask, token_type_ids = token_type_ids)

        output = outputs[0]
        # print(self.config.hidden_size)

        output = output.permute(0,2,1).contiguous()
        output = output[:,:, :Config['text_max_len']]
        output = self.conv1d(output)
        output = output.permute(0,2,1).contiguous()

        pos = self.qa_outputs(output)
        start, end = pos.split(1, dim=-1)
        ans_start_predict = start.squeeze(-1)
        ans_end_predict = end.squeeze(-1)

        if not start_positions == None: #training
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            #print('ans_start_predict: {}, start_positions: {}'.format(ans_start_predict.shape, start_positions.shape))

            loss, total_cnt = 0, 0
            for b in range(len(sen_cls)):
                for start, end in sen_cls[b]:
                    #print(start, end)
                    #print(ans_start_predict[b][start:end+1])
                    #print(start_positions[b][start:end+1])
                    #print('-----')
                    start_loss = self.cross_entropy(ans_start_predict[b][start:end+1], start_positions[b][start:end+1])
                    end_loss = self.cross_entropy(ans_end_predict[b][start:end+1], end_positions[b][start:end+1])

                    loss += (start_loss + end_loss)
                    total_cnt += 2
            #print(loss_list)
            loss = loss / total_cnt
            return loss, ans_start_predict, ans_end_predict
        else:
            return 0, ans_start_predict, ans_end_predict

# class Net(pl.LightningModule):
#   def __init__(self):
#       model = Bert.from_pretrained("bert-base-chinese")
#   def forward(self, batch):
#       loss, start_scores, end_scores, answerable = model(
#               batch['token'].to(device),
#               mask = batch['mask'].to(device),
#               token_type_id = batch['token_type'].to(device),
#               ans_start = batch['ans_start'].to(device),
#               ans_end = batch['ans_end'].to(device),
#               answerable = batch['answerable'].to(device)
#           )
#       return {'loss':loss}
