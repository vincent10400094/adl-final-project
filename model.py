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
        self.conv1d = nn.Conv1d(config.hidden_size, 768, 3, stride=1, padding=1)
        self.config = config

    def forward(self, token, attention_mask, token_type_ids, start_positions, end_positions):
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
        if not start_positions == None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = ans_start_predict.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            # print(ans_start_predict.shape)
            # print(ans_end_predict.shape)
            # print(ans_start.shape)
            # print(ans_end.shape)
            loss_fn = CrossEntropyLoss()
            start_loss = loss_fn(ans_start_predict, start_positions)
            end_loss = loss_fn(ans_end_predict, end_positions)
            return (start_loss + end_loss) / 2, ans_start_predict, ans_end_predict
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