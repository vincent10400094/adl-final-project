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
from dataset import create_dataloader, tokenizer
import os

device = None

def build_model():
	model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=40)
	model.resize_token_embeddings(len(tokenizer))

	#setting optimizer
	param_optimizer = list(model.named_parameters())
	optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not 'bias' in n], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if 'bias' in n], 'weight_decay_rate': 0.0}
    ]
	optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['lr'], eps=1e-8)
	#optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], amsgrad=True)
	model = model.to(device)

	pos_weight = [3777, 3777, 2698, 3777, 3383, 3063, 2982, 2764, 3908, 18889, 3285, 4927, 2764, 830, 1471, 2575, 1691, 1770,
				1139, 1302, 1122, 888, 109, 224, 183, 435, 359, 351, 343, 1127, 224, 379, 174, 46, 133, 147, 93, 99, 65, 60]
	criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
	criterion = criterion.to(device)
	return model, optimizer, criterion

def dev(devloader, model, criterion):
	model.eval()
	with torch.no_grad():
		dev_loss = []
		for step, batch in enumerate(devloader):
			token = batch['token'].to(device)
			token_type = batch['token_type'].to(device)
			mask = batch['mask'].to(device)
			label = batch['label'].to(device)

			scores = model(token, attention_mask=mask, token_type_ids=token_type.to(device))[0]
			batch_token_len = scores.size(0) * scores.size(1)

			#masking padding token
			padding_mask = ~mask.bool()
			scores[padding_mask] = -1e3

			loss = criterion(scores.view(batch_token_len, -1), label.view(batch_token_len, -1))

			# print('step : ', step, ' dev loss : ', loss.item(), end='\r')
			dev_loss.append(loss.item())
		return dev_loss

def train(model, optimizer, criterion, trainloader, devloader):
	for epoch in range(1, config["epoch"] + 1):
		print('epoch:', epoch)
		model.train()
		train_loss = []
		epoch_start_time = time.time()

		#for step, batch in tqdm(enumerate(trainloader), total=len(trainloader)):
		for step, batch in enumerate(trainloader):
			token = batch['token'].to(device)
			token_type = batch['token_type'].to(device)
			mask = batch['mask'].to(device)
			label = batch['label'].to(device)

			scores = model(token, attention_mask=mask, token_type_ids=token_type)[0]
			batch_token_len = scores.size(0) * scores.size(1)
			#print(scores)

			#masking padding token
			padding_mask = ~mask.bool()
			scores[padding_mask] = -1e3
			loss = criterion(scores.view(batch_token_len, -1), label.view(batch_token_len, -1))

			print('step : {}/{} | train loss : {}'.format(step, len(trainloader), loss.item()), end='\r')

			train_loss.append(loss.item())
			optimizer.zero_grad()
			loss.backward()
			# torch.nn.utils.clip_grad_value_(model.parameters(), 1)
			optimizer.step()

		print('')
		dev_loss = dev(devloader, model, criterion)
		print('[%03d/%03d] %2.2f sec(s) Tarin loss: %3.6f Dev loss: %3.6f' % (epoch , config["epoch"], time.time() - epoch_start_time, mean(train_loss), mean(dev_loss)))
		checkpoint_path = f'./checkpoint.{epoch}.pt'
		torch.save({'state_dict' : model.state_dict(), 'epoch' : epoch, }, checkpoint_path)

def main(config):
	print('building model...')
	model, optimizer, criterion = build_model()

	train_dataloader = create_dataloader(train_data_path, mode='train', batch_size=config['batch_size'], load=True)
	dev_dataloader = create_dataloader(dev_data_path, mode='dev', batch_size=config['batch_size'], load=True)

	train(model, optimizer, criterion, train_dataloader, dev_dataloader)

if __name__ == '__main__':

	train_data_path = './release/train/ca_data/'
	dev_data_path = './release/dev/ca_data/'

	#reading config
	with open("./config.json") as f:
		config = json.load(f)

	device = torch.device("cuda:{}".format(config['gpus']) if torch.cuda.is_available() else "cpu")

	main(config)
