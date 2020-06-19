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
from dataset import MyDataset, tokenizer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
	model = Bert.from_pretrained("bert-base-multilingual-cased")
	model.resize_token_embeddings(len(tokenizer))
	optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], amsgrad=True)
	model.to(device)
	return model, optimizer

def dev(devloader):
	model.eval()
	with torch.no_grad():
		dev_loss = []
		for step, batch in enumerate(devloader):
			loss, start_scores, end_scores = model(
				batch['token'].to(device),
				attention_mask = batch['mask'].to(device),
				token_type_ids = batch['token_type'].to(device),
				start_positions = batch['start'].to(device),
				end_positions = batch['end'].to(device),
				sen_cls = batch['sen_cls']
			)
			# print('step : ', step, ' dev loss : ', loss.item(), end='\r')
			dev_loss.append(loss.item())
		return dev_loss

def train(model, optimizer, trainloader, devloader):
	for epoch in range(1, config["epoch"] + 1):
		print('epoch:', epoch)
		model.train()
		train_loss = []
		epoch_start_time = time.time()
		for step, batch in enumerate(trainloader):
			loss, start_scores, end_scores = model(
				batch['token'].to(device),
				attention_mask = batch['mask'].to(device),
				token_type_ids = batch['token_type'].to(device),
				start_positions = batch['start'].to(device),
				end_positions = batch['end'].to(device),
				sen_cls = batch['sen_cls']
			)
			print('step : {}/{}, train loss : {}'.format(step, len(trainloader), loss.item()), end='\r')

			train_loss.append(loss.item())
			optimizer.zero_grad()
			loss.backward()
			# torch.nn.utils.clip_grad_value_(model.parameters(), 1)
			optimizer.step()
		print('')
		dev_loss = dev(devloader)
		print('[%03d/%03d] %2.2f sec(s) Tarin loss: %3.6f Dev loss: %3.6f' % (epoch , config["epoch"], time.time() - epoch_start_time, mean(train_loss), mean(dev_loss)))

		checkpoint_path = f'./checkpoint.{epoch}.pt'
		torch.save({'state_dict' : model.state_dict(), 'epoch' : epoch }, checkpoint_path)

if __name__ == '__main__':

	print('loading data...')
	with open("./config.json") as f:
		config = json.load(f)
	with open("./train_dataset_cnn.pkl", 'rb') as f:
		train_dataset = pickle.load(f)
	with open("./dev_dataset_cnn.pkl", 'rb') as f:
		dev_dataset = pickle.load(f)

	train_dataloader = DataLoader(
						dataset = train_dataset,
						batch_size = config["batch_size"],
						shuffle = True,
	                    collate_fn=train_dataset.collate_fn
					)
	dev_dataloader = DataLoader(
						dataset = dev_dataset,
						batch_size = config["batch_size"],
						shuffle = True,
	                    collate_fn=dev_dataset.collate_fn
					)

	print('building model...')
	model, optimizer = build_model()
	train(model, optimizer, train_dataloader, dev_dataloader)
