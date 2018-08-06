import torch
import random
import numpy as np
from itertools import chain, repeat, islice
import torch.nn as nn
from torchzoo import RWA
import torch.optim as optim
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

def generate_dataset(maxl, total=1000):
	dataset = []
	labelset = []
	for l in range(total):
		seq = [np.random.uniform(0, 1) for _ in range(maxl)]
		index = random.sample(range(maxl), 2)
		ones = [1 if _ in index else 0 for _ in range(maxl)] 
		dataset.append(list(zip(seq, ones)))
		tot = 0
		for i in ones:
			if i == 1:
#                 w()
				tot += seq[i]
		labelset.append(tot)
	return dataset, labelset

class RWANet(nn.Module):
	def __init__(self, idim, odim):
		super().__init__()
		self.rwa = RWA(idim, odim)
		self.out = nn.Linear(odim, 1)

	def forward(self, x):
		x = self.rwa(x)
		return (self.out(x[:, -1]))

class LSTMNet(nn.Module):
	def __init__(self, idim, odim):
		super().__init__()
		self.lstm = nn.LSTM(idim, odim)
		self.out = nn.Linear(odim, 1)

	def forward(self, x):
		x, (a, b) = self.lstm(x)
		return (self.out(x[:, -1]))

if __name__ == '__main__':
	maxl = 100
	hidden_dim = 100
	max_training_steps = 1000

	train_inp, train_lab = generate_dataset(maxl)
	train_inp = torch.FloatTensor(train_inp)
	train_lab = torch.unsqueeze(torch.FloatTensor(train_lab), dim=1)

	dev_inp, dev_lab = generate_dataset(maxl, 200)
	dev_inp = torch.FloatTensor(dev_inp)
	dev_lab = torch.unsqueeze(torch.FloatTensor(dev_lab), dim=1)

	rwa = RWANet(train_inp.size()[-1], hidden_dim)
	lstm = LSTMNet(train_inp.size()[-1], hidden_dim)

	criterion = nn.MSELoss()
	lstm_opt = optim.Adam(lstm.parameters())
	rwa_opt = optim.Adam(rwa.parameters())

	lstm_write = SummaryWriter('logs/lstm')
	rwa_write = SummaryWriter('logs/rwa')
	for train_step in trange(max_training_steps, desc='Training'):
		for log, opt, net in [[lstm_write, lstm_opt, lstm],
							  [rwa_write, rwa_opt, rwa]]:
			opt.zero_grad()   # zero the gradient buffers
			tr_P = net(train_inp)
			loss = criterion(tr_P, train_lab)
			log.add_scalar('addition/train-loss', loss, train_step)
			# -------------- learn
			loss.backward()
			opt.step()
			# -------------- calculate dev loss
			de_P = net(dev_inp)
			loss = criterion(de_P, dev_lab)
			log.add_scalar('addition/dev-loss', loss, train_step)
	# torch.save(tr_P, 'addition_model.pt')