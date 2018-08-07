import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import trange
from torchzoo import RWA
import torch.optim as optim
from tensorboardX import SummaryWriter
from itertools import chain, repeat, islice


def generate_pair(maxl):
    length = int(random.random() * maxl)
    seq = [(np.random.normal(0, 1), 1) for _ in range(length)]
    seq = list(islice(chain(seq, repeat((0, 0))), maxl))
    return seq, 1 if (length > maxl//2) else 0


def generate_datasets(tr_n, de_n, maxl):
    I, O = [], []
    for _ in trange(tr_n + de_n, desc='Generating Data'):
        i, o = generate_pair(maxl)
        I.append(i)
        O.append(o)
    return I[:tr_n], O[:tr_n], I[tr_n:], O[tr_n:]


class LSTMNet(nn.Module):
    def __init__(self, idim, odim):
        super().__init__()
        self.lstm = nn.LSTM(idim, odim)
        self.out = nn.Linear(odim, 1)

    def forward(self, x):
        x, (a, b) = self.lstm(x)
        return nn.Sigmoid()(self.out(x[:, -1]))


class RWANet(nn.Module):
    def __init__(self, idim, odim):
        super().__init__()
        self.rwa = RWA(idim, odim)
        self.out = nn.Linear(odim, 1)

    def forward(self, x):
        x = self.rwa(x)
        return nn.Sigmoid()(self.out(x[:, -1]))


if __name__ == '__main__':
    K = 1000  # thousand
    train_samples = int(1*K)
    dev_samples = int(1*K)
    max_len = 200
    hidden_dim = 100
    max_training_steps = 5000

    _ = generate_datasets(train_samples, dev_samples, max_len)
    tr_I, tr_O, de_I, de_O = [torch.FloatTensor(i) for i in _]
    tr_O = torch.unsqueeze(tr_O, 1)
    de_O = torch.unsqueeze(de_O, 1)

    lstm = LSTMNet(tr_I.size()[-1], hidden_dim)
    rwa = RWANet(tr_I.size()[-1], hidden_dim)

    criterion = nn.BCELoss()
    lstm_opt = optim.Adam(lstm.parameters())
    rwa_opt = optim.Adam(rwa.parameters())

    lstm_write = SummaryWriter('logs/lstm')
    rwa_write = SummaryWriter('logs/rwa')
    for train_step in trange(max_training_steps, desc='Training'):
        for log, opt, net in [[lstm_write, lstm_opt, lstm],
                              [rwa_write, rwa_opt, rwa]]:
            opt.zero_grad()   # zero the gradient buffers
            tr_P = net(tr_I)
            loss = criterion(tr_P, tr_O)
            log.add_scalar('sequence_length/train-loss', loss, train_step)
            # -------------- learn
            loss.backward()
            opt.step()
            # -------------- calculate dev loss
            de_P = net(de_I)
            loss = criterion(de_P, de_O)
            acc = torch.mean(((de_P > 0.5).float() == de_O).float())
            log.add_scalar('sequence_length/dev-loss', loss, train_step)
            log.add_scalar('sequence_length/dev-acc', acc, train_step)
