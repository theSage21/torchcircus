import torch
import random
import torch.nn as nn
from tqdm import trange
from torchzoo import RWA
import torch.optim as optim
from tensorboardX import SummaryWriter


def one_hot(i, maxl):
    v = [0] * maxl
    v[i] = 1
    return v


def make_sequence(k, s, t):
    l = int(random.random() * s)
    blank = k
    delim = k + 1
    # -----------
    seq = [random.choice(range(k)) for _ in range(l)]
    target = list(seq)

    # ----------- add blanks and delim
    inp = seq + [blank] * t
    inp = inp[:t]
    flip_idx = random.choice(range(l, t-l-1))
    inp[flip_idx] = delim

    target = [blank] * t
    for i, v in enumerate(seq):
        target[flip_idx + i + 1] = v
    # ----------- pad
    inp = [one_hot(i, k+3) for i in inp]
    target = [one_hot(i, k+3) for i in target]
    return inp, target


def generate_dataset(k, s, t, tr_n, de_n):
    inp, out = [], []
    for _ in trange(tr_n + de_n, desc='Generating Data'):
        seq, targ = make_sequence(k, s, t)
        inp.append(seq)
        out.append(targ)
    return inp[:tr_n], out[:tr_n], inp[tr_n:], out[tr_n:]


class LSTMNet(nn.Module):
    def __init__(self, idim, hdim, odim):
        super().__init__()
        self.lstm = nn.LSTM(idim, hdim)
        self.out = nn.Linear(hdim, odim)

    def forward(self, x):
        x, (a, b) = self.lstm(x)
        return nn.Softmax(dim=-1)(self.out(x))


class RWANet(nn.Module):
    def __init__(self, idim, hdim, odim):
        super().__init__()
        self.rwa = RWA(idim, hdim)
        self.out = nn.Linear(hdim, odim)

    def forward(self, x):
        x = self.rwa(x)
        return nn.Softmax(dim=-1)(self.out(x))


if __name__ == '__main__':

    K = 1000  # thousand
    train_samples = int(1*K)
    dev_samples = int(1*K)
    max_training_steps = 5000
    k, s, t = 8, 10, 200
    hidden_dim = 32

    _ = generate_dataset(k, s, t, train_samples, dev_samples)
    _ = [torch.FloatTensor(i) for i in _]
    tr_I, tr_O, de_I, de_O = [torch.FloatTensor(i) for i in _]

    lstm = LSTMNet(tr_I.size()[-1], hidden_dim, k+3)
    rwa = RWANet(tr_I.size()[-1], hidden_dim, k+3)

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
            log.add_scalar('variable_copy/train-loss', loss, train_step)
            # -------------- learn
            loss.backward()
            opt.step()
            # -------------- calculate dev loss
            de_P = net(de_I)
            loss = criterion(de_P, de_O)
            err = torch.mean((torch.argmax(de_P, dim=2) != torch.argmax(de_O, dim=2)).float())
            log.add_scalar('variable_copy/dev-loss', loss, train_step)
            log.add_scalar('variable_copy/dev-err', err, train_step)
