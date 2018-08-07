import torch
import random
import torch.nn as nn
from torchzoo import RWA
import torch.optim as optim
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from itertools import repeat, islice, chain

# We use ~ as padding
letter_to_index = {l: i for i, l in enumerate('~btpsxve')}
next_states = {1: (2, 3),
               2: (4, 2),
               3: (5, 3),
               4: (3, 6),
               5: (4, 6),
               6: None}
letters = {(1, 2): 't',
           (1, 3): 'p',
           (2, 2): 's',
           (2, 4): 'x',
           (3, 3): 't',
           (3, 5): 'v',
           (4, 3): 'x',
           (4, 6): 's',
           (5, 4): 'p',
           (5, 6): 'v',
           (6, None): 'e'}


def generate_valid_string():
    string = ['b']
    state = 1
    while state is not None:
        valid = next_states[state]
        next = random.choice(valid) if valid is not None else None
        string.append(letters[(state, next)])
        if next is None:
            break
        state = next
    return ''.join(string)


def make_invalid(string):
    i = random.choice(range(len(string)))
    new = random.choice(list(set(string) - set(string[i])))
    ns = string[:i] + new + string[i+1:]
    assert len(ns) == len(string)
    return ns


def make_train_and_dev_dataset(trn_n, dev_n):
    data = set()
    n = trn_n + dev_n
    with tqdm(total=n, desc='Generating data') as pbar:
        while (len(data) < n):
            # add one valid, one invalid
            while True:  # Valid
                string = generate_valid_string()
                if (string, 1) not in data:
                    data.add((string, 1))
                    break
            while True:  # InValid
                string = generate_valid_string()
                string = make_invalid(string)
                if (string, 0) not in data:
                    data.add((string, 0))
                    break
            pbar.update(2)
    data = list(data)[:n]
    return data[:trn_n], data[trn_n:]


def one_hot(l):
    v = [0] * len(letter_to_index)
    v[letter_to_index[l]] = 1
    return v


def encode_data(data, max_l):
    I, O = [], []
    for inp, out in tqdm(data, desc='Encoding data'):
        vec = [one_hot(l) for l in islice(chain(inp, repeat('~')), max_l)]
        I.append(vec)
        O.append(out)
    return I, O


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
    K = 1000   # thousand
    train_samples = int(0.5*K)
    dev_samples = int(0.5*K)
    hidden_dim = 100
    max_training_steps = 2000

    train, dev = make_train_and_dev_dataset(train_samples, dev_samples)
    max_l = max([len(i) for i, _ in tqdm(train + dev,
                                         desc='Finding MaxLen')])
    tr_I, tr_O = encode_data(train, max_l)
    de_I, de_O = encode_data(dev, max_l)
    del(train)
    del(dev)

    tr_I = torch.FloatTensor(tr_I)
    de_I = torch.FloatTensor(de_I)
    tr_O = torch.unsqueeze(torch.torch.FloatTensor(tr_O), 1)
    de_O = torch.unsqueeze(torch.FloatTensor(de_O), 1)

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
            log.add_scalar('artificial_grammar/train-loss', loss, train_step)
            # -------------- learn
            loss.backward()
            opt.step()
            # -------------- calculate dev loss
            de_P = net(de_I)
            loss = criterion(de_P, de_O)
            acc = torch.mean(((de_P > 0.5).float() == de_O).float())
            log.add_scalar('artificial_grammar/dev-loss', loss, train_step)
            log.add_scalar('artificial_grammar/dev-acc', acc, train_step)
