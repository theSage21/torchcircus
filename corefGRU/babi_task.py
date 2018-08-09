import os
import sys
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torchzoo.rnn import CorefGRU
from tensorboardX import SummaryWriter
import pandas as pd
from tqdm import tqdm
from itertools import repeat
from subprocess import run


PAD = '<<PAD>>'
index_tables = {}


class Word:
    def __init__(self, token=None):
        self.text = token.text if token is not None else PAD
        self.pos_ = token.pos_ if token is not None else PAD
        self.tag_ = token.tag_ if token is not None else PAD
        self.dep_ = token.dep_ if token is not None else PAD
        self.lemma_ = token.lemma_ if token is not None else PAD
        self.i = token.i if token is not None else None
        self.idx = token.idx if token is not None else None
        try:
            self.coref = [cl.i for cl in token._.coref_clusters]
        except (TypeError, AttributeError):
            self.coref = []


if not os.path.exists('tasksv11'):
    if os.path.exists('tasks_1-20_v1-1.tar.gz'):
        run(' tar xf tasks_1-20_v1-1.tar.gz', shell=True)
    else:
        print('''
        ===================================================
        Please download the v1.1 task from the below URL
        At the time of writing this software, the dataset
        resided at:
            http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-1.tar.gz

        If that does not work, please find the new url on the page below
            https://research.fb.com/downloads/babi/

        Re run the program once you have a file named
            tasks_1-20_v1-1.tar.gz
        ===================================================
        ''')
    sys.exit(0)
elif not os.path.exists('data.pickle'):
    rows = []
    root = 'tasksv11/en'
    files = [os.path.join(root, i) for i in os.listdir(root)]
    for path in files:
        with open(path, 'r') as fl:
            r = []
            for line in fl.readlines():
                parts = line.strip().split(' ', 1)
                if parts[0] == '1':
                    r = []
                parts = line.strip().split('\t')
                if len(parts) == 1:
                    r.append(line.split(' ', 1)[1])
                elif len(parts) == 3:
                    parts = parts[0].split(' ', 1) + parts[1:]
                    ex = [' '.join(r), 'train' in path,
                          parts[1], parts[2], path,
                          path.split('_')[0]]
                    rows.append(ex)
                else:
                    raise Exception(line.split('\t'))
    df = pd.DataFrame(rows)
    df.columns = ['text', 'train', 'question', 'answer', 'path', 'task']
    # ------------ filter out the non-squad like tasks.
    # ------------ I don't have the compute for a full test.
    mask = [a in d and d.count(a) == 1
            for a, d in zip(df.answer, df.text + ' ' + df.question)]
    df = df.loc[mask]
    # ------------ parse it all
    nlp = spacy.load('en_coref_sm')
    text = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        t = row.text + ' ' + row.question
        text.append([Word(w) for w in nlp(t)])
    df['parsed'] = text
    df.to_pickle('data.pickle')
else:
    df = pd.read_pickle('data.pickle')

mask = [' ' not in a for a in df.answer]
df = df.loc[mask]
train_df = df.loc[df.train]
dev_df = df.loc[~df.train]
df.info()
print(train_df.shape[0], 'train', dev_df.shape[0], 'dev')

# --------------------


def toidx(n, table):
    global index_tables
    if table not in index_tables:
        index_tables[table] = {PAD: 0}
    if n not in index_tables[table]:
        index_tables[table][n] = len(index_tables[table])
    return index_tables[table][n]


def coref_train(words):
    last_mention = {}
    coref = []
    for w in words:
        pointers = []
        for cl in w.coref:
            if cl in last_mention:
                pointers.append(last_mention[cl])
            last_mention[cl] = w.i + 1
        if len(pointers) == 0:
            pointers = 0
        else:
            pointers = pointers[0]
        coref.append(pointers)
    return coref


def vectorize(B, maxl):
    indices = []
    coref = []
    targets = []
    for _, row in B.iterrows():
        pad = [Word() for _ in range(maxl - len(row.parsed))]
        for i, p in enumerate(pad):
            p.i = len(row.parsed) + i
        # TODO: mark start and end pointers
        mark = [1 if row.answer == w.text else 0
                for w in row.parsed + pad]
        if 1 in mark:
            indices.append([(toidx(w.pos_, 'pos'),
                             toidx(w.text, 'word'))
                            for w in row.parsed + pad])
            mark = mark.index(1)
            targets.append(mark)
            coref.append(coref_train(row.parsed + pad))
    indices = torch.LongTensor(indices)
    coref = torch.LongTensor(coref)
    targets = torch.LongTensor(targets)
    return indices, coref, targets


maxl = max(df.parsed.apply(len))
print('Max length', maxl)


class Net(nn.Module):
    def __init__(self, n_pos, n_tag, embed_dim):
        super().__init__()
        self.pos_table = nn.Embedding(n_pos, embed_dim, padding_idx=0)
        self.word_table = nn.Embedding(n_tag, embed_dim, padding_idx=0)
        self.rnn = CorefGRU(2*embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, x, co):
        pos_idx = x[:, :, 0]
        word_idx = x[:, :, 1]
        pos = self.pos_table(pos_idx)
        tag = self.word_table(word_idx)
        rep = torch.cat([pos, tag], dim=2)

        und = self.rnn(rep, co)
        final = self.out(und)
        final = nn.Softmax(dim=1)(final)
        return torch.squeeze(final)


writer = SummaryWriter('logs')
B = 64
steps = 10
[(toidx(w.pos_, 'pos'), toidx(w.text, 'word'))
 for text in tqdm(df.parsed, desc='Populating POS table')
 for w in text]
net = Net(len(index_tables['pos']), len(index_tables['word']), 100)
criteria = nn.NLLLoss()
opt = optim.Adam(net.parameters())


step_count = 0
for _ in tqdm(repeat(1), desc='Epochs'):
    train_df = train_df.sample(train_df.shape[0])
    dev_df = dev_df.sample(dev_df.shape[0])
    for i in range(10):
        part = train_df[i * B: (i + 1) * B]
        a, b, target = vectorize(part, maxl)
        opt.zero_grad()
        p = net.forward(a, b)
        loss = criteria(p, target)
        loss.backward()
        opt.step()
        writer.add_scalar('train-loss', loss, step_count + i)
    for i in range(10):
        part = dev_df[i * B: (i + 1) * B]
        a, b, target = vectorize(part, maxl)
        p = net.forward(a, b)
        loss = criteria(p, target)
        writer.add_scalar('dev-loss', loss, step_count + i)
        pred = torch.argmax(p, dim=1)
        mtp = torch.mean((pred == target).float())
        index_diff = torch.mean(torch.abs(pred - target).float())
        writer.add_scalar('dev-TP', mtp, step_count + i)
        writer.add_scalar('dev-Diff', index_diff, step_count + i)
    step_count += steps
