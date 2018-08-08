import os
import sys
import pandas as pd
from subprocess import run


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
else:
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
    df.columns=['text', 'train', 'question', 'answer', 'path', 'task']
