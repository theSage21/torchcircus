Torch Circus
=============


    Where the animals from TorchZoo perform.


This holds the reproduction studies done for items in [TorchZoo](https://github.com/theSage21/torchzoo/)

How to reproduce?
-----------

Say you wanted to reproduce the results of the [RWA paper](https://arxiv.org/abs/1703.01253)

```bash
git clone https://github.com/theSage21/torchcircus
cd torchcircus
git checkout rwa
pipenv install --deploy
cd rwa
```

Now run whatever experiment you want to run.
To see which tag you need to use for a particular paper, the below table may be useful.


Arxiv ID                | tag
------------------------|------
1703.01253              | rwa


We use Pytorch on CPU. If you have another setup, make sure you install the proper versions of things before running.
