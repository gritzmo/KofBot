# KOFBot

This repository contains utilities for training a Rainbow RDQN agent for *The King of Fighters 2002 UM* using [RLlib](https://docs.ray.io/en/latest/rllib.html).

## Recording gameplay

You can create an offline training dataset by recording gameplay or an existing agent using RLlib's rollout tool:

```bash
rllib rollout <checkpoint_path> --algo DQN --episodes 10 --out offline_data
```

This writes JSON files under `offline_data/` that can later be used for offline learning.

## Training

`train.py` can operate in either online or offline mode. You can explicitly set
the mode using the `--mode` flag or leave it off and choose interactively at
startup:

```bash
# Online
python train.py --mode online

# Offline
python train.py --mode offline --offline-dataset offline_data
```

If `--mode` is not given, `train.py` prompts you to pick a mode. When
selecting offline mode you can specify a dataset path or leave it blank. If you
leave it blank, a `dataset/` folder will be created next to `train.py` and used
as the dataset location.

