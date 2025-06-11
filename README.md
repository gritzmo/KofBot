# KOFBot

This repository contains utilities for training a Rainbow RDQN agent for *The King of Fighters 2002 UM* using [RLlib](https://docs.ray.io/en/latest/rllib.html).

## Recording gameplay

You can create an offline training dataset by recording gameplay or an existing agent using RLlib's rollout tool:

```bash
rllib rollout <checkpoint_path> --algo DQN --episodes 10 --out offline_data
```

This writes JSON files under `offline_data/` that can later be used for offline learning.

## Training

Run `train.py` directly for online training or pass the path to a dataset for offline training:

```bash
# Online (default)
python train.py

# Offline
python train.py --offline-dataset offline_data
```

The script will automatically switch to offline mode when the dataset path is provided.
