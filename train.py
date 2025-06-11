import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Array, Event

from env import KOFEnv, action_map
from wrappers import KOFActionRepeatEnv

from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env
import ray
from ray.rllib.algorithms.dqn import DQN
import argparse

# Shared plotting utilities
N_ACTIONS = len(action_map)
shared_action_counts = Array('i', N_ACTIONS, lock=True)
plot_stop_event = Event()

def plot_worker(shared_counts: Array, stop_event: Event):
    n = len(shared_counts)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.ion()

    colors = [action_map[i]['color'] for i in range(n)]
    bars = ax.bar(range(n), [0]*n, color=colors)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Action")
    ax.set_ylabel("Usage %")
    ax.set_title("Live Action Distribution")
    ax.set_xticks(range(n))
    ax.set_xticklabels([action_map[i]['name'] for i in range(n)], rotation=45, ha='right')

    fig.canvas.draw()
    plt.pause(0.01)

    while not stop_event.is_set():
        counts = np.frombuffer(shared_counts.get_obj(), dtype=np.int32).copy()
        total = counts.sum() if counts.sum() > 0 else 1
        percentages = counts / total * 100.0

        for bar, h in zip(bars, percentages):
            bar.set_height(h)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.1)

    plt.close(fig)


def kof_rainbow_env_creator(env_config: EnvContext):
    frame_skip = env_config.get("frame_skip", 1)
    base_env_cls = env_config["base_env_cls"]
    return KOFActionRepeatEnv(base_env_cls, frame_skip=frame_skip)

register_env("KOF-RDQN-v0", kof_rainbow_env_creator)


def get_rainbow_rdqn_config():
    return {
        "env": "KOF-RDQN-v0",
        "env_config": {
            "base_env_cls": KOFEnv,
            "frame_skip": 1,
        },
        "num_workers": 0,
        "num_gpus": 0,
        "framework": "torch",
        "batch_mode": "complete_episodes",
        "model": {
            "use_lstm": True,
            "lstm_cell_size": 256,
            "noisy": True,
            "dueling": True,
            "num_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
        },
        "buffer_size": 500_000,
        "n_step": 3,
        "replay_sequence_length": 20,
        "burn_in": 5,
        "zero_init_states": False,
        "exploration_config": {},
        "learning_starts": 100_000,
        "train_batch_size": 64,
        "target_network_update_freq": 1_000,
        "lr": 1e-4,
        "gamma": 0.99,
        "double_q": True,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "prioritized_replay_eps": 1e-6,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train KOF2002-UM with Rainbow RDQN (LSTM)."
    )
    parser.add_argument("--stop-iters", type=int, default=30_000)
    parser.add_argument("--stop-timesteps", type=int, default=3_000_000)
    parser.add_argument("--stop-reward", type=float, default=350.0)
    parser.add_argument(
        "--offline-dataset",
        type=str,
        default=None,
        help="Path to an offline dataset (JSON format). If provided, training is offline.",
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="json",
        help="Format of the offline dataset if --offline-dataset is set.",
    )
    args = parser.parse_args()

    p = Process(target=plot_worker, args=(shared_action_counts, plot_stop_event), daemon=True)
    p.start()

    ray.init(ignore_reinit_error=True)
    config = get_rainbow_rdqn_config()

    if args.offline_dataset:
        config["input"] = args.offline_dataset
        config["input_config"] = {"format": args.dataset_format}
        print(f"Training offline using dataset: {args.offline_dataset}")
    else:
        config["input"] = "sampler"

    trainer = DQN(config=config)

    for i in range(args.stop_iters):
        result = trainer.train()
        mean_reward = result.get("episode_reward_mean", None)
        total_ts = result.get(
            "timesteps_total",
            result.get("agent_timesteps_total", result.get("timesteps_this_iter", 0))
        )
        if mean_reward is None:
            print(f"[Iter {i:4d}] no full episode this iter; env_steps={total_ts:,}")
        else:
            print(f"[Iter {i:4d}] reward_mean={mean_reward:.2f}  env_steps={total_ts:,}")
        if (mean_reward is not None and mean_reward >= args.stop_reward) or total_ts >= args.stop_timesteps:
            print("Stopping!")
            break

    plot_stop_event.set()
    p.join()

    checkpoint_path = trainer.save("./kof_rainbow_rdqn_checkpoints")
    print(f"Checkpoint saved at: {checkpoint_path}")

    ray.shutdown()
