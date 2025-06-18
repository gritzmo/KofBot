import time
import os
from env import KOFEnv
from wrappers import KOFActionRepeatEnv

from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env
import ray
from ray.rllib.algorithms.r2d2 import R2D2
import argparse

"""
Updates applied (June 18 2025):
1. Let RLlib auto-derive `replay_sequence_length` (max_seq_len + burn_in) by
   setting it to -1.  (max_seq_len = 20, burn_in = 5 → replay_sequence_length = 25.)
2. Switched `batch_mode` to "complete_episodes" – required by R2D2.
3. Set `rollout_fragment_length` to 25 to ensure ≥ replay_sequence_length.
Everything else remains unchanged.
"""

def kof_rainbow_env_creator(env_config: EnvContext):
    """Create the KOF environment used by RLlib.

    RLlib assigns each remote worker a ``worker_index`` starting at 1 while the
    local worker (driver) uses index 0.  To map these workers to emulator
    windows we convert the RLlib index to a zero-based ``window_index`` where the
    driver and first remote worker both map to window ``0``.
    """

    frame_skip = env_config.get("frame_skip", 1)
    base_env_cls = env_config["base_env_cls"]
    base_env_kwargs = env_config.get("base_env_kwargs", {})

    window_idx = env_config.worker_index - 1 if env_config.worker_index > 0 else 0
    print(
        f"[kof_rainbow_env_creator] worker_index={env_config.worker_index} -> window_idx={window_idx}",
        flush=True,
    )
    base_env_kwargs["window_index"] = window_idx

    def factory():
        return base_env_cls(**base_env_kwargs)

    return KOFActionRepeatEnv(factory, frame_skip=frame_skip)


register_env("KOF-RDQN-v0", kof_rainbow_env_creator)


def get_r2d2_config():
    return {
        "env": "KOF-RDQN-v0",
        "env_config": {
            "base_env_cls": KOFEnv,
            "frame_skip": 1,
            "base_env_kwargs": {
                "game_exe_path": None,
                "launch_game": False,
                "auto_start": True,
                "window_index": 0,
            },
        },
        "num_workers": 1,
        "num_gpus": 0,
        "framework": "torch",
        # R2D2 needs complete episode batches.
        "batch_mode": "complete_episodes",
        "model": {
            "use_lstm": True,
            "lstm_cell_size": 256,
            "max_seq_len": 20,
        },
        # Must be ≥ (max_seq_len + burn_in) == 25.
        "rollout_fragment_length": 25,
        "noisy": True,
        "dueling": True,
        "num_atoms": 1,
        "n_step": 3,
        "burn_in": 5,
        "zero_init_states": True,
        "exploration_config": {},
        "num_steps_sampled_before_learning_starts": 100_000,
        "train_batch_size": 64,
        "target_network_update_freq": 1_000,
        "lr": 1e-4,
        "gamma": 0.99,
        "double_q": True,
        # Replay buffer configuration using new API in RLlib 2.x.
        "replay_buffer_config": {
            # <<< change this line (or remove it altogether)
            "type": "MultiAgentReplayBuffer",
            # everything below can stay as you have it
            "capacity": 50_000,
            "storage_unit": "sequences",        # needed for RNNs
            # -1 lets RLlib derive  max_seq_len + burn_in automatically
            "replay_sequence_length": -1,
            "replay_zero_init_states": True,
        }

    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train KOF2002-UM with R2D2 (LSTM)."
    )
    parser.add_argument("--stop-iters", type=int, default=30_000)
    parser.add_argument("--stop-timesteps", type=int, default=3_000_000)
    parser.add_argument("--stop-reward", type=float, default=350.0)
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        default=None,
        help=(
            "Training mode. If omitted, you will be prompted at startup."
        ),
    )
    parser.add_argument(
        "--offline-dataset",
        type=str,
        default=None,
        help="Path to an offline dataset (JSON format) used when in offline mode.",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=0,
        help="When multiple game windows exist, attach to the Nth one (0-based).",
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="json",
        help="Format of the offline dataset if provided.",
    )
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    config = get_r2d2_config()
    config["env_config"]["base_env_kwargs"]["window_index"] = args.window_index

    # Determine training mode.
    mode = args.mode
    if mode is None:
        # Prompt the user when no mode is specified via CLI.
        while True:
            choice = input("Select mode: [o]nline or [f]offline? ").strip().lower()
            if choice in ("o", "online"):
                mode = "online"
                break
            if choice in ("f", "offline"):
                mode = "offline"
                break
            print("Please enter 'o' for online or 'f' for offline.")

    if mode == "offline":
        dataset = args.offline_dataset
        if dataset is None:
            prompt = "Enter path to offline dataset [default: ./dataset]: "
            dataset = input(prompt).strip()
        if not dataset:
            dataset = os.path.join(os.path.dirname(__file__), "dataset")
            os.makedirs(dataset, exist_ok=True)
            print(f"Created default dataset directory at {dataset}")
        config["input"] = dataset
        config["input_config"] = {"format": args.dataset_format}
        print(f"Training offline using dataset: {dataset}")
    else:
        if args.offline_dataset:
            print("Warning: --offline-dataset specified but mode is online; ignoring dataset path.")
        config["input"] = "sampler"

    trainer = R2D2(config=config)

    for i in range(args.stop_iters):
        train_result = trainer.train()

        if isinstance(train_result, dict):
            mean_reward = train_result.get("episode_reward_mean", None)
            total_ts = train_result.get(
                "timesteps_total",
                train_result.get(
                    "agent_timesteps_total",
                    train_result.get("timesteps_this_iter", 0),
                ),
            )

            if mean_reward is None:
                print(
                    f"[Iter {i:4d}] no full episode this iter; env_steps={total_ts:,}"
                )
            else:
                print(
                    f"[Iter {i:4d}] reward_mean={mean_reward:.2f}  env_steps={total_ts:,}"
                )

                print(
                f"[Iter {i:4d}] trainer.train() returned {type(train_result).__name__};"
                " skipping metrics"
            )
        

    checkpoint_path = trainer.save("./kof_r2d2_checkpoints")
    print(f"Checkpoint saved at: {checkpoint_path}")

    ray.shutdown()
