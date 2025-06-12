import time
import os
from env import KOFEnv, action_map
from wrappers import KOFActionRepeatEnv

from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env
import ray
from ray.rllib.algorithms.dqn import DQN, DQNConfig
import argparse

def kof_rainbow_env_creator(env_config: EnvContext):
    frame_skip = env_config.get("frame_skip", 1)
    base_env_cls = env_config["base_env_cls"]
    return KOFActionRepeatEnv(base_env_cls, frame_skip=frame_skip)

register_env("KOF-RDQN-v0", kof_rainbow_env_creator)


def get_rainbow_rdqn_config() -> dict:
    """Return a dictionary config for Rainbow RDQN using RLlib's builder API."""
    cfg = (
        DQNConfig()
        .environment(
            "KOF-RDQN-v0",
            env_config={"base_env_cls": KOFEnv, "frame_skip": 1},
        )
        .rollouts(num_rollout_workers=0, batch_mode="complete_episodes")
        .framework("torch")
        .resources(num_gpus=0)
        .training(
            model={
                "use_lstm": True,
                "lstm_cell_size": 256,
                "noisy": True,
                "dueling": True,
                "num_atoms": 51,
                "v_min": -10.0,
                "v_max": 10.0,
            },
            n_step=3,
            replay_sequence_length=20,
            burn_in=5,
            zero_init_states=False,
            exploration_config={},
            num_steps_sampled_before_learning_starts=100_000,
            train_batch_size=64,
            target_network_update_freq=1_000,
            lr=1e-4,
            gamma=0.99,
            double_q=True,
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 500_000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
            },
        )
    )
    return cfg.to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train KOF2002-UM with Rainbow RDQN (LSTM)."
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
        "--dataset-format",
        type=str,
        default="json",
        help="Format of the offline dataset if provided.",
    )
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    config = get_rainbow_rdqn_config()

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

    checkpoint_path = trainer.save("./kof_rainbow_rdqn_checkpoints")
    print(f"Checkpoint saved at: {checkpoint_path}")

    ray.shutdown()
