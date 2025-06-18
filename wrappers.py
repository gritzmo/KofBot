import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete
from env import KOFEnv

class KOFActionRepeatEnv(Env):
    """Repeat actions for a fixed number of steps.

    This wrapper exposes ``KOFEnv`` using a :class:`gymnasium.spaces.Discrete` action space. Each selected action is repeated for ``frame_skip`` steps. Any 4-tuple output from ``KOFEnv`` is expanded to the 5-value format used by Gymnasium and RLlib.
    """
    def __init__(self, base_env_factory, frame_skip: int = 1):
        """
        Args:
            base_env_factory: callable that returns a configured ``KOFEnv`` instance
            frame_skip: how many emulator ticks to hold each button before asking agent again
        """
        super().__init__()

        def log(msg: str) -> None:
            print(f"[KOFActionRepeatEnv:init] {msg}", flush=True)

        log("Creating base environment")
        # Instantiate the underlying KOFEnv
        self.orig_env: KOFEnv = base_env_factory()
        log("Base environment created")

        # Ensure original action_space is Discrete(n_buttons)
        assert isinstance(self.orig_env.action_space, Discrete), \
            "KOFEnv.action_space must be Discrete(n_buttons)"
        self.num_buttons = int(self.orig_env.action_space.n)

        # Expose only Discrete(num_buttons) to the agent
        self.action_space = Discrete(self.num_buttons)
        # Observation space remains the same
        self.observation_space = self.orig_env.observation_space
        log("Observation and action spaces set")

        self.frame_skip = int(frame_skip)
        log("Wrapper initialization complete")

    def reset(self, **kwargs):
        """
        Calls orig_env.reset() and returns only obs.
        """
        obs, info = self.orig_env.reset(**kwargs)
        return obs, info

    def step(self, action: int):
        """
        Gymnasium‐style step() → (obs, reward, terminated, truncated, info).
        Converts any 4‐tuple (obs, reward, done, info) into
        (obs, reward, terminated=done, truncated=False, info).
        """
        total_reward = 0.0
        last_obs = None
        terminated = False
        truncated = False
        info_out = {}

        for _ in range(self.frame_skip):
            out = self.orig_env.step(int(action))

            if len(out) == 5:
                obs_t, rew_t, term_t, trunc_t, info_t = out
                terminated = terminated or term_t
                truncated = truncated or trunc_t
                info_out = info_t
            else:
                obs_t, rew_t, done_t, info_t = out
                terminated = terminated or done_t
                truncated = truncated or False
                info_out = info_t

            total_reward += rew_t
            last_obs = obs_t

            if terminated or truncated:
                break

        # Return exactly 5 values as required by Gymnasium v1.x and RLlib v2.x:
        return last_obs, total_reward, terminated, truncated, info_out

