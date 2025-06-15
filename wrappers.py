import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
import pydirectinput
from env import KOFEnv, action_map, VK

class KOFActionRepeatEnv(Env):
    """
    Wraps your original KOFEnv (MultiDiscrete space) into a Discrete(n_buttons) environment
    by “repeating” each chosen button press for exactly `frame_skip` internal ticks of KOFEnv.
    Also collapses the 5‐tuple (obs, rew, done, truncated, info) into (obs, rew, done, info).
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

        # Ensure original action_space is MultiDiscrete([n_buttons, max_hold])
        assert isinstance(self.orig_env.action_space, MultiDiscrete), \
            "KOFEnv.action_space must be MultiDiscrete([n_buttons, max_hold])"
        self.num_buttons = int(self.orig_env.action_space.nvec[0])

        # Expose only Discrete(num_buttons) to the agent
        self.action_space = Discrete(self.num_buttons)
        # Observation space remains the same
        self.observation_space = self.orig_env.observation_space
        log("Observation and action spaces set")

        self.frame_skip = int(frame_skip)
        self.key_buffer = None
        log("Wrapper initialization complete")

    def reset(self, **kwargs):
        """
        Calls orig_env.reset() and returns only obs.
        """
        obs, info = self.orig_env.reset(**kwargs)
        # Release any held keys if necessary
        if self.key_buffer:
            for k in self.key_buffer:
                pydirectinput.keyUp(VK[k])
        self.key_buffer = None
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

        # Release old keys if switching buttons
        keys = action_map[int(action)]['keys']
        if self.key_buffer and self.key_buffer != keys:
            for k in self.key_buffer:
                pydirectinput.keyUp(VK[k])
            self.key_buffer = None

        for _ in range(self.frame_skip):
            md_action = np.array([int(action), 1], dtype=np.int64)
            out = self.orig_env.step(md_action)

            if len(out) == 5:
                # (obs, reward, terminated, truncated, info)
                obs_t, rew_t, term_t, trunc_t, info_t = out
                terminated = terminated or term_t
                truncated = truncated or trunc_t
                info_out = info_t
            else:
                # (obs, reward, done, info) → map done→terminated, truncated=False
                obs_t, rew_t, done_t, info_t = out
                terminated = terminated or done_t
                truncated = truncated or False
                info_out = info_t

            total_reward += rew_t
            last_obs = obs_t

            if terminated or truncated:
                break

        # If done mid‐frame‐skip, release any held keys
        if (terminated or truncated) and self.key_buffer:
            for k in self.key_buffer:
                pydirectinput.keyUp(VK[k])
            self.key_buffer = None

        # Return exactly 5 values as required by Gymnasium v1.x and RLlib v2.x:
        return last_obs, total_reward, terminated, truncated, info_out

# ────────────────────────────────────────────────────────────────────────────────
