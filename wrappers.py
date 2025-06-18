import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
from env import KOFEnv, action_map

class KOFActionRepeatEnv(Env):
    """Wrap a :class:`KOFEnv` so agents see a simple ``Discrete`` action space.

    Each chosen button press is repeated for ``frame_skip`` internal ticks of the
    underlying environment.  The wrapper works with both the original
    ``MultiDiscrete`` action space as well as newer ``Discrete`` versions.

    It also converts 4-tuples returned by older envs into the 5-value tuple
    expected by Gymnasium.
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

        # Determine whether the base env expects a MultiDiscrete or Discrete
        # action. Older versions of :class:`KOFEnv` used ``MultiDiscrete`` where
        # the first element was the button index.  Newer versions expose a plain
        # ``Discrete`` space.  Support both so the wrapper works across versions.
        if isinstance(self.orig_env.action_space, MultiDiscrete):
            self._md_env = True
            self.num_buttons = int(self.orig_env.action_space.nvec[0])
        elif isinstance(self.orig_env.action_space, Discrete):
            self._md_env = False
            self.num_buttons = int(self.orig_env.action_space.n)
        else:
            raise TypeError(
                "Unsupported action_space type: "
                f"{type(self.orig_env.action_space).__name__}"
            )

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
            if self._md_env:
                md_action = np.array([int(action), 1], dtype=np.int64)
            else:
                md_action = int(action)
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

        # Return exactly 5 values as required by Gymnasium v1.x and RLlib v2.x:
        return last_obs, total_reward, terminated, truncated, info_out

# ────────────────────────────────────────────────────────────────────────────────
