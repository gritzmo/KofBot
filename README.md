# KofBot

This repository contains utilities and environments for training agents in **King of Fighters 2002 UM**.

## Logging Episodes

`KOFEnv` can optionally record every transition.  Pass a path when creating the environment:

```python
from env import KOFEnv
env = KOFEnv(record_path="episode_log.jsonl")
```

Each call to `reset()` or `step()` will append a JSON line with the keys `"obs"`, `"action"`, `"reward"`, `"next_obs"`, and `"done"`.  Call `env.flush_log()` or `env.close_log()` after an episode to make sure the data is written to disk.
