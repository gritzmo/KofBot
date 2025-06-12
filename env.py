import time
import numpy as np
import ctypes
from ctypes import byref, sizeof
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete
import win32gui
import pydirectinput
from collections import deque
import torch
import os
from ReadWriteMemory import ReadWriteMemory
from colorama import init, Fore, Style
import json

TF_ENABLE_ONEDNN_OPTS = 0  # Disable oneDNN optimizations for reproducibility

torch.set_num_threads( min(8, os.cpu_count()) )

# --- Constants for penalizing idle/repeat behavior ≪ADDED≫ ---
LOC_THRESHOLD    = 500    # steps staying in place before penalty
LOC_PENALTY      = 0.1  # reward subtracted when idle too long
REPEAT_THRESHOLD = 4 # repeated same action before penalty
REPEAT_PENALTY   = 50   # reward subtracted when action repeated too much

# --- Additional aggression bonuses ---
# Consecutive frames within close range of the opponent
CLOSE_RANGE_DIST   = 50      # distance threshold for "close"
CLOSE_RANGE_FRAMES = 20      # how many consecutive frames to trigger bonus
CLOSE_RANGE_BONUS  = 1.0     # reward for staying close

# Bonus when opponent's guard gauge is nearly depleted and a hit lands
GUARD_CRUSH_THRESHOLD = 10
GUARD_CRUSH_BONUS     = 2.0

# Reward scales for high-damage combos
COMBO_HIT_BONUS    = 0.5   # bonus per hit beyond the first when a combo ends
COMBO_DAMAGE_SCALE = 0.05  # multiplier for total damage dealt during a combo



# Eliminate internal pauses in pydirectinput for max speed
pydirectinput.PAUSE = 0

# Initialize colorama
init(autoreset=True)



# Addresses in fixed order
ADDR_KEYS = [
    'p1_hp', 'p2_hp', 'p1_super', 'p2_super', 'p1_guard', 'p2_guard',
    'p1_location', 'p2_location', 'p1_action', 'p2_action', 'p1_hitcounter',
    'p2_char_id'          # <— new entry
]
ADDR = {
    'p1_hp':       0x1440BA354,
    'p2_hp':       0x1440BA574,
    'p1_super':    0x1440BA304,
    'p2_super':    0x1440BA524,
    'p1_guard':    0x1440BA360,
    'p2_guard':    0x1440BA580,
    'p2_location': 0x1440AC320,
    'p1_location': 0x1440AC100,
    'p1_action':   0x1440BA28C,
    'p2_action':   0x1440BA4AC,
    'p1_hitcounter': 0x1440BA5EC,
    # ── NEW: P2’s character ID ──
    'p2_char_id':   0x1440CE44C,
}


# Virtual Key mappings
VK = {
    'left':  'left',
    'right': 'right',
    'up':    'up',
    'down':  'down',
    '7':     '7',
    '8':     '8',
    '9':     '9',
    '0':     '0',
    'enter': 'enter',
    'u':     'u',
    'p':     'p',
    'o':     'o',
    'i':     'i',
}

# Action map
action_map = {
    0:{'keys':[],'name':'Nothing','color':Fore.WHITE},
    1:{'keys':['left'],'name':'Move Left','color':Fore.CYAN},
    2:{'keys':['right'],'name':'Move Right','color':Fore.CYAN},
    3:{'keys':['up'],'name':'Jump','color':Fore.MAGENTA},
    4:{'keys':['down'],'name':'Crouch','color':Fore.MAGENTA},
    5:{'keys':['7'],'name':'Light Punch','color':Fore.BLUE},
    6:{'keys':['8'],'name':'Strong Punch','color':Fore.LIGHTRED_EX},
    7:{'keys':['9'],'name':'Light Kick','color':Fore.GREEN},
    8:{'keys':['0'],'name':'Strong Kick','color':Fore.RED},
    9:{'keys':['u'],'name':'Strong Punch + Strong Kick','color':Fore.CYAN},
    10:{'keys':['p'],'name':'Light Punch + Light Kick','color':Fore.MAGENTA},
    11:{'keys':['o'],'name':'Light Punch + Strong Punch','color':Fore.BLUE},
    12:{'keys':['i'],'name':'Light Kick + Strong Kick','color':Fore.LIGHTGREEN_EX},
}

# Defeat threshold: HP > threshold means defeated
DEFEAT_THRESHOLD = 200

# High-speed tap for instantaneous inputs
def fast_press(keys, hold=0.005):
    for k in keys:
        pydirectinput.keyDown(VK[k])
    time.sleep(hold)
    for k in keys:
        pydirectinput.keyUp(VK[k])
# Single key press
def press_key(key, hold=0.1):
    pydirectinput.keyDown(VK[key]); time.sleep(hold); pydirectinput.keyUp(VK[key])

# ---------------------------------------------------------------------------
# Reward helper functions
# ---------------------------------------------------------------------------
def early_damage_reward(dmg_dealt: float, step_count: int) -> float:
    """Bonus for damage dealt earlier in the round."""
    decay = np.exp(-0.01 * step_count)
    return dmg_dealt * decay


def step_time_penalty() -> float:
    """Small negative reward each step to encourage faster victories."""
    return -0.01


def combo_hits_reward(combo_hits: int) -> float:
    """Bonus based on the number of hits in a finished combo."""
    if combo_hits <= 1:
        return 0.0
    return (combo_hits - 1) * COMBO_HIT_BONUS


def combo_damage_reward(combo_damage: float) -> float:
    """Bonus proportional to total damage dealt in a combo."""
    return combo_damage * COMBO_DAMAGE_SCALE


def repeat_action_penalty(current: int, last: int | None,
                          repeat_count: int) -> tuple[float, int]:
    """Penalty when the same action is repeated too many times."""
    if current == last:
        repeat_count += 1
    else:
        repeat_count = 0
    if repeat_count >= REPEAT_THRESHOLD:
        return -REPEAT_PENALTY, repeat_count
    return 0.0, repeat_count


def out_of_range_penalty(distance: float, action_idx: int,
                         dmg_dealt: float, striking_range: float) -> float:
    """Penalty for attacking when too far away."""
    if action_idx in [5, 6, 7, 8] and distance > striking_range and dmg_dealt <= 0:
        return -0.25
    return 0.0


def stationary_penalty(count: int) -> float:
    """Penalty for staying in one location too long."""
    if count > LOC_THRESHOLD:
        return -LOC_PENALTY
    return 0.0


def retreat_penalty(distance: float, prev_distance: float,
                    action_idx: int) -> float:
    """Small penalty each frame the agent moves away from the opponent."""
    if action_idx in [1, 2] and distance > prev_distance:
        return -0.01
    return 0.0


def approach_streak_bonus(count: int, threshold: int, bonus: float) -> float:
    """Bonus when moving toward the opponent for several frames."""
    if count >= threshold:
        return bonus
    return 0.0





class KOFEnv(Env):
    metadata = {'render.modes': []}

    def __init__(self,
                 process_name="KingOfFighters2002UM_x64.exe",
                 window_title="King of Fighters 2002 Unlimited Match",
                 record_path: str | None = None):
        super().__init__()

        def log(msg: str) -> None:
            """Utility for initialization progress messages."""
            print(f"[KOFEnv:init] {msg}", flush=True)

        log("Starting environment initialization")

        self.record_path = record_path
        self._log_fh = None
        self._last_obs = None
        if self.record_path:
            os.makedirs(os.path.dirname(self.record_path) or '.', exist_ok=True)
            self._log_fh = open(self.record_path, 'a', buffering=1)
            log(f"Logging enabled: {self.record_path}")

        # at top of __init__
        self.in_transition = False
        self._transition_start = None
        self._transition_duration = 0.5
        self.round = 0
        self.nstep = 0
        # attach to process
         # track how many consecutive frames the agent has moved TOWARD opponent
        self.approach_count = 0
        # track how long we maintain close proximity to the opponent
        self.close_range_count = 0
      
       

        self.FWD_THRESHOLD = 5      # e.g. 5 frames in a row
        self.FWD_BONUS = 0.5        # once you hit 5 consecutive, give +0.1
            
        self.zerolimit = 0
        log(f"Searching for process '{process_name}'")
        self.rwm = ReadWriteMemory()
        self.process = self.rwm.get_process_by_name(process_name)
        if not self.process:
            raise Exception(f"Process '{process_name}' not found.")
        log("Opening process handle")
        self.process.open()
        self.handle = self.process.handle
        log(f"Looking for window titled '{window_title}'")
        self.hwnd = win32gui.FindWindow(None, window_title)
        if not self.hwnd:
            raise Exception("Game window not found; check `window_title`.")
        log("Window handle obtained")



        # ─── History of action‐state codes ───

        self.HISTORY_LEN = 100
        # each entry is a pair (p1_code, p2_code)
        self.code_history = deque([(0,0)]*self.HISTORY_LEN, maxlen=self.HISTORY_LEN)

        # expand obs‐space to include history
        obs_dim = len(ADDR_KEYS) + 2*self.HISTORY_LEN
        self.observation_space = Box(0, 2**32-1, (obs_dim,), np.float32)
        MAX_HOLD = 10  # max hold time for actions
        self.action_space = MultiDiscrete([len(action_map), MAX_HOLD+1])
        log("Observation and action spaces configured")

        # store previous HPs for reward
        self.prev = {'p1':None, 'p2':None}
        self.prev['p1_location'] = None
        self.prev['p2_location'] = None
        self.prev['p1_hitcounter']   = None
        self.prev['p1_super'] = None
        self.prev['combo_len']   = 0
        self.prev['p2_char_id'] = None

        self.episode_rewards = []
        # new: track per-episode totals
        self.lose_streak       = 0
        self.current_dmg_dealt    = 0.0
        self.current_dmg_taken    = 0.0
        self.current_return       = 0.0
        self.episode_action_counts = []
        self.same_loc_count = 0
        self.repeat_count = 0
        self.last_action = None
        

      
        self.n_actions = len(action_map)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.bar_update_counter = 0
        self.bar_update_interval = 5   # redraw once every 5 steps

        # input buffer
        self.key_buffer = None
        self.buffer_remaining = 0
        log("Initialization complete")

    
    def _read(self, addr):
        buf = ctypes.c_uint32()
        bytes_read = ctypes.c_size_t()
        if not ctypes.windll.kernel32.ReadProcessMemory(
            self.handle, ctypes.c_void_p(addr), byref(buf), sizeof(buf), byref(bytes_read)
        ):
            raise ctypes.WinError()
        return buf.value

    def _get_obs(self):
        vals = [self._read(ADDR[k]) for k in ADDR_KEYS]
        # map raw defeat flag
        for i in (0,1):  # p1_hp, p2_hp
            if vals[i] == 65534:
                vals[i] = 0
        # pause detection
        if vals[0] == 0 and vals[1] == 0:
            while True:
                time.sleep(0.05)
                p1 = self._read(ADDR['p1_hp'])
                p2 = self._read(ADDR['p2_hp'])
                if not (p1 == 0 and p2 == 0):
                    vals[0] = 0 if p1==65534 else p1
                    vals[1] = 0 if p2==65534 else p2
                    break

        
        # 2) append the history of codes
        #    ADDR_KEYS[8] = 'p1_action', ADDR_KEYS[9] = 'p2_action'
        p1_code, p2_code = vals[8], vals[9]
        self.code_history.append((p1_code, p2_code))

        hist = []
        for c1, c2 in self.code_history:
            hist.extend([c1, c2])

        

        return np.array(vals + hist, dtype=np.float32)
       

 



    def reset(self, **kwargs):

        


         # --- ensure no keys remain held on reset ---
        if self.key_buffer:
            for k in self.key_buffer:
                pydirectinput.keyUp(VK[k])
            self.key_buffer = None
            self.buffer_remaining = 0
        
        self.key_buffer=None; self.buffer_remaining=0
        
    
          # only after first episode
        if self.episode_rewards or self.current_return != 0.0:
            # update returns data
            self.episode_rewards.append(self.current_return)
            # store & log
            self.episode_action_counts.append(self.action_counts.copy())
            ep = len(self.episode_rewards)
            ret = self.episode_rewards[-1]
            total = max(self.action_counts.sum(), 1)
            pct_ = (self.action_counts / total * 100).round(1)
            total_reward = sum(self.episode_rewards)
            print(
                f"Episode {ep}: Return={ret:.2f} | Total={total_reward:.2f} | "
                + ", ".join(
                    f"{action_map[i]['name']}={pct_[i]}%" for i in range(self.n_actions)
                )
            )
            print("-" * 60)

        # reset per-episode state
        self.current_return = 0.0
        self.approach_count = 0
        self.close_range_count = 0
        self.same_loc_count = 0
        self.repeat_count = 0
        self.last_action = None

        # reset counters
        self.action_counts.fill(0)
        self.bar_update_counter = 0

        # handle P1 defeat animation and Enter
        raw_p1 = self._read(ADDR['p1_hp'])
        if raw_p1 > DEFEAT_THRESHOLD:

                # handle initial menu pause: both HP zero
            raw_p1 = self._read(ADDR['p1_guard'])
            raw_p2 = self._read(ADDR['p2_guard'])
            
            for dots in range(1,4):
                print(f"Restarting environment{'.'*dots}", end='\r', flush=True)
                time.sleep(5)
                raw_p1 = self._read(ADDR['p1_guard'])
                raw_p2 = self._read(ADDR['p2_guard'])
                if raw_p1 > 0 and raw_p2 > 0:
                    print(f"Current Guard: P1:{raw_p1:.0f} P2:{raw_p2:.0f}")
                    print("Pressed Enter.")
                    press_key('enter', hold=0.2)
                else:
                    print("Moving On...")
                    pass

        

       
                
        raw_p1 = self._read(ADDR['p1_guard'])
        raw_p2 = self._read(ADDR['p2_guard'])
               
        
        if raw_p1 == 0 and raw_p2 == 0:
            print("P Guards are zero.")
            # selecting endless mode countdown
            for sec in range(3,0,-1):
                print(f"Selecting Endless Mode in {sec} seconds", end='\r', flush=True)
                time.sleep(1)
            print()
            print("Selecting Endless Mode...")
            press_key('enter', hold=0.2)

            # wait before selecting character
            for sec in range(3,0,-1):
                print(f"Waiting {sec} Seconds To Select Character...", end='\r', flush=True)
                time.sleep(1)
            print()

            print("Selecting Character: Angel")
            for _ in range(5):
                press_key('right', hold=0.1)
                print(f"Pressed Right {_} times...", end='\r', flush=True)
                time.sleep(2)
            for _ in range(4):
                press_key('down', hold=0.1)
                print(f"Pressed Down {_} times...", end='\r', flush=True)
                time.sleep(2)
            press_key('enter', hold=0.1)
            print("CHARACTER SELECTED!")
            

        

        obs = self._get_obs()
        obs = np.array(obs, dtype=np.float32)
    
    
        assert obs.shape == self.observation_space.shape, f"Inconsistent obs: {obs.shape} vs {self.observation_space.shape}"
        self.prev['p1'], self.prev['p2'] = obs[0], obs[1]
        self.prev['p1_location'] = obs[6]
        self.prev['p2_location'] = obs[7]
        self.prev['p1_hitcounter']   =int(obs[10]) & 0xFF
        self.prev['combo_len']  = self.prev['p1_hitcounter']
        self.prev['p1_super'] = obs[2]
        self.prev['p2_char_id'] =  obs[ADDR_KEYS.index('p2_char_id')]


        # clear out history on reset
        self.code_history = deque([(0,0)]*self.HISTORY_LEN, maxlen=self.HISTORY_LEN)

        self._last_obs = obs
        if self._log_fh:
            self._log_fh.write(json.dumps({
                "obs": obs.tolist(),
                "action": None,
                "reward": None,
                "next_obs": None,
                "done": False
            }) + "\n")

        return obs, {}


        

        


    def step(self, action):
        # single-step update
        obs_prev = self._last_obs
        self.nstep += 1
        STRIKING_RANGE = 63
        win32gui.SetForegroundWindow(self.hwnd)

        # 1) decode action
        btn_idx = int(action[0])
        keys    = action_map[btn_idx]['keys']
        if self.key_buffer and self.key_buffer != keys:
            for k in self.key_buffer:
                pydirectinput.keyUp(VK[k])
            self.key_buffer = None
        if keys and self.key_buffer != keys:
            for k in keys:
                pydirectinput.keyDown(VK[k])
            self.key_buffer = keys.copy()

        # 1) Count the chosen action
        self.action_counts[btn_idx] += 1

        

        def health_bonus_bucket(hp: float) -> float:
            """
            Scaled rewards:
            - hp >= 120 → +0.10 per step
            - hp >=  80 → +0.05 per step
            - hp >=  40 → +0.00 per step
            - hp <   40 → -0.10 per step
            """
            if hp >= 120:
                return 0.25
            elif hp >= 80:
                return 0.05
            elif hp >= 40:
                return 0.0
            else:
                return -0.10
            
        # 2) preserve old combo_len, super, and combo_dmg for deltas
        prev_hits    = self.prev.get('combo_len', 0) or 0
        prev_super   = self.prev.get('p1_super', 0) or 0
        prev_combo_dmg = self.prev.get('combo_dmg', 0) or 0

        # 3) read obs
        obs        = self._get_obs()
        p1, p2     = obs[0], obs[1]
        p1_x, p2_x = obs[6], obs[7]
        raw_hits   = int(obs[10])
        hit_ct     = raw_hits & 0xFF    # mask down to 0–255
        p1_action  = obs[8]
        my_super   = obs[2]

        # 4) compute base reward from damage
        distance  = abs(p1_x - p2_x)
        dmg_dealt = max(0, min(self.prev['p2'] - p2, 120))
        dmg_taken = max(0, min(self.prev['p1'] - p1, 120))
        reward    = 2.0 * dmg_dealt - 0.5 * dmg_taken

        # Extra bonus for early damage and step penalty to promote quick fights
        reward += early_damage_reward(dmg_dealt, self.nstep)
        reward += step_time_penalty()

        # Penalize excessive action repetition
        rep_pen, self.repeat_count = repeat_action_penalty(
            btn_idx, self.last_action, self.repeat_count
        )
        reward += rep_pen
        if rep_pen:
            print(f"\U0001f501 Repeat penalty {rep_pen}")

        # 5) range bonuses / penalties
        if btn_idx in [5,6,7,8] and distance <= STRIKING_RANGE and dmg_dealt > 0:
            reward += 1
            print("✅ Hit in range +50")
        out_pen = out_of_range_penalty(distance, btn_idx, dmg_dealt, STRIKING_RANGE)
        if out_pen:
            reward += out_pen
            print("💨 Out-of-range attack −25")

        # 6) finishing combos
        combo_ended = (prev_hits > 0 and hit_ct == 0)
        if combo_ended:
            # Bonuses based on hits and total damage inflicted
            hit_bonus = combo_hits_reward(prev_hits)
            dmg_bonus = combo_damage_reward(prev_combo_dmg)
            combo_bonus = hit_bonus + dmg_bonus
            reward += combo_bonus
            print(
                f"🔥 {prev_hits}-hit combo for {prev_combo_dmg:.1f} dmg +{combo_bonus:.1f}"
            )
            prev_combo_dmg = 0  # reset after awarding

        # 7) combo step rewards (optional)
        delta_hits = hit_ct - prev_hits
        if delta_hits > 0 and not combo_ended:
            step_bonus = combo_hits_reward(hit_ct) - combo_hits_reward(prev_hits)
            reward += step_bonus
            print(f"↗ Combo hit +{step_bonus:.2f}")

        # 8) super meter shaping
        if my_super > prev_super:
            reward += 1; print("⚡ Super fill +1")
        if p1_action == 12320806:
            reward += 5; print("🌀 Super move landed +5")
        if p1_action == 12648486:
            reward += 10; print("💥 Mega move landed +10")

        # 9) closing / retreat shaping
        prev_dist = (
            abs(self.prev['p1_location'] - self.prev['p2_location'])
            if self.prev['p1_location'] is not None
            else distance
        )
        if btn_idx in [1, 2] and distance < prev_dist:
            reward += 0.05
            print("⬆️ Closing in +0.05")
        ret_pen = retreat_penalty(distance, prev_dist, btn_idx)
        if ret_pen:
            reward += ret_pen
            print("⬇️ Backing off −0.01")

        # --- New aggressive incentives ---
        #  a) Staying within CLOSE_RANGE_DIST for consecutive frames
        if distance < CLOSE_RANGE_DIST:
            self.close_range_count += 1
            if self.close_range_count >= CLOSE_RANGE_FRAMES:
                reward += CLOSE_RANGE_BONUS
                print(f"🔴 Close-range streak +{CLOSE_RANGE_BONUS} (count={self.close_range_count})")
                self.close_range_count = 0
        else:
            self.close_range_count = 0

        #  b) Guard crush reward when opponent's guard is very low and we land a hit
        p2_guard = obs[5]
        if p2_guard <= GUARD_CRUSH_THRESHOLD and dmg_dealt > 0:
            reward += GUARD_CRUSH_BONUS
            print(f"🛡️ Guard crush! +{GUARD_CRUSH_BONUS}")

          # 10) “move‐toward‐opponent” streak bonus & same‐location penalty (new)
        if p2_x is not None and p1_x is not None:
            if (p2_x > p1_x and btn_idx == 2) or (p2_x < p1_x and btn_idx == 1):
                self.approach_count += 1
            else:
                self.approach_count = 0
        else:
            self.approach_count = 0

        streak_bonus = approach_streak_bonus(
            self.approach_count, self.FWD_THRESHOLD, self.FWD_BONUS
        )
        if streak_bonus:
            reward += streak_bonus
            print(f"➡️ Approach streak bonus +{streak_bonus:.2f} (count={self.approach_count})")


       # 10) stay‐in‐place drain (penalize if P1 stays in same x‐location too many steps)
        # Check if we’re in the same spot as last frame:
        if self.prev['p1_location'] is not None and p1_x == self.prev['p1_location']:
            self.same_loc_count += 1
        else:
            self.same_loc_count = 0

        loc_pen = stationary_penalty(self.same_loc_count)
        if loc_pen:
            reward += loc_pen
            print(f"⏳ Idle-location penalty {loc_pen} (count={self.same_loc_count})")

        # 11) defeat checks
        p1_hp = self._read(ADDR['p1_hp'])
        p2_hp = self._read(ADDR['p2_hp'])
        if p1_hp > DEFEAT_THRESHOLD and p2_hp <= DEFEAT_THRESHOLD:
            reward -= 10 * self.lose_streak; print(f"❌ P1 defeated −{reward}")
            self.round = 0
            self.lose_streak += 1

            obs, _ = self.reset()
            done = True
            if self._log_fh:
                self._log_fh.write(json.dumps({
                    "obs": None if obs_prev is None else obs_prev.tolist(),
                    "action": action.tolist() if hasattr(action, "tolist") else action,
                    "reward": reward,
                    "next_obs": obs.tolist(),
                    "done": done
                }) + "\n")
            self._last_obs = obs
            return obs, reward, True, False, {}
        if p2_hp > DEFEAT_THRESHOLD and p1_hp <= DEFEAT_THRESHOLD:
            self.round += 1
            self.lose_streak = 0
            if self.lose_streak == 0:
                reward += 25
                print(f"🎉 Redemption Bonus! + {reward}")
                print()
            reward += 100 * self.round; print(f"🏆 P2 defeated + {reward}")
            time.sleep(11)
            obs_next, _ = self.reset()
            done = True
            if self._log_fh:
                self._log_fh.write(json.dumps({
                    "obs": None if obs_prev is None else obs_prev.tolist(),
                    "action": action.tolist() if hasattr(action, "tolist") else action,
                    "reward": reward,
                    "next_obs": obs_next.tolist(),
                    "done": done
                }) + "\n")
            self._last_obs = obs_next
            return obs_next, reward, True, False, {}

        # ── NOW update combo_dmg in prev: ──
        if hit_ct > 0:
            prev_combo_dmg += dmg_dealt
        else:
            prev_combo_dmg = 0


        # 2) P1‐HP shaping (higher P1‐HP → more reward)
        hb1 = health_bonus_bucket(p1_hp)
        reward += hb1
        if hb1 > 0:
            print(f"💚  Self HP bucket bonus: +{hb1}")
        elif hb1 < 0:
            print(f"💔  Self HP bucket penalty: {hb1}")

        # 3) P2‐HP shaping (lower P2‐HP → more reward)
        #    we invert the bucket so that low HP → positive bonus
        hb2 = health_bonus_bucket(p2_hp)
        #    since health_bonus_bucket gives -1 when P2 is very low,
        #    we subtract it → +1 reward; when P2 is full, we subtract +2 → -2 penalty.
        reward -= hb2
        if hb2 > 0:
            print(f"💢  Opponent high‐HP penalty: -{hb2}")
        elif hb2 < 0:
            print(f"🔥  Opponent low‐HP bonus: +{-hb2}")

            

        # 12) update running return and other prev‐state for next frame
        self.current_return += reward
        self.prev['p1']            = p1
        self.prev['p2']            = p2
        self.prev['p1_location']   = p1_x
        self.prev['p2_location']   = p2_x
        self.prev['combo_len']     = hit_ct
        self.prev['p1_hitcounter'] = hit_ct
        self.prev['p1_super']      = my_super
        self.prev['combo_dmg']    = prev_combo_dmg
        self.last_action           = btn_idx

        # 13) debug insane spikes
        if self.current_return > 500_000:
            print("🔥 INSANE JUMP:", {
                "reward": reward,
                "prev_hits": prev_hits,
                "hit_ct": hit_ct,
                "prev_combo_dmg": prev_combo_dmg,
                "dmg_dealt": dmg_dealt,
                "dmg_taken": dmg_taken,
                "p1_action": p1_action,
                "my_super": my_super,
            })

        # 14) final logging - single line update
        m = action_map[btn_idx]
        status = (
            f"Step {self.nstep:5d} | Action: {m['name']} | "
            f"HP P1:{p1:.0f} P2:{p2:.0f} | "
            f"Return: {self.current_return:.2f}"
        )
        print(m['color'] + status + Style.RESET_ALL, end='\r', flush=True)

        done = False
        if self._log_fh:
            self._log_fh.write(json.dumps({
                "obs": None if obs_prev is None else obs_prev.tolist(),
                "action": action.tolist() if hasattr(action, "tolist") else action,
                "reward": reward,
                "next_obs": obs.tolist(),
                "done": done
            }) + "\n")
        self._last_obs = obs
        return obs, reward, False, False, {}

    def flush_log(self):
        """Flush the recording file if logging is enabled."""
        if self._log_fh:
            self._log_fh.flush()

    def close_log(self):
        """Close the recording file if it is open."""
        if self._log_fh:
            self._log_fh.flush()
            self._log_fh.close()
            self._log_fh = None

    def close(self):
        self.close_log()



