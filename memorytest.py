import pymem
import time
import ctypes
import threading
import atexit
import sys

# ── Config ─────────────────────────────────────
PROCESS_NAME   = "KingOfFighters2002UM_x64.exe"
TARGET_ADDRESS = 0x1408CAEC8        # 2-byte field
BLAST_INTERVAL = 0.001              # 1 ms
# ───────────────────────────────────────────────

# High-precision sleeps (1 ms)
ctypes.windll.winmm.timeBeginPeriod(1)
atexit.register(lambda: ctypes.windll.winmm.timeEndPeriod(1))

pm = pymem.Pymem(PROCESS_NAME)
print("[INFO] Attached to", PROCESS_NAME)

# ── Friendly command → value map ───────────────
input_map = {
    "none":        0,
    "left":        4,
    "right":       8,
    "down":        2,
    "downleft":    6,
    "downright": 10,
    "up":        1,
    "up_right":  9,
    "up_left":   5,
    "lp":          128,
    "uplp": 129,
    "downlp": 130,
    "leftlp": 132,
    "rightlp": 136,
    "sp":          256,
    "upsp": 257,
    "leftsp": 260,
    "downsp": 258,
    "rightsp": 264,
    "lk":          32,
    "uplk": 33,
    "downlk": 34,
    "leftlk": 36,
    "rightlk": 40,
    "sk":          64,
    "downsk": 66,
    "upsk": 65,
    "leftsk": 68,
    "rightsk": 72,
    "sk+sp":       320,
    "left+sk+sp":  324,
    "rightsksp": 328,
    "upsksp": 321,
    "downsksp": 322,
    "rightlpsp": 392,
    "leftlpsp": 388,
    "uplpsp": 385,
    "downlpsp": 386,
    "lp+sp":       384,
    "lk+sk":       96,
    "uplksk": 97,
    "downlksk": 98,
    "left+lk+sk":  100,
    "rightlksk": 104,
    "lp+lk": 160,
    "leftlplk": 164,
    "rightlplk": 168,
    "uplplk": 161,
    "downlplk": 162,




}
# ───────────────────────────────────────────────

current_input = 0
is_paused     = False
running       = True
print_lock    = threading.Lock()   # avoid mixed prints

def blast_loop():
    """Write current_input  every BLAST_INTERVAL milliseconds."""
    global running, current_input, is_paused
    while running:
        if not is_paused:
            pm.write_bytes(TARGET_ADDRESS,
                           current_input.to_bytes(2, "little"),
                           2)
        time.sleep(BLAST_INTERVAL)

def monitor_loop():
    """Watch TARGET_ADDRESS and print only when the value changes."""
    global running
    last_seen = None
    while running:
        val = int.from_bytes(pm.read_bytes(TARGET_ADDRESS, 2), "little")
        if val != last_seen:
            with print_lock:
                print(f"[MEM] 0x{TARGET_ADDRESS:X} = 0x{val:04X} ({val})")
            last_seen = val
        time.sleep(0.01)   # poll 100×/s without flooding CPU

def input_loop():
    """Interactive console."""
    global running, current_input, is_paused
    while running:
        try:
            raw = input("\n>>> ").strip().lower()
            if raw == "exit":
                running = False
            elif raw == "pause":
                is_paused = True
                with print_lock:
                    print("[INPUT] Paused.")
            elif raw == "resume":
                is_paused = False
                with print_lock:
                    print("[INPUT] Resumed (0x%04X)" % current_input)
            elif raw in input_map:
                current_input = input_map[raw]
                is_paused = False
                with print_lock:
                    print(f"[INPUT] Now blasting {raw} (0x{current_input:04X})")
            else:
                with print_lock:
                    print("⚠️ Unknown command. Try 'left', 'sk+sp', 'none', …")
        except (EOFError, KeyboardInterrupt):
            running = False

# ── Launch threads ─────────────────────────────
threads = [
    threading.Thread(target=blast_loop,   daemon=True),
    threading.Thread(target=monitor_loop, daemon=True),
]
for t in threads:
    t.start()

input_loop()          # main thread → console

running = False       # stop worker threads
for t in threads:
    t.join()
print("[EXIT] Done.")
