# Alpha Worms

AlphaZero implementation for **Heckmeck** (Pickomino) — a two-player dice game where players
roll 8 dice, commit faces one at a time, and race to claim tiles from a shared board.
The NN (`WormsNet`) is a 5-block ResNet1D with 64 channels (~180K parameters).

## Requirements

- Python ≥ 3.12
- [Poetry](https://python-poetry.org/)
- PyTorch ≥ 2.10 with CUDA (CPU-only works but is slow)

```bash
pip install poetry
poetry install
```

> 💡 **Tip for Dual Local/Cloud Dev**: To prevent Poetry from downloading a massive generic PyTorch wheel over a cloud instance's pre-configured CUDA setup, add an environment marker to your `pyproject.toml`:
> `torch = { version = "^2.10.0", markers = "sys_platform != 'linux'" }`

## Local Training

```bash
eval $(poetry env activate)
```

```bash
python -u gpu_test.py 2>&1 | tee gpu_test_output.log
```

Checkpoints are saved to `checkpoints/`. Training resumes automatically from the last best
checkpoint if one exists. To start fresh, delete `checkpoints/history/*.pkl`, `checkpoints/checkpoint_best.pth.tar`,
and any `checkpoints/checkpoint_*.pth.tar` iteration files.

Default config (RTX 3050 Ti, 16-core CPU):

| Parameter | Value | Notes |
|---|---|---|
| `num_iterations` | 50 | iterations per run |
| `num_episodes` | 14 | self-play games/iter (one per CPU worker) |
| `numMCTSSims` | 80 | MCTS simulations per move |
| `arenaCompare` | 15 | arena games per iteration |
| `arenaThreshold` | 0.55 | challenger win-rate to pass |
| `batch_size` | 256 | training batch size |
| `max_buffer_size` | 20000 | replay buffer cap (RAM: ~684 MB) |
| `keep_last_checkpoints` | 5 | per-iteration `.pth.tar` files kept on disk |
| `max_history_files` | 5 | history `.pkl` files kept on disk |

**Storage budget per run** (14 eps/iter, ~100 moves avg):
- Each history pkl ≈ 49 MB → 5 files ≈ 245 MB
- Per-iteration checkpoints: 5 × ~5 MB ≈ 25 MB
- `checkpoint_best.pth.tar`: ~5 MB (always kept)
- **Total disk: < 300 MB** regardless of run length

Expected speed: **~22–24 min/iter** (self-play ~15 min + arena ~6–8 min + training ~1–2 min).

---

## Cloud Training (Vast.ai)

The bottleneck is CPU parallelism (MCTS + arena games), not GPU VRAM.
WormsNet is tiny (~2 MB checkpoint) — any GPU with ≥ 4 GB VRAM works.

### 1. Recommended Host Selection Criteria
When renting an instance on Vast.ai, ensure your hardware filters match or exceed these balances to avoid out-of-memory errors:
* **GPU:** 1x RTX 3090 (or similar)
* **CPU Cores:** ≥ 32 cores (Crucial for MCTS simulations)
* **System RAM:** ≥ 64 GB (Rule of thumb: 1.5–2 GB RAM per core to protect against OOM process killing)
* **Disk Allocation:** 40–50 GB (More than enough for setup, logs, and checkpoints without wasting stopped-state storage money)

### 2. Required Docker Configurations
Before deploying, edit your Vast.ai Template **Docker Create Options** or **Extra Arguments** text box to include:
```text
--shm-size=16g --ulimit nofile=1000000:1000000
```

> ⚠️ **CRITICAL ENVIRONMENT WARNING**: 
> 1. **Shared Memory (`--shm-size=16g`):** Default Docker templates limit shared memory (`/dev/shm`) to 64 MB. Because `torch.multiprocessing` utilizes system shared memory to seamlessly share tensors and game states across your 32 worker threads, a restrictive default will cause your pipeline to crash instantly with a silent `Bus error (core dumped)`.
> 2. **File Descriptors (`--ulimit nofile=...`):** Spawning 32 workers utilizing pipes/queues opens hundreds of concurrent connections. Failing to lift the standard 1024 open file limit results in a mid-run `Too many open files` crash.

### 3. Recommended Config Changes (`gpu_test.py` for 32 Cores)
```python
'num_iterations': 100,   # longer run
'num_episodes': 28,      # fill the extra cores (cpu_count - 2)
'arenaCompare': 20,      # more arena games, still affordable with all threads
'max_buffer_size': 20000,        # RAM cap: ~684 MB (each example is ~35 KB)
'keep_last_checkpoints': 3,      # only 3 × ~5 MB = ~15 MB of iteration files
'max_history_files': 5,          # 5 × ~98 MB = ~490 MB history (28 eps/iter)
```

> **Storage note (32 cores)**: each history pkl is ≈ 98 MB (28 eps × 100 moves × 35 KB).
> Five files = ~490 MB. Total cloud disk including checkpoints: **< 600 MB**.

### 4. Instance Setup and Execution
Deploy using a PyTorch base image containing CUDA + Python 3.12 (e.g., `pytorch/pytorch:latest`).

**Step 1: Sync your project files to the instance**
```bash
rsync -avz -e 'ssh -p 42652' \
    --exclude='checkpoints/' \
    --exclude='*.db' \
    --exclude='.venv/' \
    --exclude='eval/' \
    --exclude='tests/' \
    /home/pablo/Documents/projects/alpha_worms/ \
    root@171.247.180.119:/workspace/alpha_worms/
```

**Step 2: Initialize environment and run using Tmux**
Always launch inside a `tmux` session so that transient network drops or closing your local laptop does not send a hang-up signal (`SIGHUP`) and kill your multi-hour pipeline.
```bash
cd alpha_worms

# Automates official Poetry installation, fixes PATH, and allows system package inheriting
bash setup_vast.sh 
source ~/.bashrc
poetry install

# Open a persistent detached window session
# tmux new -s alpha_run

# Enable env
eval $(poetry env activate)

# Inside the tmux window, execute training:
python -u gpu_test.py 2>&1 | tee gpu_test_output.log
```
*To disconnect safely from tmux without killing the run: Press `Ctrl+B`, let go, then hit `D`.* *To reconnect later to check progress: SSH back into the instance and run `tmux attach -t alpha_run`.*

> Game recording (`worms_logger`) is optional and disabled by default unless `DATABASE_URL`
> is set. No extra setup needed for cloud.

**Expected cost to human-level** (~100–150 iterations with correct rules):
~28–30 hours × $0.22/hr ≈ **$6–7 total**.

After training, sync back just the checkpoint:

```bash
rsync -avz --progress -e 'ssh -p 42652'     root@171.247.180.119:/workspace/alpha_worms/checkpoints/checkpoint_best.pth.tar     ./checkpoints/
```

---

## Extracting the Best Model from Vast.ai

### Check disk usage before downloading

```bash
# How much disk is the run using?
ssh <user@host> "du -sh ~/alpha_worms/checkpoints/ ~/alpha_worms/checkpoints/history/"

# Quick view of what's there
ssh <user@host> "ls -lh ~/alpha_worms/checkpoints/*.pth.tar | sort -V"
```

### Download only what you need (zero storage waste)

`checkpoint_best.pth.tar` is the only file you need locally — it is the arena-validated
best model and is ~5 MB. Download it and nothing else:

```bash
rsync -avz --progress -e 'ssh -p 40142'     root@78.141.73.233:/workspace/alpha_worms/checkpoints/checkpoint_best.pth.tar     ./checkpoints/
```

To also grab the training log:

```bash
rsync -avz --progress \
    <user@host>:~/alpha_worms/gpu_test_output.log \
    ./gpu_test_output.log
```

### If the run was interrupted

If training was killed mid-iteration, `checkpoint_best.pth.tar` still exists and is
**always** the safest checkpoint — it is only overwritten when a challenger passes the
arena, so it is never left in a half-written state.

To also grab the last fully-written per-iteration checkpoint (which may be newer than
the best but has not yet been arena-validated):

```bash
# 1. List checkpoints on the remote to find the latest
ssh <user@host> "ls -lh ~/alpha_worms/checkpoints/checkpoint_*.pth.tar | sort -V"

# 2. Download the best (arena-validated) model — always safe
rsync -avz --progress \
    <user@host>:~/alpha_worms/checkpoints/checkpoint_best.pth.tar \
    ./checkpoints/checkpoint_best.pth.tar

# 3. Optionally download the latest iteration checkpoint too
rsync -avz --progress \
    <user@host>:~/alpha_worms/checkpoints/checkpoint_<N>.pth.tar \
    ./checkpoints/
```

> **Do NOT rsync the entire `checkpoints/` folder** — history pkl files are large
> (up to ~100 MB each) and are only needed to resume training on the same instance.
> The single `checkpoint_best.pth.tar` file is sufficient to resume or evaluate locally.

### Resuming from a downloaded checkpoint

Place the downloaded file in `checkpoints/` as `checkpoint_best.pth.tar`.
Training will automatically resume from the last saved iteration number:

```bash
python -u gpu_test.py 2>&1 | tee gpu_test_output.log
```

To start fully fresh (discard all prior training), remove the checkpoint files:

```bash
rm -f checkpoints/checkpoint_best.pth.tar checkpoints/checkpoint_*.pth.tar
rm -f checkpoints/history/*.pkl checkpoints/best_iteration.txt
```

---

## Arena — How Challenger Promotion Works

After each iteration the newly-trained model (challenger) plays `arenaCompare` games against
the current best (baseline). Only decisive games (wins/losses, draws excluded) count toward
the win rate. The challenger replaces the baseline only if it wins ≥ 55% of decisive games.
All CPU threads are used for arena games (`min(arena_games, cpu_count - 2)`).

> **Note**: draw rate is typically 85–90% early in training while the model is still learning.
> Raising `arenaCompare` to 25–30 gives more decisive games per arena and reduces
> the chance of a noise-driven promotion.

---

## Inference

```python
from alpha_worms.nn import WormsPredictor
from alpha_worms.reflex_endpoint.game_engine import GameEngine

predictor = WormsPredictor()
predictor.load_checkpoint(folder='./checkpoints', filename='checkpoint_best.pth.tar', load_optimizer=False)

game = GameEngine()
state = game.get_initial_state()
policy, value = predictor.predict(state)
# policy: np.ndarray shape (25,) — action probabilities
# value:  float in [-1, 1]     — expected outcome for current player
```

Actions 0–15 map to pieces 21–36, actions 16–23 to dice face selections, action 24 is roll.