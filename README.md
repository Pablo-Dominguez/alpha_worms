
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
## Cloud Training (Vast.ai)

The bottleneck is CPU parallelism (MCTS + arena games), not GPU VRAM.
WormsNet is tiny (~2 MB checkpoint) — any GPU with ≥ 4 GB VRAM works.

**Recommended instance**: RTX 3090, ≥ 32 CPU cores (~$0.20–0.25/hr on Vast.ai).

Recommended config changes in `gpu_test.py` for a 32-core instance:

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

Setup on the cloud instance (use a PyTorch base image with CUDA + Python 3.12):

1. Sync your project files to the instance:
```bash
rsync -avz --exclude='checkpoints/' --exclude='*.db' --exclude='.venv/' \
    <local>:/home/pablo/Documents/projects/alpha_worms/ ./alpha_worms/
```

2. Run the environment configuration and boot your training session:
```bash
cd alpha_worms
# Automates official Poetry installation, configures PATH, and enables system-site-packages inheriting
bash setup_vast.sh 
source ~/.bashrc

poetry install
python -u gpu_test.py 2>&1 | tee gpu_test_output.log
```

> ⚠️ **CRITICAL MULTIPROCESSING WARNING**: Before deploying to Vast.ai, ensure your instance is explicitly configured with custom launch options set to `--shm-size=16gb` or `--ipc=host`. By default, Vast containers may restrict shared memory (`/dev/shm`) to 64 MB. Because `torch.multiprocessing` utilizes system shared memory to pass game states and tensors across your 32 worker threads, a restrictive default setting will crash your training pipeline immediately with a silent `Bus error` or memory allocation fault.

> Game recording (`worms_logger`) is optional and disabled by default unless `DATABASE_URL`
> is set. No extra setup needed for cloud.

**Expected cost to human-level** (~100–150 iterations with correct rules):
~28–30 hours × $0.22/hr ≈ **$6–7 total**.

After training, sync back just the checkpoint:

```bash
rsync -avz --progress \
    <user@host>:~/alpha_worms/checkpoints/checkpoint_best.pth.tar \
    ./checkpoints/
```
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
rsync -avz --progress \
    <user@host>:~/alpha_worms/checkpoints/checkpoint_best.pth.tar \
    ./checkpoints/checkpoint_best.pth.tar
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

## Arena — How Challenger Promotion Works

After each iteration the newly-trained model (challenger) plays `arenaCompare` games against
the current best (baseline). Only decisive games (wins/losses, draws excluded) count toward
the win rate. The challenger replaces the baseline only if it wins ≥ 55% of decisive games.
All CPU threads are used for arena games (`min(arena_games, cpu_count - 2)`).

> **Note**: draw rate is typically 85–90% early in training while the model is still learning.
> Raising `arenaCompare` to 25–30 gives more decisive games per arena and reduces
> the chance of a noise-driven promotion.

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
