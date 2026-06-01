
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
| `max_buffer_size` | 50000 | replay buffer capacity |

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
'max_buffer_size': 100000,
```

Setup on the cloud instance (use a PyTorch base image with CUDA + Python 3.12):

```bash
pip install poetry
rsync -avz --exclude='checkpoints/' --exclude='*.db' --exclude='.venv/' \
    <local>:/home/ubuntu/projects/alpha_worms/ ./alpha_worms/
cd alpha_worms && poetry install
python -u gpu_test.py 2>&1 | tee gpu_test_output.log
```

> Game recording (`worms_logger`) is optional and disabled by default unless `DATABASE_URL`
> is set. No extra setup needed for cloud.

**Expected cost to human-level** (~100–150 iterations with correct rules):
~28–30 hours × $0.22/hr ≈ **$6–7 total**.

After training, sync back just the checkpoint:

```bash
rsync cloud:~/alpha_worms/checkpoints/checkpoint_best.pth.tar ./checkpoints/
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
