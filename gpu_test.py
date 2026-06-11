from alpha_worms.reflex_endpoint.game_engine import GameEngine
from alpha_worms.nn import WormsPredictor
from alpha_worms.main import Worms
import torch
import torch.multiprocessing as mp
import time
import glob
import os
import logging

# MUST BE AT THE TOP: Prevents CPU thrashing in the workers
torch.set_num_threads(1)

# Default DB for game recording (worms_logger). Only applies if DATABASE_URL
# is not already set in the environment — export it externally to override.
os.environ.setdefault("DATABASE_URL", "sqlite:///./nn_game_records.db")


def cleanup_leaked_semaphores():
    """
    Remove POSIX semaphore files left in /dev/shm by a previously crashed run.
    Python's multiprocessing.Queue creates named semaphores with the prefix
    'sem.mp-' that persist on disk if the process was killed before cleanup.
    """
    pattern = '/dev/shm/sem.mp-*'
    leaked = glob.glob(pattern)
    if leaked:
        print(f"[Startup] Removing {len(leaked)} leaked semaphore(s) from a previous run...")
        for path in leaked:
            try:
                os.unlink(path)
            except OSError:
                pass  # already gone or no permission


if __name__ == '__main__':

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    cleanup_leaked_semaphores()

    # Route all logger.* calls to both the console AND the log file.
    # logging writes to stderr by default; without a FileHandler it is
    # missing from the log when only stdout is redirected (> file).
    _LOG_FILE = 'gpu_test_output.log'
    _file_handler = logging.FileHandler(_LOG_FILE, mode='a', encoding='utf-8')
    _file_handler.setLevel(logging.CRITICAL)  # Capture all logs; adjust if you want to filter
    _file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(_file_handler)
    root_logger.setLevel(logging.CRITICAL)

    # 2. THE MASTER CIRCUIT BREAKER: Disables ALL logs across ALL modules 
    # (including third-party libraries) that are below the ERROR/CRITICAL level.
    logging.disable(logging.ERROR)

    # 1. Initialize Game and Brain
    game = GameEngine() 
    # predictor = WormsPredictor()

    # 2. Hyperparameters — Local (RTX 3050 Ti, 16-core CPU)
    #
    # Storage budget (100 iters, 14 eps/iter, ~100 moves avg):
    #   - Each history pkl  ≈ 14 × 100 × 35 KB ≈  49 MB
    #   - keep_last_checkpoints=5  → 5 × ~5 MB  ≈  25 MB  (+ 1 best = ~30 MB total)
    #   - max_history_files=5      → 5 × 49 MB  ≈ 245 MB
    #   - Replay buffer (RAM only) = max_buffer_size × 35 KB
    #     20000 entries ≈ 684 MB RAM  (50000 entries ≈ 1.7 GB RAM)
    #
    # For cloud (32 cores, 28 eps/iter):
    #   - Each history pkl  ≈ 28 × 100 × 35 KB ≈  98 MB
    #   - max_history_files=5  → ~490 MB history disk
    #   - keep_last_checkpoints=3  → ~15 MB iteration checkpoints
    # Dynamically pick a sensible `num_episodes` based on available CPU cores.
  # Reserve 4 cores for main process orchestrations, GPU batching, and OS overhead.
    cpu_count = mp.cpu_count()
    default_num_episodes = 32 #max(1, cpu_count - 4)
    print(f"Detected {cpu_count} CPU cores; setting num_episodes={default_num_episodes} for maximum stability.")

    args = {
        'lr': 0.0005,                 # reduce if loss plateaus
        'numMCTSSims': 200,           # MCTS simulations per move
        'cpuct': 1.41,               # PUCT exploration constant
        'num_iterations': 200,       # iterations per run
        'num_episodes': default_num_episodes,  # self-play games/iter (dynamically scales to 60)
        'batch_size': 4096,           # training mini-batch size
        'max_buffer_size': 500000,   # OPTIMIZED: Replay buffer cap (RAM: 100k × 35 KB ≈ 3.5 GB)
        'checkpoint': './checkpoints/',
        'log_file': 'gpu_test_output.log',
        'train_epochs': 10,          # gradient steps per iteration
        'arenaCompare': 32,          # OPTIMIZED: Blasts past the 90% draw noise while easily fitting in your 60 threads
        'arenaThreshold': 0.55,      # challenger win-rate to promote
        'arenaSimMultiplier': 1,     # arena MCTS multiplier (1 = same as self-play)
        # --- Storage safety knobs ---
        'keep_last_checkpoints': 5,  # per-iteration .pth.tar files to keep on disk
        'max_history_files': 150,     # OPTIMIZED: Keeps the last 20 iterations to fully satisfy the 100k buffer
    }

    # 3. Initialize the Trainer
    worms_trainer = Worms(game, None, args)

    print(f"Starting GPU Stress Test with {args['num_episodes']} concurrent games...")
    start_time = time.time()

    # 4. START TRAINING
    try:
        worms_trainer.learn()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate hours, minutes, and seconds for easy reading
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("\n" + "="*50)
    print(f"🎉 TRAINING COMPLETE!")
    print(f"⏱️ Total Elapsed Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    print("="*50)