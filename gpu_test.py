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
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(_file_handler)

    # 1. Initialize Game and Brain
    game = GameEngine() 
    # predictor = WormsPredictor()

    # 2. Define STRESS TEST Hyperparameters for RTX 3050 Ti
    args = {
        'lr': 0.001,             # reduce this if plateu of loss function 
        'numMCTSSims': 80,        # Increased: Higher quality moves, still fast on CPU workers
        'cpuct': 1.41,            # Classic AlphaZero constant for exploration
        'num_iterations': 50,    # Run as many as possible in 24h
        'num_episodes': 14,       # More games per iteration for a richer replay buffer
        # 'num_episodes': 20,       # More games per iteration for a richer replay buffer
        'batch_size': 256,        # GPU can handle large batches (much faster training)
        'max_buffer_size': 50000, # Large buffer to prevent catastrophic forgetting
        'checkpoint': './checkpoints/',
        'log_file': 'gpu_test_output.log',
        'train_epochs': 20,        # Spend more time optimizing each batch on GPU
        'arenaCompare': 15,        # Number of arena games for model validation
        # 'arenaCompare': 33,        # Number of arena games for model validation
        'arenaThreshold': 0.55,    # Challenger must win at least 55% of arena games
        'arenaSimMultiplier': 1,   # Arena uses 2x the self-play MCTS sims
        # 'arenaSimMultiplier': 2,   # Arena uses 2x the self-play MCTS sims

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