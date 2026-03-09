from alpha_worms.reflex_endpoint.game_engine import GameEngine
from alpha_worms.nn import WormsPredictor
from alpha_worms.main import Worms
import torch
import torch.multiprocessing as mp
import time

# MUST BE AT THE TOP: Prevents CPU thrashing in the workers
torch.set_num_threads(1)

if __name__ == '__main__':

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 1. Initialize Game and Brain
    game = GameEngine() 
    # predictor = WormsPredictor()

    # 2. Define STRESS TEST Hyperparameters for RTX 3050 Ti
    args = {
        'lr': 0.001,              
        'numMCTSSims': 80,        # Increased: Higher quality moves, still fast on CPU workers
        'cpuct': 1.41,            # Classic AlphaZero constant for exploration
        'num_iterations': 10,    # Run as many as possible in 24h
        'num_episodes': 30,       # More games per iteration for a richer replay buffer
        'batch_size': 64,        # GPU can handle large batches (much faster training)
        'max_buffer_size': 50000, # Large buffer to prevent catastrophic forgetting
        'checkpoint': './checkpoints/',
        'train_epochs': 4,        # Spend more time optimizing each batch on GPU
        # 'arenaCompare': 10,       # Number of games to play for the progress logger
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