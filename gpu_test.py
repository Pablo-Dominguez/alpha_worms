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
    predictor = WormsPredictor()

    # 2. Define STRESS TEST Hyperparameters for RTX 3050 Ti
    args = {
        'lr': 0.001,
        'numMCTSSims': 50,        # Up from 10. Forces the MCTS to search deeper and query the GPU more often.
        'cpuct': 1.41,
        'num_iterations': 3,      # 3 full loops of Play -> Train -> Save
        'num_episodes': 4,       # THE MAGIC NUMBER: Spawns 32 threads. Fills your GPU queue for maximum batching efficiency!
        'batch_size': 64,         # Up from 8. Safe for 4GB VRAM during the neural network backpropagation phase.
        'max_buffer_size': 10000,
        'checkpoint': './checkpointsv2/'
    }

    # 3. Initialize the Trainer
    worms_trainer = Worms(game, predictor, args)

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