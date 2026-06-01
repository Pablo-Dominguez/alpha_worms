from alpha_worms.reflex_endpoint.game_engine import GameEngine
from alpha_worms.nn import WormsPredictor
from alpha_worms.main import Worms
import torch
import time


# MUST BE AT THE TOP: Prevents CPU thrashing in the workers
torch.set_num_threads(1)


if __name__ == '__main__':
    # This ensures only the main process kicks off the training
    # The workers will skip over this entirely.
    # 1. Initialize Game and Brain
    game = GameEngine() 
    predictor = WormsPredictor()

    # 2. Define TINY Hyperparameters for testing
    args = {
        'lr': 0.001,              
        # Sims: On CPU, 25-40 is enough to see if MCTS is actually 'thinking'
        # without waiting 5 minutes per move.
        'numMCTSSims': 25,       
        
        'cpuct': 1.25,            
        
        # Iterations: Keep this at 1 or 2 for a "Smoke Test"
        'num_iterations': 2,     
        
        # Episodes: Very low for a test run. This will result in 1 game 
        # per worker if you have 2 workers.
        'num_episodes': 2,       
        
        # Batch Size: Smaller batches are better for CPU memory cache.
        'batch_size': 32,         
        
        'max_buffer_size': 1000, 
        'checkpoint': './checkpoints/',
        
        # Train Epochs: 1 is enough to verify the backpropagation 
        # logic works without overheating the CPU.
        'train_epochs': 1         
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