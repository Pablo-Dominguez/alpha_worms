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
        'numMCTSSims': 100,       # REDUCED: 100 sims strikes the perfect balance between smart moves and speed.
        'cpuct': 1.25,            
        'num_iterations': 1,     # REDUCED: 25 iterations will give a clear learning curve.
        'num_episodes': 2,       # REDUCED: 40 games / 5 workers = 8 games per worker per iteration. Very manageable.
        'batch_size': 64,         
        'max_buffer_size': 10000, # Cap the buffer slightly to save RAM over the 24 hours.
        'checkpoint': './checkpoints/',
        'train_epochs': 2
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