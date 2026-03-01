from alpha_worms.reflex_endpoint.game_engine import GameEngine
from alpha_worms.nn import WormsPredictor
from alpha_worms.main import Worms
import torch

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
        'num_iterations': 25,     # REDUCED: 25 iterations will give a clear learning curve.
        'num_episodes': 40,       # REDUCED: 40 games / 5 workers = 8 games per worker per iteration. Very manageable.
        'batch_size': 64,         
        'max_buffer_size': 10000, # Cap the buffer slightly to save RAM over the 24 hours.
        'checkpoint': './checkpoints/'
    }

    # 3. Initialize the Trainer
    worms_trainer = Worms(game, predictor, args)

    # 4. START TRAINING
    # This will run for a while, saving a .pth.tar file after every iteration
    worms_trainer.learn()