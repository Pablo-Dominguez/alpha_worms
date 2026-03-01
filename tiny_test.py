from alpha_worms.reflex_endpoint.game_engine import GameEngine
from alpha_worms.nn import WormsPredictor
from alpha_worms.main import Worms
import torch

# MUST BE AT THE TOP: Prevents CPU thrashing in the workers
torch.set_num_threads(1)


if __name__ == '__main__':

    # 1. Initialize Game and Brain
    game = GameEngine() 
    predictor = WormsPredictor()

    # 2. Define TINY Hyperparameters for testing
    args = {
        'lr': 0.001,
        'numMCTSSims': 10,        # Reduced from 100. Just enough to build a tiny tree.
        'cpuct': 1.41,
        'num_iterations': 2,      # Reduced from 50. Just enough to test the loop and save state.
        'num_episodes': 2,        # Reduced from 10. Two games per iteration.
        'batch_size': 8,          # Reduced from 64. 
        'max_buffer_size': 1000,
        'checkpoint': './checkpointsv2/'
    }

    # 3. Initialize the Trainer
    worms_trainer = Worms(game, predictor, args)

    # 4. START TRAINING
    # This will run for a while, saving a .pth.tar file after every iteration
    worms_trainer.learn()