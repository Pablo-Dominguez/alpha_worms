from alpha_worms.reflex_endpoint.game_engine import GameEngine
from alpha_worms.nn import WormsPredictor
from alpha_worms.main import Worms

# 1. Initialize Game and Brain
game = GameEngine() 
predictor = WormsPredictor()

# 2. Define TINY Hyperparameters for testing
args = {
    'lr': 0.001,              
    'numMCTSSims': 50,        # REDUCED: 50 is the sweet spot for CPU. Deep enough to beat a random player, fast enough to finish games.
    'cpuct': 1.41,            # Back to standard. With fewer sims, we need to encourage a bit more exploration.
    'num_iterations': 15,     # REDUCED: We want to ensure it completes enough full cycles to show improvement.
    'num_episodes': 25,       # REDUCED: 25 games per iteration gives a good batch of data without taking 3 hours per loop.
    'batch_size': 32,         # REDUCED: 32 is much friendlier on CPU cache and RAM during backpropagation than 64.
    'max_buffer_size': 5000,  # REDUCED: Holds about the last 4-5 iterations of history.
    'checkpoint': './checkpoints/'
}

# 3. Initialize the Trainer
worms_trainer = Worms(game, predictor, args)

# 4. START TRAINING
# This will run for a while, saving a .pth.tar file after every iteration
worms_trainer.learn()