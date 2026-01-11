from alpha_worms.reflex_endpoint.game_engine import GameEngine
from alpha_worms.nn import WormsPredictor
from alpha_worms.main import Worms

# 1. Initialize Game and Brain
game = GameEngine() 
predictor = WormsPredictor(device="cuda")

# 2. Define Hyperparameters
args = {
    'lr': 0.001,
    'numMCTSSims': 100,
    'cpuct': 1.41,
    'num_iterations': 50,
    'num_episodes': 10,
    'batch_size': 64,
    'max_buffer_size': 20000,
    'checkpoint': './checkpoints/'
}

# 3. Initialize the Trainer
worms_trainer = Worms(game, predictor, args)

# 4. START TRAINING
# This will run for a while, saving a .pth.tar file after every iteration
worms_trainer.learn()