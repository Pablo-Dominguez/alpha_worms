from alpha_worms.reflex_endpoint.game_engine import GameEngine
from alpha_worms.nn import WormsPredictor
from alpha_worms.mcts import MCTS
import concurrent.futures
import numpy as np
from collections import deque
import random

class Worms:
    def __init__(self, game: GameEngine, nnet: WormsPredictor, args):
        self.game = game
        self.nnet = nnet
        self.args = args  
        self.train_data = deque(maxlen=args['max_buffer_size'])

    def execute_episode(self):
        """
        Executes one full game of self-play using MCTS.
        """
        train_examples = []
        state = self.game.get_initial_state()
        
        # curr_player tracks the 'real' identity (1 or -1)
        # so we can assign the correct reward at the end.
        curr_player = 1 

        # We refresh MCTS for every new game to clear the tree
        mcts = MCTS(self.game, self.nnet, self.args)

        while True:
            # 1. CANONICALIZE: Always present the board to the MCTS/NN 
            # as if it is Player 1's turn.
            canonical_state = self.game.get_canonical_form(state)
            
            # 2. MCTS SEARCH: Get move probabilities for the canonical state
            pi = mcts.get_action_probs(canonical_state, temp=1) # should review temp usage
            
            # 3. STORE EXPERIENCE: Record the state from the perspective 
            # of the player whose turn it was.
            train_examples.append([canonical_state, pi, curr_player]) # review curr_player usage

            # 4. ACTION: Pick a move
            action = np.random.choice(len(pi), p=pi)
            
            # 5. TRANSITION: Get next state and check if the turn switched
            # Based on our previous discussion, your engine returns (state, turn_switched)
            
            try:
                next_s = self.game.get_next_state(state, action)
            except Exception as e:
                print('Pi',pi)
                print('Selected action',action)
                print(f"Error in get_next_state: {e}")
                raise e
            turn_switched = next_s.board_map.current_player != state.board_map.current_player
            state = next_s

            if turn_switched:
                curr_player *= -1

            # 6. TERMINAL CHECK
            reward = self.game.get_winner(state)

            if reward != 0:
                # Game over! 
                # 'reward' is usually 1 (P1 wins), -1 (P2 wins), or 0.1 (draw)
                # We return a list of (state, pi, z) where z is the result 
                # from the perspective of the player who was acting at that state.
                return [
                    (hist_state, hist_pi, reward * ((-1)**(hist_player != 1))) 
                    for hist_state, hist_pi, hist_player in train_examples
                ]

    def learn(self):
        """
        The main training loop, supercharged for multi-threaded GPU batching.
        """
        # (Deleted all the mp.spawn and share_memory stuff!)

        for i in range(1, self.args['num_iterations'] + 1):
            print(f"\n--- Iteration {i}/{self.args['num_iterations']} ---")
            
            # -----------------------------------------------------------
            # PHASE 1: MULTI-THREADED SELF-PLAY
            # -----------------------------------------------------------
            iteration_train_data = []
            num_episodes = self.args['num_episodes']
            
            # Threads are lightweight! We can easily spawn one thread per game 
            # (as long as it doesn't exceed our GPU queue batch size of 32).
            num_workers = min(num_episodes, 32) 
            print(f"Spawning {num_workers} threads to play {num_episodes} games...")
            
            # Changed to ThreadPoolExecutor and removed mp_context
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 1. Dispatch all games to the worker threads
                futures = [executor.submit(self.execute_episode) for _ in range(num_episodes)]
                
                # 2. Collect the data as each thread finishes a game
                for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    try:
                        game_history = future.result()
                        iteration_train_data.extend(game_history)
                        
                        if count % max(1, (num_episodes // 5)) == 0:
                            print(f"  -> Completed {count}/{num_episodes} games...")
                    except Exception as e:
                        print(f"A worker thread crashed: {e}")

            self.train_data.extend(iteration_train_data)
            print(f"Added {len(iteration_train_data)} moves to buffer. Total buffer: {len(self.train_data)}")

            # -----------------------------------------------------------
            # PHASE 2: TRAINING
            # -----------------------------------------------------------
            # Safety check in case threads failed
            if len(self.train_data) == 0:
                print("⚠️ Warning: Buffer is empty! Skipping training this iteration.")
                continue

            self.train_network()

            # -----------------------------------------------------------
            # PHASE 3: SAVE
            # -----------------------------------------------------------
            self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=f'checkpoint_{i}.pth.tar')

    def train_network(self):
        """
        Samples a batch from the buffer and trains the NN.
        """
        # We sample a random batch to break the correlation between consecutive moves
        batch_size = min(len(self.train_data), self.args['batch_size'])
        batch = random.sample(self.train_data, batch_size)
        
        # The predictor.train method we built previously expects a list of examples
        # and handles the tensor conversion itself.
        try:
            loss = self.nnet.train(batch)
            print(f"Training Loss: {loss:.4f}")
        except Exception as e:
            # Save batch to text file for debugging
            with open('training_batch_debug.txt', 'w') as f:
                for example in batch:
                    f.write(f"{example}\n")
            print(f"Error during training: {e}")
            raise e