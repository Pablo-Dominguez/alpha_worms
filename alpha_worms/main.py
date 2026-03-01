from alpha_worms.reflex_endpoint.game_engine import GameEngine
from alpha_worms.nn import WormsPredictor
from alpha_worms.mcts import MCTS
# import concurrent.futures
import numpy as np
from collections import deque
import torch.multiprocessing as mp
import random

class Worms:
    def __init__(self, game: GameEngine, nnet: WormsPredictor, args):
        self.game = game
        self.nnet = nnet
        self.args = args  
        self.train_data = deque(maxlen=args['max_buffer_size'])

    @staticmethod
    def execute_episode(game, mcts=None):
        """
        Executes one full game of self-play using MCTS.
        """
        train_examples = []
        state = game.get_initial_state()
        
        # curr_player tracks the 'real' identity (1 or -1)
        # so we can assign the correct reward at the end.
        curr_player = 1 

        # We refresh MCTS for every new game to clear the tree

        while True:
            # 1. CANONICALIZE: Always present the board to the MCTS/NN 
            # as if it is Player 1's turn.
            canonical_state = game.get_canonical_form(state)
            
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
                next_s = game.get_next_state(state, action)
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
            reward = game.get_winner(state)

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
        True AlphaZero IPC Multiprocessing Pipeline.
        """
        if self.nnet is None:
            from alpha_worms.nn import WormsPredictor
            self.nnet = WormsPredictor()

        for i in range(1, self.args['num_iterations'] + 1):
            print(f"\n--- Iteration {i}/{self.args['num_iterations']} ---")
            
            num_episodes = self.args['num_episodes']
            # Cap workers to your CPU cores to prevent thrashing
            num_workers = min(num_episodes, max(1, mp.cpu_count() - 2)) 
            
            # 1. BUILD THE IPC PIPELINE
            request_queue = mp.Queue()
            # Create a dedicated pipe for each worker so the GPU can send answers back
            parent_pipes, child_pipes = zip(*[mp.Pipe() for _ in range(num_workers)])
            
            # 2. SPAWN THE WAITERS (CPU WORKERS)
            processes = []
            for worker_id in range(num_workers):
                # Calculate how many games this specific worker should play
                games_per_worker = num_episodes // num_workers
                if worker_id < (num_episodes % num_workers):
                    games_per_worker += 1
                
                p = mp.Process(
                    target=Worms.worker_play_games, # Call via Class, not self!
                    args=(worker_id, games_per_worker, request_queue, child_pipes[worker_id], 
                        self.args, self.game) # Only pass picklable data!
                )
                p.start()
                processes.append(p)

            # 3. THE CHEF: MAIN GPU BATCHING LOOP
            iteration_train_data = []
            active_workers = num_workers
            requests_batch = []
            
            print(f"Started {num_workers} CPU workers. GPU Batching server listening...")
            
            while active_workers > 0:
                # Try to scoop a request from the queue
                try:
                    # Timeout is crucial! It allows the GPU to process half-full batches 
                    # if workers are taking too long, preventing deadlocks.
                    req = request_queue.get(timeout=0.05) 
                    
                    # If a worker sends 'DONE', they finished their games
                    if req == "DONE":
                        active_workers -= 1
                        continue
                    elif req[0] == "DATA":
                        # Worker sends completed game data back to master buffer
                        iteration_train_data.extend(req[1])
                        continue
                        
                    requests_batch.append(req) # req is (worker_id, state)
                except Exception: # Queue is empty
                    pass
                
                # If we hit our batch limit OR the queue is empty but we have requests waiting
                if len(requests_batch) >= self.args['batch_size'] or (len(requests_batch) > 0 and request_queue.empty()):
                    
                    # Unpack the batch
                    worker_ids = [r[0] for r in requests_batch]
                    states = [r[1] for r in requests_batch]
                    
                    # --- DIRECT GPU INFERENCE ---
                    # (You will need to ensure your predictor can accept a list of states!)
                    policies, values = self.nnet.predict_batch(states)
                    
                    # Send the answers back down the specific pipes
                    for idx, w_id in enumerate(worker_ids):
                        parent_pipes[w_id].send((policies[idx], values[idx]))
                    
                    requests_batch = [] # Reset for next batch

            # Wait for all processes to clean up gracefully
            for p in processes:
                p.join()
                
            self.train_data.extend(iteration_train_data)
            print(f"Added {len(iteration_train_data)} moves. Total buffer: {len(self.train_data)}")

            # -----------------------------------------------------------
            # PHASE 2 & 3: TRAIN AND SAVE (Unchanged)
            # -----------------------------------------------------------
            if len(self.train_data) > 0:
                self.train_network()
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
    
    @staticmethod
    def worker_play_games(worker_id, num_games, request_queue, worker_pipe, args, game):
        """
        The isolated process that runs the games and MCTS on a single CPU core.
        """
        # 1. Create the fake predictor
        proxy_nnet = IPCPredictorProxy(worker_id, request_queue, worker_pipe)
        
        worker_train_data = []
        
        for game_num in range(1, num_games + 1):
            # 2. Initialize a fresh MCTS for this game using the PROXY
            mcts = MCTS(game, proxy_nnet, args)
            
            # NOTE: You will need to ensure your self.execute_episode() 
            # uses this specific 'mcts' instance! 
            # Example: game_history = self.execute_episode(mcts)
            game_history = Worms.execute_episode(game, mcts=mcts) 
            
            worker_train_data.extend(game_history)
            print(f"Worker {worker_id} completed game {game_num}/{num_games}")
            
        # 3. Game over! Send the training data to the Chef
        request_queue.put(("DATA", worker_train_data))
        
        # 4. Tell the Chef this worker is clocking out
        request_queue.put("DONE")
        
class IPCPredictorProxy:
    """
    A fake neural network that simply forwards requests to the GPU Chef 
    and waits for the answer.
    """
    def __init__(self, worker_id, request_queue, worker_pipe):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.worker_pipe = worker_pipe

    def predict(self, state):
        # 1. Slide the order (board state) into the kitchen queue
        self.request_queue.put((self.worker_id, state))
        
        # 2. Wait at the table (pipe) for the Chef to send the food back
        policy, value = self.worker_pipe.recv() 
        
        return policy, value