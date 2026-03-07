from alpha_worms.reflex_endpoint.game_engine import GameEngine
from alpha_worms.nn import WormsPredictor
from alpha_worms.mcts import MCTS
# import concurrent.futures
import numpy as np
from collections import deque
import torch
import torch.multiprocessing as mp
import random
import pickle
import os
from datetime import datetime

class Worms:
    def __init__(self, game: GameEngine, nnet: WormsPredictor, args):
        self.game = game
        self.nnet = nnet
        self.args = args  
        self.train_data = deque(maxlen=args.get('max_buffer_size', 50000))

    @staticmethod
    def execute_episode(game, mcts=None, tensor_engine=None):
        train_examples = []
        state = game.get_initial_state()
        curr_player = 1 

        while True:
            canonical_state = game.get_canonical_form(state)
            pi = mcts.get_action_probs(canonical_state, temp=1)
            
            # Store the DTO temporarily during the game
            train_examples.append([canonical_state, pi, curr_player])

            action = np.random.choice(len(pi), p=pi)
            next_s = game.get_next_state(state, action)
            
            # Link the history for the NEXT step's canonicalization
            next_s.previous_dto = state 
            
            turn_switched = next_s.board_map.current_player != state.board_map.current_player
            state = next_s

            if turn_switched:
                curr_player *= -1

            reward = game.get_winner(state)

            if reward != 0:
                # --- THE FIX: Convert to Tensors and break links ---
                final_examples = []
                for hist_state_dto, hist_pi, hist_player in train_examples:
                    # 1. Transform DTO + History -> Stacked Tensor
                    # Assuming 'game' has access to the to_temporal_tensor method
                    raw_tensor = tensor_engine.to_temporal_tensor(hist_state_dto)
                    state_tensor = np.array(raw_tensor, copy=True, dtype=np.float32)
                    # 2. Calculate perspective-correct reward
                    z = reward * ((-1)**(hist_player != 1))
                    
                    # 3. Store only the raw data (tensor, pi, z)
                    final_examples.append((state_tensor, hist_pi, z))
                    
                    # 4. Explicitly break link to help GC (optional but safer)
                    hist_state_dto.previous_dto = None
                
                train_examples.clear()
                return final_examples

    def load_model(self):
        """
        Scans the checkpoint folder for the latest saved model and loads it.
        Returns the iteration number it found so the loop can resume properly.
        """
        folder = self.args['checkpoint']
        if not os.path.exists(folder):
            print("No checkpoint folder found. Starting fresh.")
            return 0

        # Find all .pth.tar files
        files = [f for f in os.listdir(folder) if f.startswith("checkpoint_") and f.endswith(".pth.tar")]
        if not files:
            print("No checkpoints found. Starting fresh.")
            return 0

        # Extract iteration numbers and find the highest one
        latest_iter = 0
        latest_file = ""
        for f in files:
            try:
                iter_num = int(f.split("_")[1].split(".")[0])
                if iter_num > latest_iter:
                    latest_iter = iter_num
                    latest_file = f
            except ValueError:
                continue

        if latest_file:
            print(f"Loading checkpoint: {latest_file}")
            self.nnet.load_checkpoint(folder=folder, filename=latest_file)
            return latest_iter
        
        return 0

    def load_history(self):
        """
        Loads all previous self-play games from the history folder into the deque.
        """
        history_folder = os.path.join(self.args['checkpoint'], "history")
        if not os.path.exists(history_folder):
            print("No history folder found. Starting with an empty buffer.")
            return

        print("Loading history from disk... This might take a moment.")
        # Sort files so we load them chronologically
        files = sorted([f for f in os.listdir(history_folder) if f.endswith(".pkl")])
        
        loaded_moves = 0
        for file in files:
            filepath = os.path.join(history_folder, file)
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    # Because train_data is a deque with maxlen, it will safely 
                    # push out old moves if the loaded files exceed max_buffer_size
                    self.train_data.extend(data)
                    loaded_moves += len(data)
            except Exception as e:
                print(f"Failed to load {file}: {e}")
                
        print(f"Successfully loaded {loaded_moves} total moves. Current buffer size: {len(self.train_data)}")

    def learn(self):
        """
        True AlphaZero IPC Multiprocessing Pipeline.
        """
        if self.nnet is None:
            from alpha_worms.nn import WormsPredictor
            self.nnet = WormsPredictor()

        # --- NEW: LOAD PREVIOUS STATE ---
        start_iter = self.load_model()
        self.load_history()

        # Offset the loop so if we load checkpoint_5, we start training on iteration 6
        for i in range(start_iter + 1, start_iter + self.args['num_iterations'] + 1):
            print(f"\n--- Iteration {i} ---")
            
            num_episodes = self.args['num_episodes']
            num_workers = min(num_episodes, max(1, mp.cpu_count() - 2)) 
            
            request_queue = mp.Queue()
            parent_pipes, child_pipes = zip(*[mp.Pipe() for _ in range(num_workers)])
            
            processes = []
            for worker_id in range(num_workers):
                games_per_worker = num_episodes // num_workers
                if worker_id < (num_episodes % num_workers):
                    games_per_worker += 1
                
                p = mp.Process(
                    target=Worms.worker_play_games,
                    args=(worker_id, games_per_worker, request_queue, child_pipes[worker_id], 
                        self.args, self.game)
                )
                p.start()
                processes.append(p)

            # --- FIXED INDENTATION: This block MUST be inside the for loop ---
            iteration_train_data = []
            active_workers = num_workers
            requests_batch = []
            
            print(f"Started {num_workers} CPU workers. GPU Batching server listening...")
            
            while active_workers > 0:
                try:
                    req = request_queue.get(timeout=0.05) 
                    if req == "DONE":
                        active_workers -= 1
                        continue
                    elif req[0] == "DATA":
                        iteration_train_data.extend(req[1])
                        continue
                    requests_batch.append(req)
                except Exception:
                    pass
                
                if len(requests_batch) >= self.args['batch_size'] or (len(requests_batch) > 0 and request_queue.empty()):
                    worker_ids = [r[0] for r in requests_batch]
                    states = [r[1] for r in requests_batch]
                    policies, values = self.nnet.predict_batch(states)
                    for idx, w_id in enumerate(worker_ids):
                        parent_pipes[w_id].send((policies[idx], values[idx]))
                    requests_batch = []

            for p in processes:
                p.join()
                
            # --- FIX 3: CLOSE MULTIPROCESSING RESOURCES ---
            request_queue.close()
            request_queue.join_thread() # Kill the background buffer thread
            for p_pipe, c_pipe in zip(parent_pipes, child_pipes):
                p_pipe.close()
                c_pipe.close()
            # --- PERSISTENCE LAYER ---
            if len(iteration_train_data) > 0:
                self.save_iteration_data(i, iteration_train_data)
                self.train_data.extend(iteration_train_data)
                # Removed the slice logic here; deque handles sizing automatically!

            # --- TRAINING LAYER ---
            if len(self.train_data) > 0:
                print(f"Training on {len(self.train_data)} moves...")
                self.train_network(epochs=self.args.get('train_epochs', 10))
                
            self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=f'checkpoint_{i}.pth.tar')
    
    def save_iteration_data(self, iteration, data):
        folder = os.path.join(self.args['checkpoint'], "history")
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        filename = os.path.join(folder, f"iteration_{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Successfully saved {len(data)} moves to {filename}")

    def train_network(self, epochs=1):
        """
        Samples a batch from the buffer and trains the NN.
        """
        for epoch in range(epochs):
            # We sample a random batch to break the correlation between consecutive moves
            batch_size = min(len(self.train_data), self.args['batch_size'])
            batch = random.sample(self.train_data, batch_size)
            # need to order the batch examples and convert them to tensor history
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

        # Each worker gets its own CPU-based tensor engine for DTO conversion
        from alpha_worms.reflex_endpoint.tensor_engine import TensorEngine
        local_tensor_engine = TensorEngine()
        
        worker_train_data = []
        
        for game_num in range(1, num_games + 1):
            # 2. Initialize a fresh MCTS for this game using the PROXY
            mcts = MCTS(game, proxy_nnet, args)
            
            # NOTE: You will need to ensure your self.execute_episode() 
            # uses this specific 'mcts' instance! 
            # Example: game_history = self.execute_episode(mcts)
            game_history = Worms.execute_episode(game, mcts=mcts,tensor_engine=local_tensor_engine)
            
            worker_train_data.extend(game_history)
            print(f"Worker {worker_id} completed game {game_num}/{num_games}")
            
        # 3. Game over! Send the training data to the Chef
        request_queue.put(("DATA", worker_train_data))
        
        # --- FIX 4: FREE WORKER MEMORY IMMEDIATELY ---
        worker_train_data = []
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

        from alpha_worms.reflex_endpoint.tensor_engine import TensorEngine
        self.tensor_engine = TensorEngine()

    def predict(self, state):
        # 1. Convert DTO to a simple NumPy array/Tensor INSIDE the worker
        # This is much "lighter" to pickle than a full DTO object
        raw_tensor = self.tensor_engine.to_temporal_tensor(state)
        # If it's already a Torch Tensor, move it to CPU and convert to Numpy 
        # to ensure the most efficient pickling across the IPC pipe.
        if isinstance(raw_tensor, torch.Tensor):
            raw_tensor = raw_tensor.detach().cpu().numpy()

        safe_tensor = np.array(raw_tensor, copy=True, dtype=np.float32)
        # 2. Send the raw data, not the DTO
        self.request_queue.put((self.worker_id, safe_tensor))
        
        policy, value = self.worker_pipe.recv() 
        return policy, value