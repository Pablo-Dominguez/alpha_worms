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
import shutil
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
        turn_count = 0        # 1. Add a turn counter
        max_turns = 150       # 2. Set a reasonable upper limit for a game of Worms

        while True:
            turn_count += 1

            canonical_state = game.get_canonical_form(state)
            # Use temp=1 for the first 30 moves, then temp=0.2 to sharpen the policy
            temp = 1 if turn_count < 30 else 0.1
            pi = mcts.get_action_probs(canonical_state, temp=temp)
            
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

            # 4. Check for a forced draw if the game drags on too long
            if turn_count >= max_turns and reward == 0:
                # Use the point calculation to see who was winning
                reward = game.get_score_winner(state)
                
                # If it's a dead tie on points, use a tiny value to represent a draw
                if reward == 0:
                    reward = 1e-5
                print(f"Game hit turn limit ({max_turns})! Winner by points: {reward}")

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

    def get_best_checkpoint_path(self):
        return os.path.join(self.args['checkpoint'], 'checkpoint_best.pth.tar')

    def save_best_checkpoint(self, source_filename):
        best_path = self.get_best_checkpoint_path()
        if not os.path.exists(self.args['checkpoint']):
            os.makedirs(self.args['checkpoint'], exist_ok=True)
        shutil.copyfile(os.path.join(self.args['checkpoint'], source_filename), best_path)
        print(f"Saved best model to {best_path}")

    def play_arena_game(self, mcts_p1, mcts_p2, max_turns=200):
        state = self.game.get_initial_state()
        turn_count = 0

        while True:
            turn_count += 1
            current_player = state.board_map.current_player
            canonical_state = self.game.get_canonical_form(state)

            active_mcts = mcts_p1 if current_player == 1 else mcts_p2
            pi = active_mcts.get_action_probs(canonical_state, temp=0)
            action = int(np.argmax(pi))
            state = self.game.get_next_state(state, action)

            winner = self.game.get_winner(state)
            if winner != 0:
                return winner

            if turn_count >= max_turns:
                score_winner = self.game.get_score_winner(state)
                if score_winner == 0:
                    return 0
                return score_winner

    def arena_compare(self, challenger, baseline):
        arena_games = self.args.get('arenaCompare', 20)
        arena_threshold = self.args.get('arenaThreshold', 0.55)
        sim_multiplier = self.args.get('arenaSimMultiplier', 2)

        if arena_games <= 0:
            print("Arena comparison disabled (arenaCompare <= 0).")
            return True

        print(f"Starting arena comparison: {arena_games} games, threshold={arena_threshold}, sims={self.args.get('numMCTSSims')}x{sim_multiplier}")

        arena_args = self.args.copy()
        arena_args['numMCTSSims'] = int(self.args.get('numMCTSSims', 80) * sim_multiplier)

        challenger_wins = 0
        baseline_wins = 0
        draws = 0

        for game_idx in range(1, arena_games + 1):
            if game_idx % 2 == 1:
                p1_nnet, p2_nnet = challenger, baseline
                challenger_is_p1 = True
            else:
                p1_nnet, p2_nnet = baseline, challenger
                challenger_is_p1 = False

            mcts_p1 = MCTS(self.game, p1_nnet, arena_args)
            mcts_p2 = MCTS(self.game, p2_nnet, arena_args)
            result = self.play_arena_game(mcts_p1, mcts_p2)

            if result == 1:
                if challenger_is_p1:
                    challenger_wins += 1
                else:
                    baseline_wins += 1
            elif result == -1:
                if challenger_is_p1:
                    baseline_wins += 1
                else:
                    challenger_wins += 1
            else:
                draws += 1

            print(f"Arena game {game_idx}/{arena_games}: challenger={challenger_wins}, baseline={baseline_wins}, draws={draws}")

        win_rate = challenger_wins / arena_games
        print(f"Arena results: challenger wins={challenger_wins}, baseline wins={baseline_wins}, draws={draws}, win_rate={win_rate:.3f}")

        if win_rate >= arena_threshold:
            print(f"Challenger passed the arena threshold ({arena_threshold}).")
            return True

        print(f"Challenger failed the arena threshold ({arena_threshold}).")
        return False

    def learn(self):
        """
        True AlphaZero IPC Multiprocessing Pipeline.
        """

        os.nice(10)
        if self.nnet is None:
            from alpha_worms.nn import WormsPredictor
            self.nnet = WormsPredictor()

        # --- NEW: LOAD PREVIOUS STATE ---
        start_iter = self.load_model()
        self.load_history()

        for i in range(start_iter + 1, start_iter + self.args['num_iterations'] + 1):
            print(f"\n--- Iteration {i} ---")
            
            baseline_predictor = None
            if self.args.get('arenaCompare', 0) > 0:
                best_path = self.get_best_checkpoint_path()
                if os.path.exists(best_path):
                    baseline_predictor = WormsPredictor()
                    baseline_predictor.load_checkpoint(folder=self.args['checkpoint'], filename='checkpoint_best.pth.tar')
                else:
                    print("No baseline best checkpoint found yet. First candidate will become the baseline automatically.")

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
                
            request_queue.close()
            request_queue.join_thread()
            for p_pipe, c_pipe in zip(parent_pipes, child_pipes):
                p_pipe.close()
                c_pipe.close()

            if len(iteration_train_data) > 0:
                self.save_iteration_data(i, iteration_train_data)
                self.train_data.extend(iteration_train_data)

            if len(self.train_data) > 0:
                print(f"Training on {len(self.train_data)} moves...")
                self.train_network(epochs=self.args.get('train_epochs', 10))

            checkpoint_name = f'checkpoint_{i}.pth.tar'
            self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=checkpoint_name)

            if baseline_predictor is not None:
                challenger_passed = self.arena_compare(self.nnet, baseline_predictor)
                if challenger_passed:
                    self.save_best_checkpoint(checkpoint_name)
                else:
                    print("Reloading baseline model after arena rejection.")
                    best_path = self.get_best_checkpoint_path()
                    if os.path.exists(best_path):
                        self.nnet.load_checkpoint(folder=self.args['checkpoint'], filename='checkpoint_best.pth.tar')
            elif self.args.get('arenaCompare', 0) > 0:
                # first accepted model becomes the baseline
                self.save_best_checkpoint(checkpoint_name)
    
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