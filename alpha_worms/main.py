from alpha_worms.reflex_endpoint.game_engine import GameEngine
from alpha_worms.nn import WormsPredictor
from alpha_worms.mcts import MCTS
import numpy as np
import queue
import threading
from collections import deque
import torch
import torch.multiprocessing as mp
import random
import pickle
import os
import shutil
from datetime import datetime
import gc
import logging
import sys

# Spawn method is set inside learn() to ensure it applies regardless of entry script.

# Configure logging for worker heartbeat
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

    def _get_latest_checkpoint_iter(self):
        """
        Scans the checkpoint folder and returns the highest iteration number without loading weights.
        Used when the best checkpoint will be loaded anyway, making a full load_model() wasteful.
        """
        folder = self.args['checkpoint']
        if not os.path.exists(folder):
            return 0
        latest_iter = 0
        for f in os.listdir(folder):
            if f.startswith("checkpoint_") and f.endswith(".pth.tar"):
                try:
                    iter_num = int(f.split("_")[1].split(".")[0])
                    if iter_num > latest_iter:
                        latest_iter = iter_num
                except ValueError:
                    continue
        return latest_iter

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

    def check_gpu_memory_usage(self):
        """
        FIX 4: Helper to check GPU memory usage percentage.
        Returns percentage (0-100) or None if CUDA not available.
        """
        if not torch.cuda.is_available():
            return None
        try:
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            usage_pct = (allocated / total) * 100
            return usage_pct
        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}")
            return None

    def save_best_checkpoint(self, source_filename, iteration):
        best_path = self.get_best_checkpoint_path()
        if not os.path.exists(self.args['checkpoint']):
            os.makedirs(self.args['checkpoint'], exist_ok=True)
        shutil.copyfile(os.path.join(self.args['checkpoint'], source_filename), best_path)
        # Save the iteration number for resume
        with open(os.path.join(self.args['checkpoint'], 'best_iteration.txt'), 'w') as f:
            f.write(str(iteration))
        print(f"Saved best model to {best_path} (iteration {iteration})")

    def play_arena_game(self, mcts_p1, mcts_p2, max_turns=200):
        # This method is now unused since we use multi-threaded play_single_arena_game
        pass

    def arena_compare(self, challenger, baseline):
        arena_games = self.args.get('arenaCompare', 20)
        arena_threshold = self.args.get('arenaThreshold', 0.55)
        sim_multiplier = self.args.get('arenaSimMultiplier', 2)

        if arena_games <= 0:
            print("Arena comparison disabled (arenaCompare <= 0).")
            return True

        print(f"Starting arena comparison: {arena_games} games, threshold={arena_threshold}, sims={self.args.get('numMCTSSims')}x{sim_multiplier}")
        
        gpu_usage = self.check_gpu_memory_usage()
        if gpu_usage is not None:
            logger.info(f"GPU memory usage before arena: {gpu_usage:.1f}%")

        arena_args = self.args.copy()
        arena_args['numMCTSSims'] = int(self.args.get('numMCTSSims', 80) * sim_multiplier)

        # Dual-model batching setup for arena
        arena_queue = queue.Queue()
        batching_thread = threading.Thread(target=self.batch_arena_requests, args=(arena_queue, challenger, baseline), daemon=True)
        batching_thread.start()

        # Multi-threaded arena evaluation using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        num_workers = min(arena_games, 4)  # Limit to 4 threads to avoid GPU overload
        
        results = []
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(self.play_single_arena_game, challenger, baseline, arena_args, arena_queue, game_idx % 2 == 0)
                    for game_idx in range(arena_games)
                ]
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        # A single game failure (e.g. 120s proxy TimeoutError) counts as a draw
                        # rather than crashing the entire arena comparison and killing the training loop.
                        logger.error(f"Arena game failed with exception: {e}. Counting as draw.")
                        results.append(0)
        finally:
            # Always stop the batching thread — even if a future raises an exception
            # (e.g. TimeoutError from the 120s proxy timeout), so it doesn't spin forever.
            arena_queue.put("STOP")
            batching_thread.join(timeout=2.0)

        challenger_wins = sum(1 for r in results if r == 1)
        baseline_wins = sum(1 for r in results if r == -1)
        draws = sum(1 for r in results if r == 0)

        decisive = challenger_wins + baseline_wins
        win_rate = challenger_wins / decisive if decisive > 0 else 0.5
        print(f"Arena results: challenger wins={challenger_wins}, baseline wins={baseline_wins}, draws={draws}, win_rate={win_rate:.3f} ({decisive} decisive)")
        
        if win_rate >= arena_threshold:
            print(f"Challenger passed the arena threshold ({arena_threshold}).")
            return True

        print(f"Challenger failed the arena threshold ({arena_threshold}).")
        return False

    def batch_arena_requests(self, arena_queue, challenger, baseline):
        """
        Batches inference requests for both challenger and baseline models to optimize GPU usage.
        """
        batch_size = 8

        def flush(requests):
            challenger_reqs = [(i, r) for i, r in enumerate(requests) if r[0] is challenger]
            baseline_reqs   = [(i, r) for i, r in enumerate(requests) if r[0] is baseline]
            if challenger_reqs:
                states = [r[1] for _, r in challenger_reqs]
                policies, values = challenger.predict_batch(states)
                for idx, (orig_idx, _) in enumerate(challenger_reqs):
                    requests[orig_idx][3]['pi'] = policies[idx]
                    requests[orig_idx][3]['v']  = values[idx]
                    requests[orig_idx][2].set()
            if baseline_reqs:
                states = [r[1] for _, r in baseline_reqs]
                policies, values = baseline.predict_batch(states)
                for idx, (orig_idx, _) in enumerate(baseline_reqs):
                    requests[orig_idx][3]['pi'] = policies[idx]
                    requests[orig_idx][3]['v']  = values[idx]
                    requests[orig_idx][2].set()

        while True:
            requests = []
            try:
                while len(requests) < batch_size:
                    req = arena_queue.get(timeout=0.01)
                    if req == "STOP":
                        # Flush any already-collected requests so their threads
                        # don't hang on event.wait() for the full 120s timeout.
                        if requests:
                            flush(requests)
                        return
                    requests.append(req)
            except queue.Empty:
                pass

            if requests:
                flush(requests)

    def play_single_arena_game(self, challenger, baseline, arena_args, arena_queue, challenger_goes_second):
        """
        Play a single arena game between challenger and baseline.
        Returns 1 if challenger wins, -1 if baseline wins, 0 for draw.
        """
        if challenger_goes_second:
            p1_nnet, p2_nnet = baseline, challenger
        else:
            p1_nnet, p2_nnet = challenger, baseline

        # Use proxies for batched inference
        p1_proxy = ArenaPredictorProxy(p1_nnet, arena_queue)
        p2_proxy = ArenaPredictorProxy(p2_nnet, arena_queue)

        mcts_p1 = MCTS(self.game, p1_proxy, arena_args)
        mcts_p2 = MCTS(self.game, p2_proxy, arena_args)
        
        state = self.game.get_initial_state()
        turn_count = 0
        max_turns = 200

        while True:
            turn_count += 1
            current_player = state.board_map.current_player
            canonical_state = self.game.get_canonical_form(state)

            active_mcts = mcts_p1 if current_player == 1 else mcts_p2
            pi = active_mcts.get_action_probs(canonical_state, temp=0)
            action = int(np.argmax(pi))
            prev_state = state
            state = self.game.get_next_state(state, action)
            state.previous_dto = prev_state

            winner = self.game.get_winner(state)
            if winner != 0:
                if challenger_goes_second:
                    return -winner  # Invert because challenger is P2
                return winner

            if turn_count >= max_turns:
                score_winner = self.game.get_score_winner(state)
                if score_winner == 0:
                    return 0
                if challenger_goes_second:
                    return -score_winner
                return score_winner

    def learn(self):
        """
        True AlphaZero IPC Multiprocessing Pipeline.
        """
        # Must be set before any mp.Process is created, regardless of entry script.
        # fork + active CUDA context = guaranteed segfault in the child.
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        # Prevent OpenMP thread explosion: N workers × K threads = N×K threads fighting
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')

        os.nice(10)
        if self.nnet is None:
            from alpha_worms.nn import WormsPredictor
            self.nnet = WormsPredictor()

        # --- NEW: LOAD PREVIOUS STATE ---
        # When a best checkpoint exists, only scan for the iteration number — don't load
        # the latest checkpoint's weights since they'll be immediately overwritten.
        best_path = self.get_best_checkpoint_path()
        if os.path.exists(best_path):
            start_iter = self._get_latest_checkpoint_iter()
            self.nnet.load_checkpoint(folder=self.args['checkpoint'], filename='checkpoint_best.pth.tar')
            best_iter_file = os.path.join(self.args['checkpoint'], 'best_iteration.txt')
            if os.path.exists(best_iter_file):
                with open(best_iter_file, 'r') as f:
                    best_iter = int(f.read().strip())
                print(f"Loaded best model weights (from iteration {best_iter}). Resuming loop from checkpoint {start_iter}.")
            else:
                print(f"Loaded best model weights. Resuming loop from checkpoint {start_iter}.")
        else:
            # No best checkpoint yet — load the latest checkpoint's weights to continue training.
            start_iter = self.load_model()
        self.load_history()

        for i in range(start_iter + 1, start_iter + self.args['num_iterations'] + 1):
            print(f"\n--- Iteration {i} ---")
            
            baseline_predictor = None
            if self.args.get('arenaCompare', 0) > 0:
                best_path = self.get_best_checkpoint_path()
                if os.path.exists(best_path):
                    baseline_predictor = WormsPredictor()
                    baseline_predictor.nnet.to('cpu')  # move before load so checkpoint is read directly into RAM, not VRAM
                    baseline_predictor.load_checkpoint(folder=self.args['checkpoint'], filename='checkpoint_best.pth.tar', load_optimizer=False)
                else:
                    print("No baseline best checkpoint found yet. First candidate will become the baseline automatically.")

            num_episodes = self.args['num_episodes']
            num_workers = min(num_episodes, max(1, mp.cpu_count() - 2))
            
            request_queue = mp.Queue()
            parent_pipes, child_pipes = zip(*[mp.Pipe() for _ in range(num_workers)])
            
            processes = []
            iteration_train_data = []
            try:
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

                active_workers = num_workers
                requests_batch = []
                worker_timeout = 60
                last_activity_time = datetime.now()

                logger.info(f"Started {num_workers} CPU workers. GPU Batching server listening...")

                while active_workers > 0:
                    try:
                        req = request_queue.get(timeout=worker_timeout)
                        last_activity_time = datetime.now()

                        if req == "DONE":
                            active_workers -= 1
                            logger.info(f"Worker signaled completion. {active_workers} workers remaining.")
                            continue
                        elif req[0] == "DATA":
                            iteration_train_data.extend(req[1])
                            logger.info(f"Received {len(req[1])} training examples from worker.")
                            continue
                        requests_batch.append(req)
                    except queue.Empty:
                        # Queue is idle — flush any pending inference requests immediately.
                        # Without this, a worker blocked on pipe.recv() will never get a
                        # response if the queue drains via DATA/DONE before batch_size fills.
                        if len(requests_batch) > 0:
                            worker_ids = [r[0] for r in requests_batch]
                            states = [r[1] for r in requests_batch]
                            logger.info(f"Flushing {len(requests_batch)} pending inference requests on idle queue.")
                            policies, values = self.nnet.predict_batch(states)
                            for idx, w_id in enumerate(worker_ids):
                                parent_pipes[w_id].send((policies[idx], values[idx]))
                            requests_batch = []
                            last_activity_time = datetime.now()
                            continue

                        time_elapsed = (datetime.now() - last_activity_time).total_seconds()
                        logger.warning(f"No worker activity for {time_elapsed}s. Active workers: {active_workers}")

                        if time_elapsed > worker_timeout + 10:
                            logger.critical(f"Silent deadlock detected! Terminating {len(processes)} worker processes.")
                            for p in processes:
                                if p.is_alive():
                                    p.terminate()
                                    p.join(timeout=5)
                                    if p.is_alive():
                                        p.kill()
                            raise RuntimeError("Silent deadlock detected: workers hung for >70s")
                        continue
                    except Exception as e:
                        logger.error(f"Error receiving from queue: {e}")
                        pass

                    if len(requests_batch) >= self.args['batch_size'] or (len(requests_batch) > 0 and request_queue.empty()):
                        worker_ids = [r[0] for r in requests_batch]
                        states = [r[1] for r in requests_batch]
                        logger.debug(f"Processing GPU batch with {len(requests_batch)} requests")
                        policies, values = self.nnet.predict_batch(states)
                        for idx, w_id in enumerate(worker_ids):
                            parent_pipes[w_id].send((policies[idx], values[idx]))
                        requests_batch = []

            finally:
                # Guaranteed cleanup on normal exit, KeyboardInterrupt, or any exception.
                # terminate() before join() ensures workers that are blocked on pipe.recv()
                # don't hang the main process indefinitely.
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5)
                    if p.is_alive():
                        logger.warning(f"Process {p.pid} did not terminate cleanly. Killing...")
                        p.kill()
                request_queue.close()
                request_queue.cancel_join_thread()
                for p_pipe, c_pipe in zip(parent_pipes, child_pipes):
                    p_pipe.close()
                    c_pipe.close()

            if len(iteration_train_data) > 0:
                self.save_iteration_data(i, iteration_train_data)
                self.train_data.extend(iteration_train_data)
                # deque already enforces maxlen, so no explicit slice is needed here.

            if len(self.train_data) > 0:
                print(f"Training on {len(self.train_data)} moves...")
                self.train_network(epochs=self.args.get('train_epochs', 10))
            
            # FIX 3: Replay Buffer Capping & Garbage Collection
            logger.info(f"Current train_data buffer size: {len(self.train_data)}")
            max_buffer_size = self.args.get('max_buffer_size', 50000)
            if len(self.train_data) > max_buffer_size:
                # Trim buffer to maximum size (deque maxlen should handle this, but explicit check)
                while len(self.train_data) > max_buffer_size:
                    self.train_data.popleft()
                logger.info(f"Trimmed buffer to {len(self.train_data)} examples")
            
            # Explicitly call garbage collection and CUDA cache clearing
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

            checkpoint_name = f'checkpoint_{i}.pth.tar'
            self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=checkpoint_name)

            if baseline_predictor is not None:
                challenger_passed = self.arena_compare(self.nnet, baseline_predictor)
                if challenger_passed:
                    self.save_best_checkpoint(checkpoint_name, i)
                else:
                    print("Reloading baseline model after arena rejection.")
                    best_path = self.get_best_checkpoint_path()
                    if os.path.exists(best_path):
                        self.nnet.load_checkpoint(folder=self.args['checkpoint'], filename='checkpoint_best.pth.tar')
                del baseline_predictor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif self.args.get('arenaCompare', 0) > 0:
                # first accepted model becomes the baseline
                self.save_best_checkpoint(checkpoint_name, i)
    
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
                raise
    
    @staticmethod
    def worker_play_games(worker_id, num_games, request_queue, worker_pipe, args, game):
        """
        The isolated process that runs the games and MCTS on a single CPU core.
        FIX 3: Added timeout to pipe.recv() to detect parent crash.
        FIX 5: Added heartbeat logging every 5 games.
        """
        import time
        import torch as _torch
        _torch.set_num_threads(1)  # Prevent OpenMP explosion inside each worker process
        # Configure logging — logging.basicConfig() from the parent is NOT inherited by spawned
        # subprocesses. Without this, all logger.info/error calls below are completely silent.
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # 1. Create the fake predictor
        proxy_nnet = IPCPredictorProxy(worker_id, request_queue, worker_pipe)

        # Each worker gets its own CPU-based tensor engine for DTO conversion
        from alpha_worms.reflex_endpoint.tensor_engine import TensorEngine
        local_tensor_engine = TensorEngine()
        
        worker_train_data = []
        
        try:
            for game_num in range(1, num_games + 1):
                # 2. Initialize a fresh MCTS for this game using the PROXY
                mcts = MCTS(game, proxy_nnet, args)
                
                # NOTE: You will need to ensure your self.execute_episode() 
                # uses this specific 'mcts' instance! 
                # Example: game_history = self.execute_episode(mcts)
                game_history = Worms.execute_episode(game, mcts=mcts,tensor_engine=local_tensor_engine)
                
                worker_train_data.extend(game_history)
                
                # FIX 5: Heartbeat logging every 5 games
                if game_num % 5 == 0:
                    logger.info(f"Worker {worker_id} progress: {game_num}/{num_games} games, {len(worker_train_data)} examples")
                else:
                    print(f"Worker {worker_id} completed game {game_num}/{num_games}")
                
            # 3. Game over! Send the training data to the Chef
            request_queue.put(("DATA", worker_train_data))
            logger.info(f"Worker {worker_id} finished: sent {len(worker_train_data)} training examples")
            
        except Exception as e:
            logger.error(f"Worker {worker_id} crashed with error: {e}")
            request_queue.put(("DATA", worker_train_data))
        
        # --- FIX 4: FREE WORKER MEMORY IMMEDIATELY ---
        worker_train_data = []
        gc.collect()  # Explicitly free memory
        
        # 4. Tell the Chef this worker is clocking out
        request_queue.put("DONE")
        
class ArenaPredictorProxy:
    """
    A proxy for arena models that batches inference requests to reduce GPU contention.
    """
    def __init__(self, model, arena_queue):
        self.model = model
        self.arena_queue = arena_queue

    def predict(self, state):
        # Convert DTO to numpy array immediately to save memory
        raw_tensor = self.model.tensor_engine.to_temporal_tensor(state)
        safe_tensor = np.array(raw_tensor, copy=True, dtype=np.float32)
        event = threading.Event()
        result = {}
        self.arena_queue.put((self.model, safe_tensor, event, result))
        if not event.wait(timeout=120):
            raise TimeoutError("Arena batching thread stopped responding after 120s (crash or deadlock)")
        return result['pi'], result['v']


class IPCPredictorProxy:
    """
    A fake neural network that simply forwards requests to the GPU Chef 
    and waits for the answer.
    FIX 1: Added timeout to pipe.recv() to prevent indefinite blocking.
    """
    def __init__(self, worker_id, request_queue, worker_pipe):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.worker_pipe = worker_pipe
        self.pipe_timeout = 60  # FIX 1: 60-second timeout for pipe communication

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
        
        # FIX 1: Use timeout to detect if parent process crashed
        try:
            if self.worker_pipe.poll(self.pipe_timeout):
                policy, value = self.worker_pipe.recv()
                return policy, value
            else:
                logger.error(f"Worker {self.worker_id}: Parent process not responding for {self.pipe_timeout}s. Timeout.")
                raise TimeoutError(f"Pipe communication timeout after {self.pipe_timeout}s")
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Pipe error: {e}")
            raise