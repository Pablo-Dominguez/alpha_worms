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
import multiprocessing as py_mp

# Spawn method is set inside learn() to ensure it applies regardless of entry script.

# Configure logging for worker heartbeat
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _log_mp_state(tag, processes=None, request_queue=None):
    """Lightweight diagnostics for IPC leak/deadlock triage."""
    try:
        children = py_mp.active_children()
        child_pids = [p.pid for p in children if p.pid is not None]
    except Exception as e:
        child_pids = [f"error:{e}"]

    proc_state = []
    if processes:
        for p in processes:
            try:
                proc_state.append(f"pid={p.pid},alive={p.is_alive()},exit={p.exitcode}")
            except Exception as e:
                proc_state.append(f"pid=?,err={e}")

    qsize_info = "na"
    if request_queue is not None:
        try:
            qsize_info = request_queue.qsize()
        except (NotImplementedError, AttributeError):
            qsize_info = "unsupported"
        except Exception as e:
            qsize_info = f"error:{e}"

    logger.info(
        f"[MP_STATE] {tag} | active_children={len(child_pids)} {child_pids} | "
        f"tracked_processes={proc_state} | queue_qsize={qsize_info}"
    )

def _history_sort_key(filename):
    """
    Numeric sort key for history pkl files so load_history and save_iteration_data
    always process them in true chronological order.

    Format: iteration_{N}_{YYYYMMDD}_{HHMMSS}.pkl
    Lexicographic (default) sort is wrong: "iteration_9_..." > "iteration_40_..."
    because '9' > '4', so the 'last N' slice would load old small-iteration data
    instead of the most-recent iterations.
    """
    parts = filename.replace('.pkl', '').split('_')
    try:
        iter_n = int(parts[1])
        # Timestamp suffix (YYYYMMDD_HHMMSS) sorts correctly as a string within
        # the same iteration number (handles duplicate pkl files from restarts).
        timestamp = '_'.join(parts[2:]) if len(parts) > 2 else ''
        return (iter_n, timestamp)
    except (IndexError, ValueError):
        return (-1, filename)

class Worms:
    def __init__(self, game: GameEngine, nnet: WormsPredictor, args):
        self.game = game
        self.nnet = nnet
        self.args = args  
        self.train_data = deque(maxlen=args.get('max_buffer_size', 50000))

    @staticmethod
    def execute_episode(game, mcts=None, tensor_engine=None, game_recorder=None):
        train_examples = []
        state = game.get_initial_state()
        curr_player = 1 
        turn_count = 0        # 1. Add a turn counter
        max_turns = 250       # 2. Set a reasonable upper limit for a game of Worms

        while True:
            turn_count += 1

            canonical_state = game.get_canonical_form(state)
            # Use temp=1 for the first 30 moves then temp=0 (greedy) to sharpen
            # the policy in the endgame, matching standard AlphaZero practice.
            temp = 1 if turn_count <= 30 else 0
            pi = mcts.get_action_probs(canonical_state, temp=temp)
            
            # Store the DTO temporarily during the game
            train_examples.append([canonical_state, pi, curr_player])

            action = np.random.choice(len(pi), p=pi)
            next_s = game.get_next_state(state, action)

            # Record this step to the database (no-op if game_recorder is None)
            if game_recorder is not None:
                game_recorder.step(state, action, next_s)

            # Link the history for the NEXT step's canonicalization
            next_s.previous_dto = state

            # FIX (memory): truncate the live state chain to T frames. Without this,
            # `state` retains every prior turn's DTO (and each one keeps a depth-T
            # canonical history), causing O(turns) RAM growth per game per worker.
            _node = next_s
            for _ in range(5):
                if _node.previous_dto is None:
                    break
                _node = _node.previous_dto
            _node.previous_dto = None

            turn_switched = next_s.board_map.current_player != state.board_map.current_player
            state = next_s

            if turn_switched:
                curr_player *= -1

            reward = game.get_winner(state)

            # FIX (correctness): a tied terminal state (board empty, equal worm sums)
            # returns reward=0 from get_winner, so the loop would call MCTS on a
            # terminal state with no valid moves. MCTS.search() correctly returns the
            # terminal value without populating Ps[s]/Vs[s], so the subsequent
            # `self.Ps[s] *= valids` in get_action_probs raises KeyError and the
            # worker dies. That was the cause of all the "Worker N crashed with
            # error: ((...),(),0,(),1,(...),(...))" messages in the log.
            if reward == 0 and game.is_terminal(state):
                # FIX (consistency): MCTS.search() uses 1e-4 as the sentinel
                # for a tied terminal.  Using a different constant here (1e-5)
                # meant training targets were ~10x smaller than MCTS values for
                # draws, creating a slight systematic bias.
                reward = 1e-4  # unified draw sentinel — matches MCTS Es[s]
                print(f"Game ended in tied terminal at turn {turn_count}.")

            # 4. Check for a forced draw if the game drags on too long
            if turn_count >= max_turns and reward == 0:
                # Use the point calculation to see who was winning
                reward = game.get_score_winner(state)
                
                # If it's a dead tie on points, use the unified draw sentinel
                if reward == 0:
                    reward = 1e-4  # matches MCTS Es[s] draw sentinel
                print(f"Game hit turn limit ({max_turns})! Winner by points: {reward}")

            if reward != 0:
                if game_recorder is not None:
                    game_recorder.finish(reward)

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
            try:
                if hasattr(torch, "compile"):
                    self.nnet.model = torch.compile(self.nnet.nnet)
                    logger.info(f"Successfully compiled nnet model.")
            except Exception as e:
                logger.warning(f"Could not compile nnet model: {e}")
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
        Loads previous self-play games from the history folder into the deque.
        FIX (memory/disk): only load the most recent N iterations so RAM and
        startup time don't grow unboundedly across hundreds of iterations.
        """
        history_folder = os.path.join(self.args['checkpoint'], "history")
        if not os.path.exists(history_folder):
            print("No history folder found. Starting with an empty buffer.")
            return

        print("Loading history from disk... This might take a moment.")
        # Sort files numerically by iteration number so the 'last N' slice
        # keeps the most-recent iterations, not the lexicographically-largest ones.
        max_history_files = self.args.get('max_history_files', 10)
        all_files = sorted(
            [f for f in os.listdir(history_folder) if f.endswith(".pkl")],
            key=_history_sort_key
        )
        files = all_files[-max_history_files:] if max_history_files > 0 else all_files
        skipped = len(all_files) - len(files)
        if skipped > 0:
            print(f"Skipping {skipped} old history files (keeping last {len(files)}).")
        
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
        try:
            with open('/proc/self/status') as _f:
                for _line in _f:
                    if _line.startswith('VmRSS:'):
                        logger.info(f"RAM (RSS) before arena: {_line.strip()}")
                        break
        except OSError:
            pass

        arena_args = self.args.copy()
        arena_args['numMCTSSims'] = int(self.args.get('numMCTSSims', 80) * sim_multiplier)

        # Dual-model batching setup for arena
        arena_queue = queue.Queue()
        # NOT daemon: we join explicitly below.  A daemon thread killed while
        # mid-CUDA-forward-pass can corrupt the CUDA allocator state, which
        # then causes a segfault when torch.cuda.empty_cache() runs on the
        # main thread after arena_compare() returns.
        batching_thread = threading.Thread(target=self.batch_arena_requests, args=(arena_queue, challenger, baseline))
        batching_thread.start()

        # Multi-threaded arena evaluation using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        num_workers = min(arena_games, max(1, mp.cpu_count() - 2))  # Scale with available CPUs
        
        results = []
        try:
            # Each arena thread must have its own GameEngine to avoid sharing the
            # global NumPy RandomState (legacy np.random is NOT thread-safe and
            # causes heap corruption → segfault when called concurrently).
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
            # Allow up to 30 s for the thread to finish its last predict_batch() call.
            # We must ensure it has fully exited before the caller does
            # `del baseline_predictor` and `torch.cuda.empty_cache()`, otherwise a
            # concurrent CUDA allocation in the batching thread races with the
            # allocator clean-up and can segfault.
            batching_thread.join(timeout=30.0)
            if batching_thread.is_alive():
                logger.warning("Arena batching thread did not finish within 30 s — possible hang.")

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
        # Each thread gets its own GameEngine so that calls to np.random inside
        # get_initial_state / throw_dices / end_turn are never concurrent on the
        # same RNG state (np.random legacy API is NOT thread-safe).
        game = GameEngine()

        if challenger_goes_second:
            p1_nnet, p2_nnet = baseline, challenger
        else:
            p1_nnet, p2_nnet = challenger, baseline

        # Use proxies for batched inference
        p1_proxy = ArenaPredictorProxy(p1_nnet, arena_queue)
        p2_proxy = ArenaPredictorProxy(p2_nnet, arena_queue)

        mcts_p1 = MCTS(game, p1_proxy, arena_args)
        mcts_p2 = MCTS(game, p2_proxy, arena_args)
        
        state = game.get_initial_state()
        turn_count = 0
        max_turns = 250

        while True:
            turn_count += 1
            current_player = state.board_map.current_player
            canonical_state = game.get_canonical_form(state)

            active_mcts = mcts_p1 if current_player == 1 else mcts_p2
            pi = active_mcts.get_action_probs(canonical_state, temp=0)
            action = int(np.argmax(pi))
            prev_state = state
            state = game.get_next_state(state, action)
            state.previous_dto = prev_state
            # Truncate the DTO history chain to T=5 frames. Without this,
            # the chain grows 200 nodes deep over a max-turns arena game,
            # pinning ~30-50 MB per arena thread (4 concurrent = ~200 MB).
            _node = state
            for _ in range(5):
                if _node.previous_dto is None:
                    break
                _node = _node.previous_dto
            _node.previous_dto = None

            winner = game.get_winner(state)
            if winner != 0:
                if challenger_goes_second:
                    return -winner  # Invert because challenger is P2
                return winner

            # FIX (correctness): same tied-terminal trap as in execute_episode.
            # Without this, the loop continues and the next MCTS call crashes
            # the arena thread on KeyError(Ps[s]), which the outer try/except
            # in arena_compare silently counts as a draw -- but only after
            # blocking ~120 s on TimeoutError. Massive arena slowdown source.
            if game.is_terminal(state):
                return 0  # tied terminal -> draw

            if turn_count >= max_turns:
                score_winner = game.get_score_winner(state)
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

        try:
            os.nice(10)
        except (PermissionError, OSError):
            pass  # Restricted container or non-Linux — low priority is nice-to-have
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
            try:
                if hasattr(torch, "compile"):
                    self.nnet.model = torch.compile(self.nnet.nnet)
                    logger.info(f"Successfully compiled nnet model.")
            except Exception as e:
                logger.warning(f"Could not compile nnet model after loading best checkpoint: {e}")
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
                # FIX (correctness): close the parent's copy of the child pipe ends
                # right after spawning. Otherwise a crashed worker never causes EOF
                # on the parent side, and child_pipe fds leak across the iteration.
                for c_pipe in child_pipes:
                    try:
                        c_pipe.close()
                    except OSError:
                        pass

                _log_mp_state("after_worker_start", processes=processes, request_queue=request_queue)

                active_workers = num_workers
                requests_batch = []
                worker_timeout = 5       # Short: flush partial batches quickly to avoid pipe_timeout race
                deadlock_threshold = 70  # True deadlock only after prolonged inactivity
                last_activity_time = datetime.now()

                logger.info(f"Started {num_workers} CPU workers. GPU Batching server listening...")

                while active_workers > 0:
                    try:
                        req = request_queue.get(timeout=worker_timeout)
                        last_activity_time = datetime.now()

                        if req == "DONE":
                            active_workers -= 1
                            logger.info(f"Worker signaled completion. {active_workers} workers remaining.")
                            _log_mp_state("after_done_signal", processes=processes, request_queue=request_queue)
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
                                try:
                                    parent_pipes[w_id].send((policies[idx], values[idx]))
                                except BrokenPipeError:
                                    logger.warning(f"Worker {w_id} pipe broken (already exited), dropping response.")
                            requests_batch = []
                            last_activity_time = datetime.now()
                            continue

                        time_elapsed = (datetime.now() - last_activity_time).total_seconds()
                        logger.warning(f"No worker activity for {time_elapsed}s. Active workers: {active_workers}")

                        if time_elapsed > deadlock_threshold:
                            logger.critical(f"Silent deadlock detected! Terminating {len(processes)} worker processes.")
                            for p in processes:
                                if p.is_alive():
                                    p.terminate()
                                    p.join(timeout=5)
                                    if p.is_alive():
                                        p.kill()
                                        p.join(timeout=3)
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
                            try:
                                parent_pipes[w_id].send((policies[idx], values[idx]))
                            except BrokenPipeError:
                                logger.warning(f"Worker {w_id} pipe broken (already exited), dropping response.")
                        requests_batch = []

            finally:
                _log_mp_state("cleanup_start", processes=processes, request_queue=request_queue)
                # Guaranteed cleanup on normal exit, KeyboardInterrupt, or any exception.
                # terminate() before join() ensures workers that are blocked on pipe.recv()
                # don't hang the main process indefinitely.
                for p in processes:
                    if p.is_alive():
                        logger.info(f"Terminating worker pid={p.pid}")
                        p.terminate()
                        p.join(timeout=5)
                    if p.is_alive():
                        logger.warning(f"Process {p.pid} did not terminate cleanly. Killing...")
                        p.kill()
                        p.join(timeout=3)
                _log_mp_state("after_process_terminate", processes=processes, request_queue=request_queue)
                # Drain any items left in the queue before closing.  Workers that were
                # terminated mid-flight may have partially written DATA/DONE/inference
                # requests.  Leaving them buffered causes the Queue's internal feeder
                # thread to block on join, and calling cancel_join_thread() instead
                # leaks the Queue's 3 POSIX semaphores (logged as "3 leaked semaphore
                # objects" by resource_tracker at shutdown).
                drained_items = 0
                try:
                    while True:
                        request_queue.get_nowait()
                        drained_items += 1
                except Exception:
                    pass
                logger.info(f"Queue drain complete. drained_items={drained_items}")
                _log_mp_state("before_queue_close", processes=processes, request_queue=request_queue)
                request_queue.close()
                logger.info("request_queue.close() complete")
                request_queue.join_thread()
                logger.info("request_queue.join_thread() complete")
                for p_pipe, c_pipe in zip(parent_pipes, child_pipes):
                    try:
                        p_pipe.close()
                    except OSError:
                        pass
                    try:
                        c_pipe.close()  # may already be closed (we close() right after spawn)
                    except OSError:
                        pass
                logger.info("All pipe endpoints closed")
                _log_mp_state("cleanup_end", processes=processes, request_queue=request_queue)

            if len(iteration_train_data) > 0:
                self.save_iteration_data(i, iteration_train_data)
                self.train_data.extend(iteration_train_data)
                # deque already enforces maxlen, so no explicit slice is needed here.

            if len(self.train_data) > 0:
                print(f"Training on {len(self.train_data)} moves...")
                self.train_network(epochs=self.args.get('train_epochs', 10))
            
            # deque(maxlen=...) enforces the cap automatically on every extend/append;
            # an explicit trim loop here can never fire and is dead code.
            logger.info(f"Current train_data buffer size: {len(self.train_data)}")
            
            # Explicitly call garbage collection and CUDA cache clearing
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

            checkpoint_name = f'checkpoint_{i}.pth.tar'
            self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=checkpoint_name)

            # Prune old per-iteration checkpoint files to cap disk usage over long
            # runs (500 iters x ~5 MB each = ~2.5 GB without pruning).
            # checkpoint_best.pth.tar is always preserved regardless.
            _keep = self.args.get('keep_last_checkpoints', 20)
            if _keep > 0:
                _cf = self.args['checkpoint']
                _all_ckpts = sorted(
                    [f for f in os.listdir(_cf)
                     if f.startswith('checkpoint_') and f.endswith('.pth.tar')
                     and f != 'checkpoint_best.pth.tar'],
                    key=lambda f: int(f.split('_')[1].split('.')[0])
                    if f.split('_')[1].split('.')[0].isdigit() else -1
                )
                for _old in _all_ckpts[:-_keep]:
                    try:
                        os.remove(os.path.join(_cf, _old))
                    except OSError:
                        pass
                if len(_all_ckpts) > _keep:
                    logger.info(f"Pruned {len(_all_ckpts) - _keep} old checkpoint file(s).")

            if self.args.get('arenaCompare', 0) > 0:
                best_path = self.get_best_checkpoint_path()
                if os.path.exists(best_path):
                    # Load baseline just-in-time so its weights are NOT resident in RAM
                    # during the ~30-min self-play + training phase (saves one full model copy).
                    baseline_predictor = WormsPredictor(device='cpu')
                    baseline_predictor.load_checkpoint(folder=self.args['checkpoint'], filename='checkpoint_best.pth.tar', load_optimizer=False)
                    try:
                        if hasattr(torch, "compile"):
                            baseline_predictor.model = torch.compile(baseline_predictor.nnet)
                            logger.info(f"Successfully compiled baseline predictor model.")
                    except Exception as e:
                        logger.warning(f"Could not compile baseline predictor model: {e}")
                    challenger_passed = self.arena_compare(self.nnet, baseline_predictor)
                    if challenger_passed:
                        self.save_best_checkpoint(checkpoint_name, i)
                    else:
                        print("Reloading baseline model after arena rejection.")
                        self.nnet.load_checkpoint(folder=self.args['checkpoint'], filename='checkpoint_best.pth.tar')
                    del baseline_predictor
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    # No baseline yet — first accepted model becomes the baseline.
                    self.save_best_checkpoint(checkpoint_name, i)
    
    def save_iteration_data(self, iteration, data):
        folder = os.path.join(self.args['checkpoint'], "history")
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        filename = os.path.join(folder, f"iteration_{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Successfully saved {len(data)} moves to {filename}")
        # Prune old history files so disk usage stays bounded over a long run.
        # Uses the same max_history_files cap as load_history.
        max_history_files = self.args.get('max_history_files', 10)
        if max_history_files > 0:
            all_pkls = sorted(
                [f for f in os.listdir(folder) if f.endswith(".pkl")],
                key=_history_sort_key
            )
            to_delete = all_pkls[:-max_history_files]
            for old_file in to_delete:
                try:
                    os.remove(os.path.join(folder, old_file))
                except OSError:
                    pass
            if to_delete:
                logger.info(f"Pruned {len(to_delete)} old history pkl(s) from {folder}")

    def train_network(self, epochs=1):
        """
        Samples a batch from the buffer and trains the NN.
        """
        # FIX (CPU): snapshot the deque once. random.sample on a deque is O(n)
        # because deques don't support random access, so it internally converts
        # to a list every epoch — with 20 epochs over a 50k buffer that's 1M
        # element copies per iteration.
        snapshot = list(self.train_data)
        for epoch in range(epochs):
            # We sample a random batch to break the correlation between consecutive moves
            batch_size = min(len(snapshot), self.args['batch_size'])
            batch = random.sample(snapshot, batch_size)
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
        import torch as _torch
        _torch.set_num_threads(1)  # Prevent OpenMP explosion inside each worker process
        # Configure logging — logging.basicConfig() from the parent is NOT inherited by spawned
        # subprocesses. Without this, all logger.info/error calls below are completely silent.
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Spawned workers start with a fresh interpreter: add a FileHandler so
        # their logger.* output goes to the same log file as the main process.
        # basicConfig only adds a StreamHandler (stderr); without this, worker
        # logs are invisible in the file when stderr is not redirected.
        _log_file = args.get('log_file', 'gpu_test_output.log')
        _root = logging.getLogger()
        if not any(isinstance(_h, logging.FileHandler) for _h in _root.handlers):
            _wfh = logging.FileHandler(_log_file, mode='a', encoding='utf-8')
            _wfh.setLevel(logging.DEBUG)
            _wfh.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            _root.addHandler(_wfh)
        # 1. Create the fake predictor
        proxy_nnet = IPCPredictorProxy(worker_id, request_queue, worker_pipe)

        # Each worker gets its own CPU-based tensor engine for DTO conversion
        from alpha_worms.reflex_endpoint.tensor_engine import TensorEngine
        local_tensor_engine = TensorEngine()

        try:
            import random as _random
            from alpha_worms.game_recorder import GameRecorder
            for game_num in range(1, num_games + 1):
                # Record ~5% of games chosen independently per game.
                # At 14 games/iter this gives ~51% chance of at least one recording per iter.
                recorder = GameRecorder.create() if _random.random() < 0.05 else None

                # 2. Initialize a fresh MCTS for this game using the PROXY
                mcts = MCTS(game, proxy_nnet, args)

                # NOTE: You will need to ensure your self.execute_episode() 
                # uses this specific 'mcts' instance! 
                # Example: game_history = self.execute_episode(mcts)
                game_history = Worms.execute_episode(game, mcts=mcts, tensor_engine=local_tensor_engine,
                                                     game_recorder=recorder)

                # FIX (memory): ship each game's tensors immediately instead of
                # accumulating all games in `worker_train_data` and pickling them
                # in one ~10 MB+ IPC blob at worker exit. This keeps worker RSS
                # roughly flat across games and avoids OOM spikes on small-RAM hosts.
                request_queue.put(("DATA", game_history))
                examples_sent = len(game_history)
                del game_history
                # Release recorder before gc.collect() so the SQLAlchemy engine
                # is not kept alive by this reference during collection.
                del recorder
                # Drop the MCTS tree (Ps/Qsa/Nsa/Ns/Es/Vs) before the next game.
                del mcts
                gc.collect()

                # FIX 5: Heartbeat logging every 5 games
                if game_num % 5 == 0:
                    logger.info(f"Worker {worker_id} progress: {game_num}/{num_games} games, last_game_examples={examples_sent}")
                else:
                    print(f"Worker {worker_id} completed game {game_num}/{num_games}")

            logger.info(f"Worker {worker_id} finished: sent all {num_games} games via streaming DATA")

        except Exception as e:
            logger.error(f"Worker {worker_id} crashed with error: {e}")
            # Finalize any in-progress DB recording so the match row is not
            # left as 'ongoing'.  finish() is idempotent — safe if already called.
            _rec = locals().get('recorder')
            if _rec is not None:
                _rec.finish(0)

        # --- FREE WORKER MEMORY BEFORE EXIT ---
        gc.collect()

        # 4. Tell the Chef this worker is clocking out
        logger.info(f"Worker {worker_id} sending DONE")
        request_queue.put("DONE")
        logger.info(f"Worker {worker_id} DONE sent, worker exiting")
        # Release the queue's 3 POSIX semaphores from this process before exit.
        # Without close()+join_thread(), the feeder thread keeps them alive and
        # the resource_tracker reports "3 leaked semaphore objects" at shutdown.
        request_queue.close()
        request_queue.join_thread()
        
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
        self.pipe_timeout = 120  # FIX 1: 120-second timeout for pipe communication

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
            raise  # Never silently return None — MCTS receives (None, None) and
                   # crashes with a confusing TypeError instead of this clear message.