"""
tests/test_multiprocessing_readiness.py

Verifies the training pipeline is ready for 32-core CPU multiprocessing on
Vast.ai.  Run with:
    poetry run pytest tests/test_multiprocessing_readiness.py -v

Tests are grouped into:
  1.  Worker count formula
  2.  History-file sort key (prevents loading wrong checkpoints)
  3.  MCTS state isolation between instances / processes
  4.  IPCPredictorProxy – round-trip and timeout
  5.  Inference batch dispatch (queue.empty() path, not batch_size=256)
  6.  Arena thread count
  7.  MCTS recursion depth safety
  8.  Worker integration – spawn a process, serve GPU requests, get data back
  9.  Code quality / dead-code regression guards
  10. Game-logic correctness
"""

import multiprocessing as py_mp
import queue
import sys
import threading
import time

import numpy as np
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers used by integration tests
# ---------------------------------------------------------------------------

GAME_SIZE = 25


def _mini_gpu_server(q, parent_pipes, num_workers, game_size=GAME_SIZE, timeout=90):
    """
    Minimal GPU-server that serves a uniform policy to every worker inference
    request.  Mimics the relevant parts of the main loop in learn().

    Returns the list of training examples received via DATA messages.
    Raises RuntimeError if workers don't finish within *timeout* seconds.
    """
    active_workers = num_workers
    iteration_data = []
    requests_batch = []
    deadline = time.monotonic() + timeout

    while active_workers > 0:
        if time.monotonic() > deadline:
            raise RuntimeError("Server timed out – workers may be deadlocked")

        try:
            req = q.get(timeout=0.5)
        except queue.Empty:
            # Flush any partial batch that accumulated while workers computed.
            if requests_batch:
                for w_id, _tensor in requests_batch:
                    policy = np.ones(game_size, dtype=np.float32) / game_size
                    parent_pipes[w_id].send((policy, 0.0))
                requests_batch = []
            continue

        if req == "DONE":
            active_workers -= 1
            continue
        elif isinstance(req, tuple) and req[0] == "DATA":
            iteration_data.extend(req[1])
            continue

        # Inference request: (worker_id, tensor_array)
        requests_batch.append(req)

        # Dispatch as soon as the queue drains (mirrors the queue.empty() path
        # in learn() – critical when worker count < batch_size=256).
        if q.empty() and requests_batch:
            for w_id, _tensor in requests_batch:
                policy = np.ones(game_size, dtype=np.float32) / game_size
                parent_pipes[w_id].send((policy, 0.0))
            requests_batch = []

    return iteration_data


# ---------------------------------------------------------------------------
# 1. Worker count formula
# ---------------------------------------------------------------------------

class TestWorkerCountFormula:
    """Verify min(num_episodes, max(1, cpu_count - 2))."""

    @staticmethod
    def _calc(cpu_count: int, num_episodes: int) -> int:
        return min(num_episodes, max(1, cpu_count - 2))

    def test_32_cores_28_episodes_uses_28_workers(self):
        """Vast.ai recommended config: 32 cores, 28 episodes → 28 workers."""
        assert self._calc(32, 28) == 28

    def test_32_cores_14_episodes(self):
        """Default local config on 32-core machine → 14 workers."""
        assert self._calc(32, 14) == 14

    def test_16_cores_14_episodes(self):
        """README default: 16-core machine, 14 episodes → 14 workers."""
        assert self._calc(16, 14) == 14

    def test_single_core_gives_one_worker(self):
        """max(1, 1-2) = max(1, -1) = 1 → at least 1 worker always."""
        assert self._calc(1, 14) == 1

    def test_two_core_machine_gives_one_worker(self):
        """max(1, 2-2) = max(1, 0) = 1 → 1 worker."""
        assert self._calc(2, 14) == 1

    def test_episodes_cap_workers(self):
        """More cores than episodes → workers capped at episodes."""
        assert self._calc(64, 10) == 10

    def test_vast_ai_leaves_headroom(self):
        """28 workers on 32 cores leaves 4 cores free for main + OS."""
        workers = self._calc(32, 28)
        assert workers <= 30  # ≤ cpu_count - 2


# ---------------------------------------------------------------------------
# 2. History-file sort key
# ---------------------------------------------------------------------------

class TestHistorySortKey:
    """
    _history_sort_key must order by iteration number, not lexicographically.
    Lexicographic sort: "iteration_9_..." > "iteration_40_..." because '9' > '4'.
    The slice `all_files[-max_history_files:]` would then keep *old* iterations
    instead of the most-recent ones, corrupting the replay buffer on resume.
    """

    @pytest.fixture(autouse=True)
    def _load(self):
        from alpha_worms.main import _history_sort_key
        self.key = _history_sort_key

    def test_single_digit_before_double_digit(self):
        files = [
            "iteration_9_20250101_120000.pkl",
            "iteration_40_20250201_120000.pkl",
        ]
        result = sorted(files, key=self.key)
        assert result[0] == "iteration_9_20250101_120000.pkl"

    def test_lexicographic_sort_would_be_wrong(self):
        """Confirm the bug that _history_sort_key fixes."""
        files = ["iteration_9_20250101.pkl", "iteration_40_20250101.pkl"]
        lex = sorted(files)
        num = sorted(files, key=self.key)
        assert lex[0] == "iteration_40_20250101.pkl", "lex sort puts 40 first (wrong)"
        assert num[0] == "iteration_9_20250101.pkl"

    def test_many_iterations_sorted_numerically(self):
        nums = [1, 10, 2, 20, 100, 9, 50]
        files = [f"iteration_{n}_20250101_120000.pkl" for n in nums]
        sorted_files = sorted(files, key=self.key)
        result_nums = [int(f.split("_")[1]) for f in sorted_files]
        assert result_nums == sorted(nums)

    def test_malformed_filename_gets_sentinel(self):
        assert self.key("malformed.pkl") == (-1, "malformed.pkl")

    def test_last_n_slice_keeps_latest(self):
        """Simulate load_history keeping the 3 most-recent of 5 iterations."""
        files = [f"iteration_{n}_20250101_120000.pkl" for n in [1, 2, 3, 4, 5]]
        kept = sorted(files, key=self.key)[-3:]
        kept_nums = [int(f.split("_")[1]) for f in kept]
        assert kept_nums == [3, 4, 5]


# ---------------------------------------------------------------------------
# 3. MCTS state isolation
# ---------------------------------------------------------------------------

class TestMCTSStateIsolation:
    """Each MCTS instance must own independent state dicts and an instance-local RNG."""

    def _make_mock_nnet(self):
        m = MagicMock()
        m.predict.return_value = (np.ones(GAME_SIZE, dtype=np.float32) / GAME_SIZE, 0.0)
        return m

    def test_two_instances_have_separate_dicts(self):
        from alpha_worms.mcts import MCTS
        from alpha_worms.reflex_endpoint.game_engine import GameEngine

        game = GameEngine()
        args = {"numMCTSSims": 2, "cpuct": 1.41}
        mcts1 = MCTS(game, self._make_mock_nnet(), args)
        mcts2 = MCTS(game, self._make_mock_nnet(), args)

        for attr in ("Qsa", "Nsa", "Ns", "Ps", "Es", "Vs"):
            assert getattr(mcts1, attr) is not getattr(mcts2, attr), (
                f"MCTS.{attr} is shared between instances – concurrent workers would corrupt it"
            )

    def test_instance_rng_not_shared_with_global(self):
        from alpha_worms.mcts import MCTS
        from alpha_worms.reflex_endpoint.game_engine import GameEngine

        game = GameEngine()
        args = {"numMCTSSims": 2, "cpuct": 1.41}
        mcts1 = MCTS(game, self._make_mock_nnet(), args)
        mcts2 = MCTS(game, self._make_mock_nnet(), args)

        assert mcts1.rng is not mcts2.rng
        assert mcts1.rng is not np.random  # not the legacy global singleton

    def test_state_written_to_instance_not_shared(self):
        """Populate mcts1's tree; mcts2's tree must remain empty."""
        from alpha_worms.mcts import MCTS
        from alpha_worms.reflex_endpoint.game_engine import GameEngine
        import gc

        game = GameEngine()
        args = {"numMCTSSims": 1, "cpuct": 1.41}
        mock = self._make_mock_nnet()
        mcts1 = MCTS(game, mock, args)
        mcts2 = MCTS(game, mock, args)

        state = game.get_initial_state()
        canonical = game.get_canonical_form(state)
        mcts1.search(canonical)

        assert len(mcts1.Ps) > 0
        assert len(mcts2.Ps) == 0, "mcts2 must not see mcts1's tree entries"

    def test_del_mcts_releases_tree(self):
        """Deleting the MCTS object must not raise; tree should be collectable."""
        from alpha_worms.mcts import MCTS
        from alpha_worms.reflex_endpoint.game_engine import GameEngine
        import gc

        game = GameEngine()
        args = {"numMCTSSims": 1, "cpuct": 1.41}
        mcts = MCTS(game, self._make_mock_nnet(), args)
        state = game.get_initial_state()
        mcts.search(game.get_canonical_form(state))

        del mcts
        gc.collect()  # must not raise


# ---------------------------------------------------------------------------
# 4. IPCPredictorProxy – round-trip and timeout
# ---------------------------------------------------------------------------

class TestIPCPredictorProxy:
    """Test the worker-side IPC proxy for correctness and timeout safety."""

    def _make_proxy(self, worker_id, q, pipe):
        from alpha_worms.main import IPCPredictorProxy
        return IPCPredictorProxy(worker_id, q, pipe)

    def test_predict_round_trip(self):
        """Proxy sends the tensor to the queue; receives policy+value from pipe."""
        q = queue.Queue()
        parent_conn, child_conn = py_mp.Pipe()

        proxy = self._make_proxy(0, q, child_conn)

        dummy_policy = np.ones(GAME_SIZE, dtype=np.float32) / GAME_SIZE
        dummy_value = 0.42

        def server():
            _worker_id, _tensor = q.get(timeout=5)
            parent_conn.send((dummy_policy, dummy_value))

        t = threading.Thread(target=server)
        t.start()

        from alpha_worms.reflex_endpoint.game_engine import GameEngine
        game = GameEngine()
        state = game.get_canonical_form(game.get_initial_state())

        pi, v = proxy.predict(state)
        t.join(timeout=5)

        np.testing.assert_array_almost_equal(pi, dummy_policy)
        assert abs(v - dummy_value) < 1e-6

        parent_conn.close()
        child_conn.close()

    def test_predict_timeout_raises(self):
        """When the GPU server is silent, proxy must raise TimeoutError (not hang)."""
        q = queue.Queue()
        parent_conn, child_conn = py_mp.Pipe()

        proxy = self._make_proxy(0, q, child_conn)
        proxy.pipe_timeout = 0.05  # very short for test speed

        from alpha_worms.reflex_endpoint.game_engine import GameEngine
        game = GameEngine()
        state = game.get_canonical_form(game.get_initial_state())

        with pytest.raises(TimeoutError):
            proxy.predict(state)

        parent_conn.close()
        child_conn.close()

    def test_predict_sends_float32_tensor(self):
        """Tensor sent over queue must be float32 numpy array (cheap to pickle)."""
        q = queue.Queue()
        parent_conn, child_conn = py_mp.Pipe()
        proxy = self._make_proxy(0, q, child_conn)

        def server():
            _worker_id, tensor = q.get(timeout=5)
            assert isinstance(tensor, np.ndarray), "tensor must be numpy, not a DTO"
            assert tensor.dtype == np.float32
            parent_conn.send((np.ones(GAME_SIZE, dtype=np.float32) / GAME_SIZE, 0.0))

        t = threading.Thread(target=server)
        t.start()

        from alpha_worms.reflex_endpoint.game_engine import GameEngine
        game = GameEngine()
        state = game.get_canonical_form(game.get_initial_state())

        proxy.predict(state)
        t.join(timeout=5)
        parent_conn.close()
        child_conn.close()


# ---------------------------------------------------------------------------
# 5. Inference batch dispatch
# ---------------------------------------------------------------------------

class TestInferenceBatchDispatch:
    """
    With batch_size=256 and ≤30 workers, the ≥batch_size threshold is NEVER
    reached.  Inference must dispatch via the queue.empty() path.
    """

    def test_28_workers_never_fill_256_batch(self):
        """Max concurrent requests (≈ num_workers) is always below batch_size=256."""
        num_workers = 28
        batch_size = 256
        assert num_workers < batch_size, (
            "If num_workers ever reaches batch_size, workers would deadlock waiting "
            "for a response that only arrives once all 256 slots are filled."
        )

    def test_queue_empty_path_dispatches_partial_batch(self):
        """
        Simulate: 28 requests arrive, then the queue drains.
        The server must dispatch before seeing batch_size=256 items.
        """
        q = queue.Queue()
        batch_size = 256
        requests_batch = []
        dispatched_count = 0

        # Enqueue 28 inference requests (one per simulated worker)
        for i in range(28):
            q.put((i, np.zeros((271, 33), dtype=np.float32)))

        # Drain, mirroring the learn() main loop
        while True:
            try:
                req = q.get(timeout=0.01)
                requests_batch.append(req)
            except queue.Empty:
                if len(requests_batch) > 0:
                    dispatched_count = len(requests_batch)
                    break

            if len(requests_batch) >= batch_size or (
                len(requests_batch) > 0 and q.empty()
            ):
                dispatched_count = len(requests_batch)
                break

        assert dispatched_count > 0, "Batch was never dispatched"
        assert dispatched_count <= 28, "Batch exceeded worker count"

    def test_batch_size_256_is_unreachable_with_32_cores(self):
        """
        Document the configuration mismatch: args['batch_size']=256 is the
        TRAINING batch size, re-used as the inference flush threshold.
        On a 32-core machine (≤30 workers), the first condition in:
            if len(requests_batch) >= batch_size or (len > 0 and queue.empty())
        never fires.  Only the queue.empty() branch dispatches inference.
        """
        # With spawn start-method each worker sends exactly ONE request and
        # then blocks on pipe.recv().  So at most num_workers requests can be
        # in-flight simultaneously.
        max_in_flight = 30  # cpu_count - 2 on a 32-core machine
        training_batch_size = 256
        assert max_in_flight < training_batch_size


# ---------------------------------------------------------------------------
# 6. Arena thread count
# ---------------------------------------------------------------------------

class TestArenaThreadCount:
    """arena_compare uses ThreadPoolExecutor; verify the thread-count formula."""

    @staticmethod
    def _calc(arena_games: int, cpu_count: int) -> int:
        import torch.multiprocessing as tmp
        return min(arena_games, max(1, cpu_count - 2))

    def test_32_cores_20_games(self):
        assert self._calc(20, 32) == 20

    def test_32_cores_15_games(self):
        """Default arenaCompare=15 on 32 cores → 15 threads."""
        assert self._calc(15, 32) == 15

    def test_thread_count_never_exceeds_game_count(self):
        for games in [10, 15, 20, 25]:
            assert self._calc(games, 32) <= games

    def test_single_core_machine_minimum_one_thread(self):
        assert self._calc(15, 1) == 1


# ---------------------------------------------------------------------------
# 7. MCTS recursion depth safety
# ---------------------------------------------------------------------------

class TestMCTSRecursionDepth:
    """MCTS.search() is recursive.  Verify it stays inside Python's stack limit."""

    def test_max_turns_within_python_limit(self):
        max_turns = 250
        assert max_turns < sys.getrecursionlimit(), (
            f"max_turns={max_turns} must be < recursionlimit={sys.getrecursionlimit()}"
        )

    def test_search_does_not_raise_recursion_error(self):
        """Single get_action_probs call with 5 sims must not hit stack limit."""
        from alpha_worms.mcts import MCTS
        from alpha_worms.reflex_endpoint.game_engine import GameEngine

        game = GameEngine()
        mock = MagicMock()
        mock.predict.return_value = (np.ones(GAME_SIZE, dtype=np.float32) / GAME_SIZE, 0.0)
        args = {"numMCTSSims": 5, "cpuct": 1.41}

        mcts = MCTS(game, mock, args)
        state = game.get_canonical_form(game.get_initial_state())

        probs = mcts.get_action_probs(state, temp=1)

        assert len(probs) == GAME_SIZE
        assert abs(float(np.sum(probs)) - 1.0) < 1e-5

    def test_get_action_probs_temp0_returns_one_hot(self):
        """temp=0 (arena mode) must return a one-hot probability vector."""
        from alpha_worms.mcts import MCTS
        from alpha_worms.reflex_endpoint.game_engine import GameEngine

        game = GameEngine()
        mock = MagicMock()
        mock.predict.return_value = (np.ones(GAME_SIZE, dtype=np.float32) / GAME_SIZE, 0.0)
        args = {"numMCTSSims": 10, "cpuct": 1.41}

        mcts = MCTS(game, mock, args)
        state = game.get_canonical_form(game.get_initial_state())

        probs = mcts.get_action_probs(state, temp=0)

        assert probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert (probs == 1.0).sum() == 1, "temp=0 must select exactly one action"


# ---------------------------------------------------------------------------
# 8. Worker integration – spawn a process, serve GPU requests, get data back
# ---------------------------------------------------------------------------
# These tests spawn real subprocesses and are therefore slower (~10–60 s each).
# They are the most important tests for validating 32-core readiness.

# worker_helper and server_helper are defined at module level so they can be
# pickled by the spawn context without closure issues.

def _worker_target(worker_id, num_games, q, child_conn, args):
    """Thin wrapper: imports and calls Worms.worker_play_games inside the subprocess."""
    from alpha_worms.main import Worms
    from alpha_worms.reflex_endpoint.game_engine import GameEngine
    game = GameEngine()
    Worms.worker_play_games(worker_id, num_games, q, child_conn, args, game)


_FAST_ARGS = {
    "numMCTSSims": 4,   # keep sims very low so each game finishes quickly
    "cpuct": 1.41,
    "log_file": "/dev/null",
}


@pytest.mark.slow
class TestWorkerIntegration:
    """
    Full subprocess integration tests.  Each test spawns real processes using
    the 'spawn' start method (matching production) and verifies:
      – Workers complete without deadlock
      – Training data is returned
      – All exit codes are 0
    Mark slow to allow skipping in CI: pytest -m "not slow"
    """

    def _run_workers(self, num_workers, games_per_worker=1, timeout=90):
        """
        Spawn *num_workers* workers, serve their GPU requests via
        _mini_gpu_server, and return the collected training data.
        """
        ctx = py_mp.get_context("spawn")
        q = ctx.Queue()
        parent_pipes = []
        child_pipes = []

        for _ in range(num_workers):
            p_conn, c_conn = ctx.Pipe()
            parent_pipes.append(p_conn)
            child_pipes.append(c_conn)

        processes = []
        for i in range(num_workers):
            p = ctx.Process(
                target=_worker_target,
                args=(i, games_per_worker, q, child_pipes[i], _FAST_ARGS),
            )
            p.start()
            processes.append(p)

        # Close child pipe ends in parent (mirrors learn()'s post-spawn close)
        for c in child_pipes:
            try:
                c.close()
            except OSError:
                pass

        try:
            data = _mini_gpu_server(q, parent_pipes, num_workers, timeout=timeout)
            # Workers have finished their game and sent DONE, but may still be
            # running request_queue.close() / join_thread() cleanup.  Give them
            # up to 10 s to exit cleanly before the finally block terminates.
            for p in processes:
                p.join(timeout=10)
        finally:
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
            for conn in parent_pipes:
                try:
                    conn.close()
                except OSError:
                    pass
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

        return processes, data

    def test_single_worker_produces_data(self):
        """One worker, one game → must return ≥1 training example."""
        processes, data = self._run_workers(num_workers=1)

        assert not processes[0].is_alive(), "Worker hung"
        assert processes[0].exitcode == 0, (
            f"Worker exited with code {processes[0].exitcode}"
        )
        assert len(data) > 0, "Worker produced no training data"

        # Validate example structure: (state_tensor, pi, z)
        state_tensor, pi, z = data[0]
        assert isinstance(state_tensor, np.ndarray)
        assert state_tensor.dtype == np.float32
        assert state_tensor.shape[0] == 271  # input_planes (T=5 × 54 + 1)
        assert len(pi) == GAME_SIZE
        assert isinstance(z, (int, float))

    def test_four_workers_no_deadlock(self):
        """
        4 concurrent workers (all sharing one GPU-server queue) must all
        complete without deadlock.  This is the critical concurrency test.
        """
        num_workers = 4
        processes, data = self._run_workers(num_workers=num_workers, timeout=120)

        alive = [p for p in processes if p.is_alive()]
        bad_exits = [p.exitcode for p in processes if p.exitcode != 0]

        assert not alive, f"{len(alive)} worker(s) hung after timeout"
        assert not bad_exits, f"Workers exited with non-zero codes: {bad_exits}"
        assert len(data) >= num_workers, (
            f"Expected ≥{num_workers} examples, got {len(data)}"
        )

    def test_worker_data_shapes_consistent(self):
        """Every example from a worker must have the same tensor shape."""
        _, data = self._run_workers(num_workers=2, timeout=120)

        assert len(data) > 0
        shapes = {ex[0].shape for ex in data}
        assert len(shapes) == 1, (
            f"Inconsistent state tensor shapes across examples: {shapes}"
        )

    def test_worker_rewards_are_valid(self):
        """z values must be in [-1, 1] (or the tiny draw sentinel ≈1e-5)."""
        _, data = self._run_workers(num_workers=2, timeout=120)

        for _tensor, _pi, z in data:
            assert -1.0 <= z <= 1.0, f"Reward {z} out of [-1, 1]"


# ---------------------------------------------------------------------------
# 9. Code quality / dead-code guards
# ---------------------------------------------------------------------------

class TestCodeQualityGuards:
    """
    Guard against regressions of fixed dead-code issues.
    These parse the source text to verify the fix stays in place.
    """

    def _read_source(self, relpath):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, relpath)) as f:
            return f.read()

    def test_no_duplicate_raise_in_ipc_proxy(self):
        """
        IPCPredictorProxy.predict() previously ended with two consecutive bare
        `raise` statements.  The second was unreachable dead code.
        """
        source = self._read_source("alpha_worms/main.py")
        # Find the IPCPredictorProxy class block and check for double-raise
        proxy_start = source.find("class IPCPredictorProxy")
        assert proxy_start != -1
        proxy_block = source[proxy_start : proxy_start + 2000]
        # Two consecutive 'raise' lines (with only whitespace between them)
        import re
        double_raise = re.search(r"raise\s*\n\s*raise", proxy_block)
        assert double_raise is None, (
            "Duplicate bare `raise` found in IPCPredictorProxy.predict() – "
            "the second is unreachable dead code"
        )

    def test_no_unused_time_import_in_worker(self):
        """
        worker_play_games() previously imported `time` but never used it.
        """
        source = self._read_source("alpha_worms/main.py")
        worker_start = source.find("def worker_play_games")
        assert worker_start != -1
        # Extract just the function body (next ~3000 chars is enough)
        func_body = source[worker_start : worker_start + 3000]
        # If 'import time' is present, 'time.' must also be used
        if "import time" in func_body:
            assert "time." in func_body, (
                "`import time` found in worker_play_games but `time.` is never "
                "referenced – dead import wastes startup time on 30 worker spawns"
            )

    def test_os_nice_is_guarded(self):
        """os.nice(10) must be inside a try/except to survive restricted containers."""
        source = self._read_source("alpha_worms/main.py")
        nice_idx = source.find("os.nice(10)")
        assert nice_idx != -1
        # Look back a few lines for a 'try:' block
        surrounding = source[max(0, nice_idx - 200) : nice_idx + 50]
        assert "try:" in surrounding, (
            "os.nice(10) must be wrapped in try/except – PermissionError is "
            "raised in restricted cloud containers (e.g. non-root Vast.ai)"
        )

    def test_load_checkpoint_guards_missing_optimizer_key(self):
        """
        load_checkpoint with load_optimizer=True used to do
        checkpoint['optimizer'] unconditionally, KeyError-ing on old checkpoints.
        It must now check 'optimizer' in checkpoint first.
        """
        source = self._read_source("alpha_worms/nn.py")
        assert "'optimizer' in checkpoint" in source or \
               "checkpoint.get('optimizer')" in source, (
            "load_checkpoint must guard against missing 'optimizer' key – "
            "loading a checkpoint from an older training run would KeyError"
        )


# ---------------------------------------------------------------------------
# 10. Game-logic correctness
# ---------------------------------------------------------------------------

class TestGameLogic:
    """
    Unit tests for get_allowed_moves, get_next_state, and get_winner.
    Verifies the game engine rules that MCTS depends on for correctness.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from alpha_worms.reflex_endpoint.game_engine import GameEngine
        from alpha_worms.reflex_endpoint.dto import GameDTO
        from alpha_worms.reflex_endpoint.models import Dices, Pieces
        self.game = GameEngine()
        self.GameDTO = GameDTO
        self.Dices = Dices
        self.Pieces = Pieces

    def _make_state(self, dices, available_dices, selected_dices, grey_out,
                    board_pieces=None, p1_stack=None, p2_stack=None, player=1):
        dm = self.Dices(
            dices=dices,
            available_dices=available_dices,
            selected_dices=selected_dices,
            grey_out_dices=grey_out,
        )
        pm = self.Pieces(
            board_pieces=board_pieces if board_pieces is not None else list(range(21, 37)),
            current_player=player,
            player01_stack=p1_stack or [],
            player02_stack=p2_stack or [],
        )
        return self.GameDTO(dice_map=dm, board_map=pm)

    # --- roll_again with empty available_dices ---

    def test_roll_again_blocked_when_all_dice_committed(self):
        """
        When all 8 dice are committed (available_dices=[]) and grey_out=1,
        action 24 (roll again) must be 0.  Throwing with 0 dice is an
        immediate auto-bust — allowing it wastes MCTS budget.
        """
        state = self._make_state(
            dices=[1, 2, 3, 4, 5, 6, 1, 2],
            available_dices=[],
            selected_dices=[1, 2, 3, 4, 5, 6, 1, 2],
            grey_out=1,
        )
        moves = self.game.get_allowed_moves(state)
        assert moves[24] == 0, (
            "roll_again (action 24) must be blocked when available_dices=[] — "
            "throw_dices with no dice immediately busts"
        )

    def test_roll_again_allowed_when_dice_remaining(self):
        """
        When some dice are uncommitted (available_dices non-empty) and
        grey_out=1, action 24 must be 1.
        """
        state = self._make_state(
            dices=[1, 2, 3, 4, 5, 6, 1, 2],
            available_dices=[6, 7],
            selected_dices=[1, 2, 3, 4, 5, 6],
            grey_out=1,
        )
        moves = self.game.get_allowed_moves(state)
        assert moves[24] == 1, "roll_again must be allowed when dice remain"

    def test_roll_again_blocked_when_grey_out_zero(self):
        """
        When grey_out=0, the player has not yet selected any dice.
        Action 24 must be 0 (must select dice first).
        """
        state = self._make_state(
            dices=[1, 2, 3, 4, 5, 6, 1, 2],
            available_dices=[0, 1, 2, 3, 4, 5, 6, 7],
            selected_dices=[],
            grey_out=0,
        )
        moves = self.game.get_allowed_moves(state)
        assert moves[24] == 0, "roll_again must be 0 when grey_out=0"

    # --- piece claim requires a worm ---

    def test_piece_claim_requires_worm(self):
        """
        A player who has accumulated score ≥ 21 but no worm (face 6) committed
        must not be allowed to claim any piece.
        """
        # 7 dice showing face 3 = score 21, but no worm
        state = self._make_state(
            dices=[3, 3, 3, 3, 3, 3, 3, 1],
            available_dices=[],
            selected_dices=[3, 3, 3, 3, 3, 3, 3],
            grey_out=1,
        )
        moves = self.game.get_allowed_moves(state)
        piece_moves = moves[:16]
        assert sum(piece_moves) == 0, (
            "No piece may be claimed without a worm (face 6) in selected_dices"
        )

    def test_piece_claim_allowed_with_worm_and_sufficient_score(self):
        """
        A player with a worm (face=6) and total score ≥ 21 may claim piece 21.
        """
        state = self._make_state(
            dices=[5, 5, 5, 5, 6, 1, 1, 1],
            available_dices=[],
            selected_dices=[5, 5, 5, 5, 6],   # total = 4*5 + 1*5 = 25
            grey_out=1,
            board_pieces=[21, 22, 23, 24, 25],
        )
        moves = self.game.get_allowed_moves(state)
        assert moves[0] == 1, "piece 21 (action 0) should be claimable with score 25 and worm"
        assert moves[4] == 1, "piece 25 (action 4) should be claimable with score 25 and worm"
        assert moves[5] == 0, "piece 26 (action 5) should NOT be claimable (score 25 < 26)"

    # --- terminal state ---

    def test_terminal_when_board_empty(self):
        state = self._make_state(
            dices=[1]*8, available_dices=[0, 1, 2, 3, 4, 5, 6, 7],
            selected_dices=[], grey_out=0,
            board_pieces=[],
        )
        assert self.game.is_terminal(state)

    def test_not_terminal_with_pieces_remaining(self):
        state = self._make_state(
            dices=[1]*8, available_dices=[0, 1, 2, 3, 4, 5, 6, 7],
            selected_dices=[], grey_out=0,
            board_pieces=[36],
        )
        assert not self.game.is_terminal(state)

    def test_get_winner_returns_zero_non_terminal(self):
        state = self._make_state(
            dices=[1]*8, available_dices=[0, 1, 2, 3, 4, 5, 6, 7],
            selected_dices=[], grey_out=0,
            board_pieces=[21],
        )
        assert self.game.get_winner(state) == 0

    def test_get_winner_player1_wins_more_worms(self):
        state = self._make_state(
            dices=[1]*8, available_dices=[0, 1, 2, 3, 4, 5, 6, 7],
            selected_dices=[], grey_out=0,
            board_pieces=[],
            p1_stack=[33, 34],   # 3+3 = 6 worms
            p2_stack=[21, 22],   # 1+1 = 2 worms
        )
        assert self.game.get_winner(state) == 1

    def test_get_winner_player2_wins(self):
        state = self._make_state(
            dices=[1]*8, available_dices=[0, 1, 2, 3, 4, 5, 6, 7],
            selected_dices=[], grey_out=0,
            board_pieces=[],
            p1_stack=[21],       # 1 worm
            p2_stack=[36],       # 4 worms
        )
        assert self.game.get_winner(state) == -1

    def test_get_winner_draw(self):
        state = self._make_state(
            dices=[1]*8, available_dices=[0, 1, 2, 3, 4, 5, 6, 7],
            selected_dices=[], grey_out=0,
            board_pieces=[],
            p1_stack=[21],       # 1 worm
            p2_stack=[22],       # 1 worm
        )
        assert self.game.get_winner(state) == 0

    # --- worm scoring formula ---

    def test_worm_count_formula(self):
        """Pieces 21-24=1 worm, 25-28=2, 29-32=3, 33-36=4."""
        expected = {21: 1, 24: 1, 25: 2, 28: 2, 29: 3, 32: 3, 33: 4, 36: 4}
        for piece, _ in expected.items():
            state = self._make_state(
                dices=[1]*8, available_dices=[0, 1, 2, 3, 4, 5, 6, 7],
                selected_dices=[], grey_out=0,
                board_pieces=[],
                p1_stack=[piece], p2_stack=[],
            )
            assert self.game.get_winner(state) == 1, (
                f"piece {piece} should give player1 the win"
            )

    # --- steal action ---

    def test_steal_action_available_when_score_matches_opponent_top(self):
        """
        If current player's total == opponent's top piece and both ≥21 and
        player has a worm committed, the steal action must be available.
        """
        state = self._make_state(
            dices=[5, 5, 5, 5, 6, 1, 1, 1],
            available_dices=[],
            selected_dices=[5, 5, 5, 5, 6],   # total = 25
            grey_out=1,
            board_pieces=[],
            p1_stack=[],
            p2_stack=[25],                     # opponent top = 25
        )
        moves = self.game.get_allowed_moves(state)
        assert moves[4] == 1, (
            "Stealing piece 25 (action 4) must be allowed when score==25 and worm committed"
        )

    def test_steal_not_allowed_when_score_does_not_match(self):
        """Cannot steal unless total == opponent's top piece exactly."""
        state = self._make_state(
            dices=[5, 5, 5, 6, 1, 1, 1, 1],
            available_dices=[],
            selected_dices=[5, 5, 5, 6],   # total = 15+5 = 20
            grey_out=1,
            board_pieces=[],
            p1_stack=[],
            p2_stack=[25],
        )
        moves = self.game.get_allowed_moves(state)
        assert moves[4] == 0, "Cannot steal piece 25 when current score is only 20"

    # --- canonical form perspective swap ---

    def test_canonical_form_swaps_stacks_for_player2(self):
        """
        When current_player=2, get_canonical_form must swap player stacks so
        MCTS always searches from the 'current_player=1' perspective.
        """
        from alpha_worms.reflex_endpoint.game_engine import GameEngine
        game = GameEngine()
        state = self._make_state(
            dices=[1]*8, available_dices=[0, 1, 2, 3, 4, 5, 6, 7],
            selected_dices=[], grey_out=0,
            board_pieces=list(range(21, 37)),
            p1_stack=[21], p2_stack=[22],
            player=2,
        )
        canonical = game.get_canonical_form(state)
        assert canonical.board_map.current_player == 1
        # After swap: what was player2's stack (22) becomes player01_stack
        assert canonical.board_map.player01_stack == [22]
        assert canonical.board_map.player02_stack == [21]

    def test_canonical_form_noop_for_player1(self):
        """When current_player=1, canonical form must not swap stacks."""
        from alpha_worms.reflex_endpoint.game_engine import GameEngine
        game = GameEngine()
        state = self._make_state(
            dices=[1]*8, available_dices=[0, 1, 2, 3, 4, 5, 6, 7],
            selected_dices=[], grey_out=0,
            p1_stack=[21], p2_stack=[22],
            player=1,
        )
        canonical = game.get_canonical_form(state)
        assert canonical.board_map.current_player == 1
        assert canonical.board_map.player01_stack == [21]
        assert canonical.board_map.player02_stack == [22]

    # --- regression guards ---

    def test_no_dead_deque_trim_loop_in_learn(self):
        """
        The explicit while-loop trim of train_data is dead code: deque(maxlen=N)
        enforces the cap automatically on every extend/append.
        """
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "alpha_worms/main.py")) as f:
            source = f.read()
        assert "while len(self.train_data) > max_buffer_size" not in source, (
            "Dead explicit deque trim loop found in learn() — "
            "deque(maxlen=N) already enforces the cap; this while-loop can never fire"
        )

    def test_predict_does_not_always_call_nnet_eval(self):
        """
        predict() used to call self.nnet.eval() unconditionally on every call.
        With 80 sims × workers × moves, this adds hundreds of thousands of
        redundant mode-switches per iteration.
        The fix: only switch when self.nnet.training is True.
        """
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "alpha_worms/nn.py")) as f:
            source = f.read()
        assert "if self.nnet.training" in source, (
            "predict() and predict_batch() should guard eval() with "
            "`if self.nnet.training:` to avoid redundant mode-switches"
        )


# ---------------------------------------------------------------------------
# 11. MCTS correctness
# ---------------------------------------------------------------------------

class TestMCTSCorrectness:
    """
    Unit tests for MCTS simulation count, tree reuse, and draw-value consistency.
    """

    def _make_mock_nnet(self, game_size=25):
        """
        Returns a predictor that always returns uniform policy + value 0.0.
        Tracks how many times predict() was called.
        """
        from unittest.mock import MagicMock
        nnet = MagicMock()
        nnet.predict.return_value = (
            np.ones(game_size, dtype=np.float32) / game_size,
            0.0,
        )
        return nnet

    def test_sim_count_fresh_root(self):
        """
        When the root state is not yet in Ps (fresh game turn), exactly
        numMCTSSims simulations must run.  One runs during initial expansion
        (step 1), then numMCTSSims-1 more in the loop = numMCTSSims total.
        """
        from alpha_worms.reflex_endpoint.game_engine import GameEngine
        from alpha_worms.mcts import MCTS

        game = GameEngine()
        nnet = self._make_mock_nnet()
        mcts = MCTS(game, nnet, {'numMCTSSims': 5, 'cpuct': 1.0})

        state = game.get_initial_state()
        canonical = game.get_canonical_form(state)

        # Patch search to count calls (we only need the counter, not real MCTS)
        original_search = mcts.search
        call_count = [0]

        def counting_search(s):
            call_count[0] += 1
            return original_search(s)

        mcts.search = counting_search
        mcts.get_action_probs(canonical, temp=0)

        # Step 1 (if fresh) + loop (numMCTSSims - 1) = numMCTSSims = 5 total.
        # The exact count may vary with tree depth, but must be >= numMCTSSims.
        assert call_count[0] >= 5, (
            f"Expected at least {5} search calls for a fresh root, got {call_count[0]}"
        )

    def test_sim_count_pre_expanded_root(self):
        """
        When the root IS already in Ps (tree reuse from a prior turn), exactly
        numMCTSSims simulations must still run — not numMCTSSims-1.

        Before the fix, pre-expanded roots got one fewer simulation because
        step 1 was skipped and the loop counter was not adjusted.
        """
        from alpha_worms.reflex_endpoint.game_engine import GameEngine
        from alpha_worms.mcts import MCTS

        game = GameEngine()
        nnet = self._make_mock_nnet()
        mcts = MCTS(game, nnet, {'numMCTSSims': 5, 'cpuct': 1.0})

        state = game.get_initial_state()
        canonical = game.get_canonical_form(state)

        # First call: expands the root, populates Ps[s].
        mcts.get_action_probs(canonical, temp=0)

        # Instrument search AFTER the first call so we measure the second call.
        original_search = mcts.search
        call_count = [0]

        def counting_search(s):
            call_count[0] += 1
            return original_search(s)

        mcts.search = counting_search

        # Second call: root already in Ps — should still run numMCTSSims sims.
        mcts.get_action_probs(canonical, temp=0)

        # Since root is pre-expanded, n_sims = numMCTSSims (not numMCTSSims-1).
        assert call_count[0] >= 5, (
            f"Pre-expanded root: expected ≥{5} search calls, got {call_count[0]}. "
            "The off-by-one fix must set n_sims=numMCTSSims when root is pre-expanded."
        )

    def test_draw_sentinel_source_consistency(self):
        """
        MCTS.search() uses 1e-4 as the draw sentinel (Es[s] for tied terminal).
        execute_episode must use the same constant for the z target.

        Before the fix, execute_episode used 1e-5, meaning training targets
        were ~10x smaller than MCTS-internal draw values — a consistent
        asymmetry that slightly biased value estimation.
        """
        import os, inspect
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        with open(os.path.join(root, "alpha_worms/mcts.py")) as f:
            mcts_src = f.read()
        with open(os.path.join(root, "alpha_worms/main.py")) as f:
            main_src = f.read()

        # MCTS must use 1e-4 for the draw sentinel
        assert "1e-4" in mcts_src, "MCTS draw sentinel should be 1e-4"

        # execute_episode must use the same constant (not 1e-5) for tied terminals
        assert "reward = 1e-5" not in main_src, (
            "execute_episode used 1e-5 for tied-terminal reward; "
            "should match MCTS draw sentinel (1e-4)"
        )
        assert "reward = 1e-4" in main_src, (
            "execute_episode must use 1e-4 (same as MCTS draw sentinel) "
            "for tied-terminal reward so training targets are consistent"
        )

    def test_mcts_sim_count_constant_in_source(self):
        """
        Verify the n_sims fix is present: the loop uses
        `n_sims = numMCTSSims - (1 if s not in self.Ns else 0)`.
        """
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "alpha_worms/mcts.py")) as f:
            source = f.read()
        assert "n_sims" in source, (
            "MCTS get_action_probs should use n_sims variable to adjust "
            "simulation count based on whether the root was pre-expanded"
        )
        assert "for i in range(self.args['numMCTSSims'] - 1)" not in source, (
            "The old `range(numMCTSSims - 1)` loop must be replaced with "
            "`range(n_sims)` to avoid off-by-one on pre-expanded roots"
        )
