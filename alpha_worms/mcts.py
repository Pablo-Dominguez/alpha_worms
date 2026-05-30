import math
import numpy as np
from alpha_worms.reflex_endpoint.dto import GameDTO
from alpha_worms.reflex_endpoint.game_engine import GameEngine
from alpha_worms.nn import WormsPredictor

class MCTS:
    def __init__(self, game: GameEngine, nnet: WormsPredictor, args):
        self.game = game
        self.nnet = nnet
        self.args = args 
        
        self.Qsa = {}  # Stores average value of action 'a' from state 's'
        self.Nsa = {}  # Visit count for (s, a)
        self.Ns = {}   # Total visit count for state s
        self.Ps = {}   # Policy (NN predictions) for state s
        self.Es = {}   # Terminal status of state s
        self.Vs = {}   # Valid moves mask for state s
        # Instance-local RNG so concurrent arena threads never share the global
        # np.random singleton (it is NOT thread-safe and causes heap corruption).
        self.rng = np.random.default_rng()

    def get_string_hash(self, state: GameDTO):
        """
        FIX (memory/CPU): use a flat tuple of immutables as the dict key instead
        of `sha256(json.dumps(model_dump()))`. The old implementation allocated
        ~5 throwaway dicts + a JSON string + a 64-byte hex digest per call, and
        was invoked ~170k times per training iteration — the dominant source of
        Python allocator churn and apparent RSS growth during self-play.
        Tuples of ints are hashable, immutable, and ~50x faster to build.
        """
        dm = state.dice_map
        bm = state.board_map
        return (
            tuple(dm.dices),
            tuple(dm.selected_dices),
            dm.grey_out_dices,
            tuple(bm.board_pieces),
            bm.current_player,
            tuple(bm.player01_stack),
            tuple(bm.player02_stack),
        )
    
    def get_action_probs(self, state: GameDTO, temp=1):
        """
        Standard AlphaZero action probability generation, with valid move masking.
        """
        s = self.get_string_hash(state)
        
        # 1. Ensure the root node is expanded before we add noise
        if s not in self.Ps:
            self.search(state) 
            
        # 2. Add Dirichlet Noise to the Root — only during training (temp > 0).
        # Arena evaluation uses temp=0 and must play deterministically; applying noise
        # corrupts PUCT selection for both models equally, producing more draws, fewer
        # decisive games, and high win-rate variance that blocks genuinely better models
        # from passing the arena threshold.
        valids = self.game.get_allowed_moves(state)

        if temp > 0:
            epsilon = 0.25
            alpha = 0.3
            noise = self.rng.dirichlet([alpha] * self.game.game_size)
            self.Ps[s] = (1 - epsilon) * self.Ps[s] + epsilon * noise

            # Re-mask and re-normalize
            self.Ps[s] *= valids
            sum_Ps = np.sum(self.Ps[s])
            if sum_Ps > 0:
                self.Ps[s] /= sum_Ps
            else:
                self.Ps[s] = valids / np.sum(valids)

        # 3. Run the rest of the MCTS Simulations
        for i in range(self.args['numMCTSSims'] - 1):
            self.search(state)

        # 4. Calculate Action Probabilities
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.game_size)]
        counts = [c * v for c, v in zip(counts, valids)]

        if temp == 0:
            max_count = np.max(counts)
            best_as = [a for a in range(self.game.game_size) if counts[a] == max_count and valids[a]]
            best_a = self.rng.choice(best_as)
            probs = np.zeros(len(counts), dtype=np.float32)
            probs[best_a] = 1.0
            return probs

        # Safety fix: Normalize counts before applying temperature to prevent overflow
        max_c = max(counts) if max(counts) > 0 else 1
        counts = [(x / max_c) ** (1. / temp) for x in counts]
        counts = [c * v for c, v in zip(counts, valids)] 
        
        counts_sum = float(sum(counts))
        if counts_sum == 0:
            # All-zero counts — fall back to uniform over valid moves.
            return (np.asarray(valids, dtype=np.float32) / sum(valids))

        # FIX (memory): return a numpy float32 array instead of a Python list.
        # Each pi is stored in train_data for the whole iteration; switching from
        # list-of-floats (~600 B) to float32 ndarray (~140 B) cuts the replay
        # buffer footprint by ~4x at no quality cost.
        return (np.asarray(counts, dtype=np.float32) / np.float32(counts_sum))

    def search(self, state: GameDTO):
        """
        Paranoid MCTS Search with corrected Terminal Evaluation.
        """
        s = self.get_string_hash(state)

        # 1. TERMINAL CHECK (Evaluate the canonical state immediately)
        # FIX (correctness): the previous code used Es[s] == 0 as the sentinel for
        # "not terminal", which collides with a tied-terminal state (game over,
        # both players' piece sums equal -> get_winner returns 0). The fall-through
        # then re-entered expansion on a position with zero valid moves, producing
        # `Ps[s] = valids / 0` -> NaNs that poisoned the entire search tree.
        # Now we represent a terminal draw with a tiny non-zero value (1e-4) and
        # use None as the "not yet evaluated / non-terminal" marker.
        if s not in self.Es:
            if self.game.is_terminal(state):
                winner = self.game.get_winner(state)
                self.Es[s] = winner if winner != 0 else 1e-4
            else:
                self.Es[s] = 0  # 0 means not terminal

        if self.Es[s] != 0:
            return self.Es[s]

        # 2. EXPANSION (Leaf Node)
        if s not in self.Ps:
            # FIXED: Direct GPU call
            self.Ps[s], v = self.nnet.predict(state)
            
            current_valids = self.game.get_allowed_moves(state)
            self.Ps[s] = self.Ps[s] * current_valids
            sum_Ps = np.sum(self.Ps[s])
            
            if sum_Ps > 0:
                self.Ps[s] /= sum_Ps
            else:
                self.Ps[s] = current_valids / np.sum(current_valids)

            self.Vs[s] = current_valids 
            self.Ns[s] = 0
            return v

        # 3. SELECTION (PUCT)
        valids = self.Vs[s]  # cached at expansion — same hash guarantees same valid moves
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.game_size):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args['cpuct'] * self.Ps[s][a] * \
                        math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        # 4. SAFETY CHECK
        if best_act == -1 or valids[best_act] == 0:
             print(f"CRITICAL MISMATCH. Hash: {s}")
             best_act = self.rng.choice([a for a, v in enumerate(valids) if v == 1])

        # 5. TRANSITION
        next_s_raw = self.game.get_next_state(state, best_act)
        turn_switched = next_s_raw.board_map.current_player != state.board_map.current_player

        # Thread temporal history through the MCTS tree.
        # state is canonical (current_player=1) with a live previous_dto chain.
        # get_canonical_form will pass it to _get_canonical_history, which re-maps
        # all history frames consistently to the new canonical player's perspective.
        next_s_raw.previous_dto = state

        next_s = self.game.get_canonical_form(next_s_raw)
        
        # Recursively search the next canonical state
        v = self.search(next_s)
        
        # Invert the value if the turn switched to the other player
        if turn_switched:
            v = -v

        # 6. BACKPROPAGATION
        a = best_act
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v