import math
import numpy as np
import hashlib
import json
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

    def get_string_hash(self, state: GameDTO) -> str:
        """
        FIX: Uses the DTO structure for a stable, collision-resistant hash.
        This prevents the 'forbidden move' bug caused by byte-level jitter 
        or incomplete state representation in tensors.
        """
        # Converting to dict ensures we capture dices, player stacks, and current player.
        state_dict = state.model_dump()
        state_json = json.dumps(state_dict, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()

    def get_action_probs(self, state: GameDTO, temp=1):
        """
        Standard AlphaZero action probability generation.
        """
        for i in range(self.args['numMCTSSims']):
            self.search(state)

        s = self.get_string_hash(state)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.game_size)]

        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs

        # Softmax-like temperature scaling
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, state: GameDTO):
        """
        Paranoid MCTS Search.
        Never trusts the cached 'Vs' mask for selection logic.
        """
        s = self.get_string_hash(state)

        # 1. TERMINAL CHECK
        if s in self.Es:
            return self.Es[s]

        # 2. EXPANSION (Leaf Node)
        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(state)
            
            # Get FRESH allowed moves
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
        # --- PARANOID FIX: IGNORE CACHE ---
        # We calculate allowed moves freshly for this specific state instance.
        # This guarantees that even if 's' collides, we obey the actual game rules.
        valids = self.game.get_allowed_moves(state) 
        
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

        # 4. SAFETY CHECK (Optional, for peace of mind)
        if best_act == -1 or valids[best_act] == 0:
             print(f"CRITICAL MISMATCH. Hash: {s}")
             print(f"Cached Mask: {self.Vs.get(s)}")
             print(f"Actual Mask: {valids}")
             # Fallback to random valid
             best_act = np.random.choice(np.where(valids==1)[0])

        # 5. TRANSITION & PRE-CANONICAL TERMINAL CHECK
        next_s_raw = self.game.get_next_state(state, best_act)
        winner = self.game.get_winner(next_s_raw)
        
        if winner != 0 or self.game.is_terminal(next_s_raw):
            terminal_v = winner * state.board_map.current_player
            # Cache the terminal result so we don't re-calculate next time
            self.Es[self.get_string_hash(next_s_raw)] = terminal_v 
            v = terminal_v
        else:
            turn_switched = next_s_raw.board_map.current_player != state.board_map.current_player
            next_s = self.game.get_canonical_form(next_s_raw)
            v = self.search(next_s)
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