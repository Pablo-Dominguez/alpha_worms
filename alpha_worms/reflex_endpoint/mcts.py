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
        
        self.Qsa = {}  # Stores Q values for (s, a)
        self.Nsa = {}  # Stores visit counts for (s, a)
        self.Ns = {}   # Stores visit counts for state s
        self.Ps = {}   # Stores initial policy (NN probability vector)
        self.Es = {}   # Stores game ended status
        self.Vs = {}   # Stores valid moves mask

    def get_action_probs(self, state: GameDTO, temp=1):
        """
        Runs MCTS simulations and returns the policy vector pi.
        """
        # Run simulations
        for i in range(self.args['numMCTSSims']):
            self.search(state)

        s = self.nnet.tensor_engine.get_string_hash(state)
        
        # Get counts for all possible actions (size of game action space)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.game_size)]

        # --- CASE 1: COMPETITIVE PLAY (Temp -> 0) ---
        # We pick the move with the most visits (argmax)
        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs

        # --- CASE 2: EXPLORATORY PLAY (Temp > 0) ---
        # We perform soft selection based on visit counts
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, state: GameDTO):
        """
        Recursive MCTS search.
        Invariant: Returns the value of 'state' from the perspective of the
        player whose turn it is at 'state'.
        """
        s = self.nnet.tensor_engine.get_string_hash(state)
        print('='*30,'\n',f"Searching state...")
        # 1. CHECK TERMINAL STATE
        if s not in self.Es:
            self.Es[s] = self.game.get_winner(state)
        
        if self.Es[s] != 0:
            # Terminal state found.
            # Assuming get_winner returns +1 if current player won, -1 if lost.
            return self.Es[s]

        # 2. EXPANSION (Leaf Node)
        if s not in self.Ps:
            # Get raw prediction
            self.Ps[s], v = self.nnet.predict(state)
            
            # --- CRITICAL: MASK INVALID MOVES ---
            valids = self.game.get_allowed_moves(state)
            self.Ps[s] = self.Ps[s] * valids  # Zero out illegal moves
            
            # Renormalize
            sum_Ps = np.sum(self.Ps[s])
            if sum_Ps > 0:
                self.Ps[s] /= sum_Ps
            else:
                # If NN masked all valid moves (rare edge case), fallback to uniform valid
                print("All valid moves were masked, falling back to uniform.")
                self.Ps[s] = valids + 0.00000001
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            
            # Return the NN's value estimate for this state
            return v

        # 3. SELECTION (PUCT)
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # Pick the action with the highest Upper Confidence Bound
        for a in range(self.game.game_size):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        
        # 4. TRANSITION (Execute Move)
        # Because we used 'valids' in selection, 'a' is guaranteed safe.
        print(state)
        print(f"Selected action: {a}, move: {valids[a]}")
        print(valids)
        next_s_raw = self.game.get_next_state(state, a)
        
        # Check if the turn switched players (Crucial for Worms)
        turn_switched = next_s_raw.board_map.current_player != state.board_map.current_player
        
        # Canonicalize: Ensure the NN always sees the board from "Player 1's" perspective
        next_s = self.game.get_canonical_form(next_s_raw)

        # 5. RECURSION
        # Get value of the next state (from the next player's perspective)
        v = self.search(next_s)

        # 6. BACKPROPAGATION
        # If the turn switched, the value 'v' is good for the *other* player.
        # It is therefore bad for *us*.
        if turn_switched:
            v = -v
        # If turn did NOT switch (multi-move), 'v' is good for *us* (same player).
        # So we keep v positive.

        # Update Q-values
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        
        # Return the value from *our* perspective to the parent
        return v