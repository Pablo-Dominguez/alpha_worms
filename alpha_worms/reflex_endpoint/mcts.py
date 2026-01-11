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
        
        self.Qsa = {}  
        self.Nsa = {}  
        self.Ns = {}   
        self.Ps = {}   
        self.Es = {}   
        self.Vs = {}   

    def get_action_probs(self, state: GameDTO, temp=1):
        for i in range(self.args['numMCTSSims']):
            self.search(state)

        s = self.game.get_string_hash(state)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.game_size)]

        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            probs = [0] * len(counts)
            probs[np.random.choice(best_as)] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, state: GameDTO):
        """
        Recursive search optimized for multi-move turns.
        """
        s = self.game.get_string_hash(state)

        # 1. Check if terminal state
        if s not in self.Es:
            self.Es[s] = self.game.get_winner(state)
        if self.Es[s] != 0:
            # If terminal, return the reward from the perspective of the player who just moved
            return -self.Es[s]

        # 2. Expansion
        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(state)
            valids = self.game.get_allowed_moves(state)
            self.Ps[s] = self.Ps[s] * valids 
            sum_Ps = np.sum(self.Ps[s])
            if sum_Ps > 0:
                self.Ps[s] /= sum_Ps
            else:
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            # v is the NN's estimate of the state for the current player
            return -v

        # 3. Selection (PUCT)
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

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
        
        # --- MULTI-MOVE ADAPTATION ---
        # Your engine needs to tell us if the turn switched
        # next_s: the new state
        # turn_switched: Boolean (True if it's now the other player's turn)
        next_s = self.game.get_next_state(state, a)
        turn_switched = next_s.board_map.current_player != state.board_map.current_player
        
        # Always canonicalize to the perspective of the player whose turn it is
        next_s = self.game.get_canonical_form(next_s)

        # 4. Recursion
        v = self.search(next_s)

        # 5. Backpropagation
        # If the turn switched, the value v (from the other player's view) 
        # must be negated to represent OUR view. 
        # If the turn DID NOT switch, v is already from our perspective.
        if turn_switched:
            v = -v

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        
        # Pass the value back up. If turn switched, negate it for the parent.
        return -v if turn_switched else v