
import numpy as np

# Create the class GameState, with the following attributes:
# - num_players (int) (2-4), defaults 2
# - param indexes in a class GameStateSettings.
class GameState:
    """
    Represents the state of the game. This is composed of a tensor with shape (board_shape, n_features).
    The board shape is determined by the number of players times the number of pieces (n_stacks) plus number of dices plus the field slot.
    The number of features is the sum of:
    - 16 pieces: to indicate the availability (+1,-1, 0) == (available, unavailable, out) and value (21-36) of each piece. Note you can arithmetically get worm value from piece value.
    - 6 different dice values: to indicate the availability (+1,-1) == (fixed, available) and the value (1-5+) of each dice.
    - 1 turn: to indicate the current turn number.
    - pending: previous turns.
    """
    def __init__(self, num_players: int = 2):
        if num_players < 2 or num_players > 4:
            raise ValueError("Number of players must be between 2 and 4.")
        self.num_players = num_players
        self.tensor = self.build_tensor()

    def build_tensor(self):
        # Create a tensor to represent the game state.
        # The tensor shape is the board x features.
        self.n_pieces = 16
        self.n_dices = 8
        # Assuming a 'field_slot' of size of 1x1
        self.field_slot = 1
        board_shape = self.n_pieces*self.num_players + self.n_dices + self.field_slot

        # Create a tensor with the shape (board_shape, k)
        # where k is the sum of
        # 16 pieces
        # 6 types of dices
        # 1 turn
        # pending: previous turns.
        k = 16 + 6 + 1
        self.n_features = k
        tensor = np.zeros((board_shape, self.n_features), dtype=np.float32)
        
        # Initialize each layer of the tensor. Pending: parameters
        for i in range(self.n_features):
            if i < self.n_pieces: # pieces features
                tensor[-1, i] = -1*(21+i) # negative values for unavailable pieces. Set piece on board slot.
            elif i < self.n_pieces+6: # dice features
                tensor[-self.n_dices-1:-1, i] = 0 # Set fixed dices to 0 (initial placeholder)
            else: # turn features
                tensor[:, i] = 1.0

        return tensor # (board_shape, features) == (41, 23)
    
    def reset_dices(self):
        # Reset the dices to their initial state.
        # Set all dices to 0 (fixed) and available.
        all_dices = (-1)*np.random.randint(1, 7,8)
        for i in range(self.n_dices):
            self.set_dice_value(i, all_dices[i],reset=True)
    
    def throw_dices(self):
        # Throwing the dices and update the tensor accordingly.
        # Get fixed dices
        dices = self.get_dices()
        for dice in range(self.n_dices):
            if np.all(dices[dice,:] >= 0):
                pass
            elif np.any(dices[dice,:] < 0):
                # Set available dices to new random values
                new_dice_value = (-1)*np.random.randint(1, 7)
                self.set_dice_value(dice, new_dice_value, reset=True)
                
                
    def get_dices(self):
        # Get the current state of the dices.
        return self.tensor[-self.n_dices-1:-1, self.n_pieces:self.n_pieces+6]

    def set_dice_value(self, dice, value, reset=False):
        # Set the value of a specific dice.
        if dice < 0 or dice >= self.n_dices:
            raise ValueError("Invalid dice index.")
        if value < -6:
            raise ValueError("Invalid dice value.")
        if reset:
            if value > 0 :
                raise ValueError("Resetting a dice value should be negative.")
            self.tensor[-self.n_dices-1+dice, self.n_pieces:self.n_pieces+6] = 0 # Reset the dice value
        self.tensor[-self.n_dices-1+dice, self.n_pieces+abs(value)-1] = value

    def select_dices(self, dice_value):
        
        dices = self.get_dices()
        # Check if the dice value is valid.
        if np.any(dices[:,dice_value-1] > 0):
            raise ValueError("The value is already selected.")
        else:
            # Get dice indexes where the dice currently has the value.
            dice_indexes = np.where((dices < 0) & (dices == -dice_value)) # could check if second index is equal to dice_value
            if np.all(dice_indexes[1]==dice_value-1):
                print("Selecting dice value:", dice_value)
                # Select the dice.
                for i in range(len(dice_indexes[0])):
                    dice = dice_indexes[0][i]
                    self.set_dice_value(dice, dice_value)