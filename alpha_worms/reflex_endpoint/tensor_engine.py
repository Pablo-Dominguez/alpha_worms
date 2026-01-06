
from alpha_worms.reflex_endpoint.models import Dices, Pieces
from alpha_worms.reflex_endpoint.dto import GameDTO
import numpy as np

class TensorEngine:
    def __init__(self):
        
        pass

    def to_tensor(self, game_dto: GameDTO):
        # Initialize the tensor: 33 slots, 55 feature layers
        # Shape: (Slots, Features)
        # The board has 33 slots (0-32). The first 16 slots (0-15) are the first player's stack
        # The next 16 slots (16-31) are the second player's stack, and slot 32 is for pieces on the board.
        # Features: 55 layers
        tensor = np.zeros((33, 55), dtype=np.float32)

        # --- POPULATE DICE FEATURES ---
        for i, value in enumerate(game_dto.dice_map.dices): # availability
            if i in game_dto.dice_map.available_dices:
                # make whole board binary mask indicating dice is available
                tensor[:, 36+i] = len(game_dto.dice_map.available_dices) / len(game_dto.dice_map.dices)
            elif value in game_dto.dice_map.selected_dices and i not in game_dto.dice_map.available_dices:
                tensor[:, 44+i] = len(game_dto.dice_map.selected_dices) / len(game_dto.dice_map.dices)
        
        # --- POPULATE REMAINING DICE FEATURES ---
        # binary worm presence
        tensor[:, 52] = 1.0 if 6 in game_dto.dice_map.selected_dices else 0.0
        # normalized scalar for total selected dice value
        tensor[:, 53] = sum([x if x != 6 else 5 for x in game_dto.dice_map.selected_dices]) / (5 * len(game_dto.dice_map.dices))

        # --- POPULATE FIRST 16 LAYERS (Piece Locations) ---
        for piece_number in range(21, 37):
            # 1. Piece locations
            if piece_number in game_dto.board_map.board_pieces:
                tensor[32, piece_number-21] = 1.0
            elif piece_number in game_dto.board_map.player01_stack:
                idx = game_dto.board_map.player01_stack.index(piece_number)
                tensor[idx, piece_number-21] = 1.0
            elif piece_number in game_dto.board_map.player02_stack:
                idx = game_dto.board_map.player02_stack.index(piece_number)
                tensor[16+idx, piece_number-21] = 1.0
            # 2. Piece availability
                
        # Slot 32 remains 0.0 for these 16 layers unless a piece is "in transition"

        return tensor