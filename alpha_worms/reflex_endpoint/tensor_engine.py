
from alpha_worms.reflex_endpoint.dto import GameDTO
from alpha_worms.reflex_endpoint.game_engine import GameEngine
import numpy as np
import hashlib
import json
import matplotlib.cm as cm
from PIL import Image, ImageDraw


class TensorEngine:
    def __init__(self):
        
        pass

    def to_tensor(self, game_dto: GameDTO):
        # Initialize the tensor: 33 slots, 55 feature layers
        # Shape: (Slots, Features)
        # The board has 33 slots (0-32). The first 16 slots (0-15) are the first player's stack
        # The next 16 slots (16-31) are the second player's stack, and slot 32 is for pieces on the board.
        # Features: 55 layers
        # Feature order breakdown:
        # 0-15: Piece locations for pieces 21-36 (16 pieces)
        # 16-31: Piece availability for pieces 21-36 (16 pieces)
        # 32-35: Worm values (1-4)
        # 36-43: Dice availability (8 dice)
        # 44-51: Dice selection (8 dice)
        # 52: Binary worm presence (1 if worm selected, else 0)
        # 53: Normalized total selected dice value
        # 54: Turn indicator (current player - 1)
        tensor = np.zeros((55, 33), dtype=np.float32)

        # --- POPULATE DICE FEATURES ---
        for i, value in enumerate(game_dto.dice_map.dices): # availability
            if i in game_dto.dice_map.available_dices:
                # make whole board binary mask indicating dice is available
                tensor[36+i, :] = len([x for x in game_dto.dice_map.available_dices if x == i]) / len(game_dto.dice_map.dices)
            elif value in game_dto.dice_map.selected_dices and i not in game_dto.dice_map.available_dices:
                # make whole board binary mask indicating dice is selected
                tensor[44+i, :] = len([x for x in game_dto.dice_map.selected_dices if x == value]) / len(game_dto.dice_map.dices)

        # --- POPULATE REMAINING GAME FEATURES ---
        # binary worm presence
        tensor[52, :] = 1.0 if 6 in game_dto.dice_map.selected_dices else 0.0
        # normalized scalar for total selected dice value
        tensor[53, :] = sum([x if x != 6 else 5 for x in game_dto.dice_map.selected_dices]) / (5 * len(game_dto.dice_map.dices))
        # turn indicator
        tensor[54, :] = game_dto.board_map.current_player - 1
        # Game history features can be added here as needed 

        available_pieces = GameEngine().get_allowed_moves(game_dto)[:16]
        # --- POPULATE PIECE FEATURES ---
        for piece_number in range(21, 37):
            # 1. Piece locations
            if piece_number in game_dto.board_map.board_pieces:
                tensor[piece_number-21, 32] = 1.0
            elif piece_number in game_dto.board_map.player01_stack:
                idx = game_dto.board_map.player01_stack.index(piece_number)
                tensor[piece_number-21, idx] = 1.0
            elif piece_number in game_dto.board_map.player02_stack:
                idx = game_dto.board_map.player02_stack.index(piece_number)
                tensor[piece_number-21, 16+idx] = 1.0
            # 2. Piece availability
            available = available_pieces[piece_number-21]
            tensor[16 + (piece_number-21), :] = available
        # --- Worm values ---
        for i in range(4):
            # Join the location planes: Maybe this should be a summed mask
            new_layer = (i+1)*np.sum([layer for layer in tensor[4*i:4*(i+1), :]], axis=0)/ 16.0
            tensor[32 + i, :] = new_layer

        return tensor

    def to_pillow_heatmap(self, tensor):

        scale = 10  # Scale factor for better visibility
        
        # 1. Apply a colormap (like 'viridis' or 'magma')
        # This turns the (55, 33) float array into a (55, 33, 4) RGBA array
        mapped_color = cm.viridis(tensor) 
        
        # 2. Convert to 0-255 RGB
        rgb_data = (mapped_color[:, :, :3] * 255).astype(np.uint8)
        
        # 3. Create the base image and resize
        # Current size: (width=33, height=55)
        img = Image.fromarray(rgb_data, mode='RGB')
        width, height = 33 * scale, 55 * scale
        img = img.resize((width, height), resample=Image.NEAREST)
        
        # 4. Draw the Divider Lines
        draw = ImageDraw.Draw(img)
        
        # Functional block boundaries (Y-coordinates in channel space)
        # We add 0.5 to put the line between the pixels
        boundaries = [16, 32, 36, 44, 52] 
        
        for line_idx in boundaries:
            # Calculate Y position: (channel_index * scale)
            y = line_idx * scale
            # Draw a semi-transparent or solid white line
            draw.line([(0, y), (width*3, y)], fill=(255, 255, 255), width=1)

        return img