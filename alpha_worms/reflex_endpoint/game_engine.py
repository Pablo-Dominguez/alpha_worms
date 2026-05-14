
from alpha_worms.reflex_endpoint.models import Dices, Pieces
from alpha_worms.reflex_endpoint.dto import GameDTO
import numpy as np


class RawEngine:
    def __init__(self):
        pass

    def get_initial_state(self) -> GameDTO:
        """Generates the initial game state."""
        dices: list[int] = [int(x) for x in np.random.randint(1, 7, size=8)]
        available_dices: list[int] = [0,1,2,3,4,5,6,7]
        selected_dices: list[int] = []
        dices_model = Dices(dices=dices,
                             available_dices=available_dices,
                             selected_dices=selected_dices,
                             grey_out_dices = 0)

        pieces_model = Pieces(
            board_pieces=list(range(21, 37)),
            current_player=1,
            player01_stack=[],
            player02_stack=[]
        )

        return GameDTO(dice_map=dices_model, 
                        board_map=pieces_model,
                        previous_dto=None)

    def is_terminal(self, game_dto: GameDTO) -> bool:
        """
        Check if the game has reached a terminal state (i.e., one player has won).
        """
        return len(game_dto.board_map.board_pieces) == 0
    
    def get_score_winner(self, game_dto: GameDTO) -> int:
        """Calculates who is leading based on piece values."""
        p1_stack = game_dto.board_map.player01_stack
        p2_stack = game_dto.board_map.player02_stack
        
        # Your existing point logic
        p1_sum = sum([(p - 21) // 4 for p in p1_stack])
        p2_sum = sum([(p - 21) // 4 for p in p2_stack])
        
        if p1_sum > p2_sum: return 1
        if p2_sum > p1_sum: return -1
        return 0

    def get_winner(self, game_dto: GameDTO) -> int:
        if self.is_terminal(game_dto):
            return self.get_score_winner(game_dto)
        return 0
    
    def get_total_selected(self, game_dto: GameDTO) -> int:
        """Calculates the total value of selected dices."""
        return sum([x if x != 6 else 5 for x in game_dto.dice_map.selected_dices])

    def end_turn(self, game_dto: GameDTO):
        """Switches the turn to the other player."""

        # IMPORTANT: Use .model_copy(deep=True) or else both variables will point to the new data
        old_pieces_version = game_dto.board_map.model_copy(deep=True)

        # Switch current player
        old_pieces_version.current_player = game_dto.board_map.current_player % 2 + 1

        # new_seed = self.game_generator(1)[0]*10000
        dices: list[int] = [int(x) for x in np.random.randint(1, 7, size=8)]
        available_dices: list[int] = [0,1,2,3,4,5,6,7]
        selected_dices: list[int] = []
        new_dices_model = Dices(dices=dices,
                                 available_dices=available_dices,
                                 selected_dices=selected_dices,
                                 grey_out_dices = 0) # Reset dices for next turn
        
        return GameDTO(dice_map=new_dices_model, board_map=old_pieces_version, previous_dto=None)

    def fetch_from_board(self, game_dto: GameDTO, piece_number: int):
        """
        Given a board state and a piece, takes the piece from the board 
        into the current player's stack.
        """
        old_dice_model = game_dto.dice_map.model_copy(deep=True)  
        new_pieces_version = game_dto.board_map.model_copy(deep=True)

        if piece_number in new_pieces_version.board_pieces:
            # Remove the piece from the board.
            # --- 1. Calculate New Board State ---
            # Create a new list without the piece (safe mutation)
            new_pieces_version.board_pieces = [p for p in new_pieces_version.board_pieces if p != piece_number]
            
            # --- 2. Calculate New Stacks State ---
            if new_pieces_version.current_player == 1:
                current_stack = new_pieces_version.player01_stack
                current_stack.append(piece_number)
                new_pieces_version.player01_stack = current_stack
            else:
                current_stack = new_pieces_version.player02_stack
                current_stack.append(piece_number)
                new_pieces_version.player02_stack = current_stack

            # If no more board pieces, end game
            if self.is_terminal(game_dto):
                winner = self.get_winner(game_dto)
                print(f"Player {winner} wins! Game over.")
            
            return self.end_turn(GameDTO(dice_map=old_dice_model, board_map=new_pieces_version, previous_dto=None))
        
    def fetch_from_stack(self, game_dto: GameDTO,piece_number: int):
        """Moves a piece between a player's stack and the board."""
        old_dice_model = game_dto.dice_map.model_copy(deep=True)  
        old_pieces_version = game_dto.board_map.model_copy(deep=True)

        current_stack = old_pieces_version.player02_stack if old_pieces_version.current_player == 2 else old_pieces_version.player01_stack
        opp_stack = old_pieces_version.player01_stack if old_pieces_version.current_player == 2 else old_pieces_version.player02_stack
        if piece_number in current_stack and piece_number == current_stack[-1]:
            # Place the piece on the board
            current_stack.remove(piece_number)
            old_pieces_version.board_pieces.append(piece_number)
            old_pieces_version.board_pieces.sort()
            if old_pieces_version.board_pieces[-1] != piece_number:
                old_pieces_version.board_pieces = old_pieces_version.board_pieces[:-1]
        elif piece_number in opp_stack and piece_number == opp_stack[-1]:
            # Steal the piece from the opponent's stack
            opp_stack.remove(piece_number)
            current_stack.append(piece_number)
        else:
            # raise("Piece not in players stacks.")
            pass
        
        return self.end_turn(GameDTO(dice_map=old_dice_model, board_map=old_pieces_version, previous_dto=None))
    
    def grey_out_dices(self, dice_model: Dices, reset: bool = False):
        """Greys out all available dices, preventing further selection."""
        dice_model.grey_out_dices = 1 if not reset else 0
        return dice_model

    def select_dice(self, game_dto: GameDTO, dice_index: int):
        """Selects a dice from the available dices.""" 
        new_dice_model = game_dto.dice_map.model_copy(deep=True)

        if dice_index in new_dice_model.available_dices and new_dice_model.grey_out_dices == 0:
            # Find indexes of available dices with the given value
            dice_value = new_dice_model.dices[dice_index]
            available_indexes = [i for i, value in enumerate(new_dice_model.dices) if value == dice_value and i in new_dice_model.available_dices]
            new_available_idxs = [i for i in new_dice_model.available_dices if i not in available_indexes]
            # Update available_dices and selected_dices
            new_dice_model.available_dices = new_available_idxs
            new_dice_model.selected_dices = new_dice_model.selected_dices + [dice_value]*len(available_indexes)

             # After selecting a dice, automatically throw the remaining dices if no piece is available
            total_selected = self.get_total_selected(GameDTO(dice_map=new_dice_model, board_map=game_dto.board_map))
            available_pieces = [p for p in game_dto.board_map.board_pieces if p <= total_selected]
            opp_stack = game_dto.board_map.player02_stack if game_dto.board_map.current_player == 1 else game_dto.board_map.player01_stack
            available_pieces = available_pieces + [p for i,p in enumerate(opp_stack) if p == total_selected and i == len(opp_stack)-1]
            available_pieces = [] if not 6 in new_dice_model.selected_dices else available_pieces

            if not available_pieces:
                return self.throw_dices(GameDTO(dice_map=new_dice_model, board_map=game_dto.board_map, previous_dto=None))
            else:
                new_dice_model = self.grey_out_dices(new_dice_model)
        return GameDTO(dice_map=new_dice_model, board_map=game_dto.board_map, previous_dto=None)

    def throw_dices(self, game_dto: GameDTO):
        """Simulates throwing the dices and updates their values."""

        # IMPORTANT: Use .model_copy(deep=True) or else both variables will point to the new data
        new_dices_version = game_dto.dice_map.model_copy(deep=True)
        new_pieces_version = game_dto.board_map.model_copy(deep=True)

        new_dices_version = self.grey_out_dices(new_dices_version, reset=True)

        new_dices =[(x,int(y)) for x,y in zip(new_dices_version.available_dices,np.random.randint(1, 7, size=len(new_dices_version.available_dices)))]
        total_dices = [(new_dices_version.dices[k] if k not in new_dices_version.available_dices else [y for (x,y) in new_dices if k==x][0]) 
                       for k in range(8)]
        # Update dices with new values
        new_dices_version.dices = total_dices


        ## If after throwing, no available dices to get, lose a piece and turn is over
        availables_ls = [x for x in new_dices_version.available_dices if new_dices_version.dices[x] not in new_dices_version.selected_dices]
        if not availables_ls: ## need to drop a piece
            # get current player last stack piece
            player_stack = new_pieces_version.player01_stack if new_pieces_version.current_player ==1 else new_pieces_version.player02_stack
            if player_stack:
                piece_to_drop = player_stack[-1]
                return self.fetch_from_stack(GameDTO(dice_map=new_dices_version, board_map=new_pieces_version, previous_dto=None), piece_to_drop)
            else:
                return self.end_turn(GameDTO(dice_map=new_dices_version, board_map=new_pieces_version, previous_dto=None))
        else:
            return GameDTO(dice_map=new_dices_version, board_map=new_pieces_version, previous_dto=None)


    
class GameEngine(RawEngine):
    def __init__(self):
        super().__init__()
        self.game_size = 25  # Total number of possible actions/moves
    
    def get_allowed_moves(self, game_dto: GameDTO):
        """
        Takes a Game State DTO and returns allowed moves.
        Moves are represented as a list of 0s and 1s, where 1 means allowed.
        The order of moves is:
        - Pieces: 21 to 36 (16 pieces)
        - Dices: 1 to 6 (8 dices)
        - Roll again: 1 (1 move)
        Total: 25 moves
        """

        dice_map, board_map = game_dto.dice_map.model_dump(), game_dto.board_map.model_dump()

        total_selected = self.get_total_selected(game_dto)
        other_player = 2 if board_map["current_player"] == 1 else 1

        ## Get allowed dices
        dice_moves = [0 if (x in dice_map["selected_dices"] or dice_map["grey_out_dices"]) else 1 for x in dice_map["dices"]]

        ## Check if can roll again
        roll_again = [dice_map['grey_out_dices']]

        ## Get allowed pieces
        last_piece_opp_stack = board_map[f"player0{other_player}_stack"][-1] if board_map[f"player0{other_player}_stack"] else -1
        
        piece_moves = [1 if (
            (x in board_map["board_pieces"] and  x <= total_selected) or
            (x == last_piece_opp_stack and x == total_selected)
            ) else 0 for x in range(21,37)]

        return piece_moves + dice_moves + roll_again
    
    def get_next_state(self, game_dto: GameDTO, action: int) -> GameDTO:
        """
        Given a game state and an action, returns the next game state (as a new GameDTO).
        """

        allowed_moves = self.get_allowed_moves(game_dto)
        if allowed_moves[action] == 0:
            print('Dice map',game_dto.dice_map)
            print('Piece map',game_dto.board_map)
            print('Allowed moves',allowed_moves)
            print('Selected action',action)
            raise ValueError("Action not allowed in the current state.")
        else:
            if action < 16:
                # Fetch from board
                piece_number = action + 21  # Adjusting index to match piece numbering
                if piece_number in game_dto.board_map.board_pieces:
                    return self.fetch_from_board(GameDTO(dice_map=game_dto.dice_map, board_map=game_dto.board_map, previous_dto=None), piece_number)
                else:
                    return self.fetch_from_stack(GameDTO(dice_map=game_dto.dice_map, board_map=game_dto.board_map, previous_dto=None), piece_number)
            elif action < 24:
                # Select dice
                dice_number = action - 16  # Adjusting index to match dice numbering
                return self.select_dice(GameDTO(dice_map=game_dto.dice_map, board_map=game_dto.board_map, previous_dto=None), dice_number)
            else:
                # Roll again
                return self.throw_dices(GameDTO(dice_map=game_dto.dice_map, board_map=game_dto.board_map, previous_dto=None))
            
    def get_canonical_form(self, game_dto: GameDTO) -> GameDTO:
        """
        Returns a GameDTO where the current player is always 'Player 1',
        and recursively canonicalizes the history chain.
        """
        # 1. Determine if we need to swap stacks (Perspective Shift)
        # We always swap if the root of THIS call is Player 2
        should_swap = game_dto.board_map.current_player == 2

        new_board_map = game_dto.board_map.model_copy(deep=True)
        new_dice_map = game_dto.dice_map.model_copy(deep=True)

        if should_swap:
            # Swap stacks so 'current player' is always in the P1 slot
            new_board_map.player01_stack = game_dto.board_map.player02_stack
            new_board_map.player02_stack = game_dto.board_map.player01_stack
            # Note: In a true canonical form, we normalize the ID to 1
            new_board_map.current_player = 1 
        
        # 2. Create the new DTO
        canonical_dto = GameDTO(dice_map=new_dice_map, board_map=new_board_map)

        # 3. RECURSIVE STEP: Handle the history
        # Removed to prevent memory chains: 
        canonical_dto.previous_dto = self._get_canonical_history(game_dto.previous_dto, swap=should_swap)

        return canonical_dto

    def _get_canonical_history(self, dto: GameDTO, swap: bool, max_depth: int = 7) -> GameDTO:
        """
        Iterative helper to maintain the swap state down the history chain.
        Prevents RecursionError and limits history depth to save memory and CPU.
        """
        if dto is None:
            return None

        # Helper to apply your exact transformation logic to a single node
        # Helper to apply your exact transformation logic to a single node
        def _process_node(node_dto: GameDTO) -> GameDTO:
            # 1. SHALLOW COPY: 100x faster, perfectly safe for read-only history
            new_board = node_dto.board_map.model_copy(deep=True) 
            
            if swap:
                new_board.player01_stack, new_board.player02_stack = (
                    node_dto.board_map.player02_stack, node_dto.board_map.player01_stack
                )
                # Flip the turn indicator for historical accuracy in the tensor
                new_board.current_player = 1 if node_dto.board_map.current_player == 2 else 2
                
            # 2. Re-use the EXACT dice map. No need to copy it at all!
            return GameDTO(dice_map=node_dto.dice_map, board_map=new_board)

        # 1. Process the head of the chain (current state)
        head_new_dto = _process_node(dto)
        
        # 2. Setup pointers for the iterative traversal
        current_new = head_new_dto
        current_old = dto.previous_dto
        depth = 1 # The head counts as depth 1

        # 3. Walk backwards down the linked list using a while loop
        while current_old is not None and depth <= max_depth:
            # Copy and swap the historical state
            next_new_dto = _process_node(current_old)
            
            # Link it to our newly built chain
            current_new.previous_dto = next_new_dto
            
            # Move the pointers forward
            current_new = next_new_dto
            current_old = current_old.previous_dto
            depth += 1

        # 4. Explicitly sever the chain here.
        # This acts as a firewall against memory leaks, ensuring the garbage 
        # collector can delete the rest of the 1000+ move history.
        current_new.previous_dto = None

        return head_new_dto