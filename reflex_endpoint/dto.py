

def get_dto():
    """
    This method is currently a placeholder.

    Initially, create a mock file saving outputs from the models of reflex game and use it.

    """
    dice_map = {
        "dices": [0,1,2,3,4,5,6,7],
        "available_dices": [d for d in [0,1,2,3,4,5,6,7] if d % 2 == 0],
        "selected_dices": [d for d in [0,1,2,3,4,5,6,7] if d % 2 == 0 and d > 2],
        # "grey_out_dices": self.dices_model.grey_out_dices

    }
    board_map = {
        "board": [x for x in range(21,37)],
        "p1_stack": [],
        "p2_stack": [],
        "current_player": 1
    }

    return dice_map, board_map

def get_allowed_moves():
    """
    This method is currently a placeholder.
    """

    dice_map, board_map = get_dto()
    total_selected = sum([x if x != 6 else 5 for x in dice_map["selected_dices"]]) if 6 in dice_map["selected_dices"] else 0
    other_player = 2 if board_map["current_player"] == 1 else 1
    ## Get allowed dices
    dice_moves = [0 if (x in dice_map["selected_dices"] or dice_map["grey_out_dices"]) else 1 for x in dice_map["dices"]]

    ## Get allowed pieces

    last_piece_opp_stack = board_map[f"p{other_player}_stack"][-1] if board_map[f"p{other_player}_stack"] else -1
    
    piece_moves = [1 if (
        (x in board_map["board"] and  x <= total_selected) or
        (x == last_piece_opp_stack and x == total_selected)
        ) else 0 for x in range(21,37)]

    return piece_moves + dice_moves