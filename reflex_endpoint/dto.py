
from reflex_endpoint.models import Dices, Pieces
import json
import ast

def get_game_log():
    """
    Read game_activity.log and return its content as a list of dicts
    """
    log_entries = []
    with open("./reflex_endpoint/game_activity.log", "r") as log_file:
        for line in log_file:
            # get index of first '{' character
            first_brace_index = line.find('{')
            if first_brace_index != -1:
                json_str = line[first_brace_index:]
                new_entry = ast.literal_eval(json_str)
                if isinstance(new_entry, tuple):
                    # Extract the JSON part from the line
                    log_entries.append(new_entry[0])
                else:
                    log_entries.append(new_entry)
    # cleaned_entries = [x for x in log_entries if x["action"] != "reset_game"]
    return log_entries


def get_dto(step: int = 1):
    """
    This method is currently a placeholder.

    Initially, create a mock file saving outputs from the models of reflex game and use it.

    """

    log_entries = get_game_log()
    ## MIND this entries do not correspond to the same step over the game. 
    # First change in dices are not in the same step as first change in board.
    line_entry = log_entries[step]
    if any('dice' in key for key in line_entry["changes"].keys()):
        dice_entry = line_entry
        board_entry = [entry for entry in log_entries[:step] if any('board' in key for key in entry["changes"].keys())]
        board_entry = board_entry[-1] if board_entry else None
    elif any('board' in key for key in line_entry["changes"].keys()):
        board_entry = line_entry
        dice_entry = [entry for entry in log_entries[:step] if any('dice' in key for key in entry["changes"].keys())]
        dice_entry = dice_entry[-1] if dice_entry else None

    dice_entry = {k:v["after"] for k,v in dice_entry["changes"].items()} if dice_entry else {}
    board_entry = {k:v["after"] for k,v in board_entry["changes"].items()} if board_entry else {}

    dice_map = Dices(
        **dice_entry
    )

    board_map = Pieces(
        **board_entry
    )

    return dice_map, board_map

def get_allowed_moves(step: int = 1):
    """
    This method is currently a placeholder.
    """

    dice_map, board_map = get_dto(step)
    dice_map, board_map = dice_map.model_dump(), board_map.model_dump()

    total_selected = sum([x if x != 6 else 5 for x in dice_map["selected_dices"]]) if 6 in dice_map["selected_dices"] else 0
    other_player = 2 if board_map["current_player"] == 1 else 1
    ## Get allowed dices
    dice_moves = [0 if (x in dice_map["selected_dices"] or dice_map["grey_out_dices"]) else 1 for x in dice_map["dices"]]

    ## Get allowed pieces

    last_piece_opp_stack = board_map[f"player0{other_player}_stack"][-1] if board_map[f"player0{other_player}_stack"] else -1
    
    piece_moves = [1 if (
        (x in board_map["board_pieces"] and  x <= total_selected) or
        (x == last_piece_opp_stack and x == total_selected)
        ) else 0 for x in range(21,37)]

    return piece_moves + dice_moves