
from alpha_worms.reflex_endpoint.models import Dices, Pieces
import ast

class GameDTO:
    def __init__(self, dice_map: Dices, board_map: Pieces):
        self.dice_map = dice_map
        self.board_map = board_map

def parse_game_log():
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
    Initially, create a mock file saving outputs from the models of reflex game and use it.

    Currently, parsing the game 'steps' from the log file and returning the corresponding DTOs.
    Eventually, will need to properly define when in the game you get a step, i.e., a decision point.

    Decisions:
    - Select a dice.
    - Select a piece.
    - Roll again.

    1. When dices are selected, a decision was taken.
    2. When a piece is moved, a decision was taken.

    OR:
    1. After resetting the game: Select dice.
    2. After end of turn: Select dice.
    3. After rolling dices: Select dice.
    4. After selecting a dice, if available pieces: Select piece OR Roll again.

    Total actions possible: 25


    """

    log_entries = parse_game_log()
    ## MIND this entries do not correspond to the same step over the game. 
    # First change in dices are not in the same step as first change in board.

    line_entry = log_entries[step]

    if line_entry['action'] in ['reset_game', 'end_turn']:
        if 'dices' in line_entry['changes']:
            dice_entry = line_entry['changes']
            board_entry = log_entries[step+1]['changes']
        else:
            board_entry = line_entry['changes']
            dice_entry = log_entries[step-1]['changes']
        dice_entry = {k:v["after"] for k,v in dice_entry.items()}
        board_entry = {k:v["after"] for k,v in board_entry.items()}
    elif line_entry['action'] == 'throw_dices':
        dice_entry = line_entry['changes']
        dice_entry = {k:v["after"] for k,v in dice_entry.items()}

        board_entry = [entry for entry in log_entries[:step] if any('board' in key for key in entry["changes"].keys())][-1]
        board_entry = {k:v["after"] for k,v in board_entry['changes'].items()}

    elif line_entry['action'] == 'select_dice':
        dice_entry = line_entry['changes']
        dice_entry = {k:v["before"] for k,v in dice_entry.items()}
        board_entry = [entry for entry in log_entries[:step] if any('board' in key for key in entry["changes"].keys())][-1]
        board_entry = {k:v["after"] for k,v in board_entry['changes'].items()}

    elif line_entry['action'] in ['fetch_from_board', 'fetch_from_stack']:
        board_entry = line_entry['changes']
        board_entry = {k:v["before"] for k,v in board_entry.items()}
        dice_entry = [entry for entry in log_entries[:step] if any('dices' in key for key in entry["changes"].keys())][-1]
        dice_entry = {k:v["after"] for k,v in dice_entry['changes'].items()}
    
    dice_map = Dices(
        **dice_entry
    )

    board_map = Pieces(
        **board_entry
    )

    return dice_map, board_map

def get_allowed_moves(step: int = 1, dev_mode: bool = False, dice_map: Dices = None, board_map: Pieces = None):
    """
    This method is currently a placeholder.
    """
    if dev_mode:
        dice_map, board_map = get_dto(step)
    dice_map, board_map = dice_map.model_dump(), board_map.model_dump()

    total_selected = sum([x if x != 6 else 5 for x in dice_map["selected_dices"]]) if 6 in dice_map["selected_dices"] else 0
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

## make a method so I can test this in console, changing step everytime I hit enter
# if __name__ == "__main__":
#     step = 0
#     while True:
#         dto_dice, dto_board = get_dto(step)
#         allowed_moves = get_allowed_moves(step)
#         print(f"Step: {step}")
#         print("DTO Dices:", dto_dice)
#         print("DTO Board:", dto_board)
#         print("Allowed Moves:", allowed_moves)
#         step += 1
#         print('-----------------------'*2)
#         input("Press Enter to continue to the next step...")