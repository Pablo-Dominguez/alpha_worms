
from pydantic import BaseModel, Field, computed_field
import numpy as np


class Dices(BaseModel):
    dices: list[int] = Field(
        default=[int(x) for x in np.random.randint(1, 7, size=8)],
        description="List of all dice values available in the game."
    )
    available_dices: list[int] = Field(
        default=[0,1,2,3,4,5,6,7],
        description="List of available dice indices that can be used in the current turn."
    )
    selected_dices: list[int] = Field(
        default=[],
        description="List of selected dice values in the current turn."
    )
    grey_out_dices: int = Field(
        default=0,
        description="Indicates whether the dices are greyed out (not usable)."
    )

class Pieces(BaseModel):
    board_pieces: list[int] = Field(
        default=list(range(21, 37)),
        description="List of pieces currently on the board."
    )
    current_player: int = Field(
        default=1,
        description="Indicates the current player."
    )
    player01_stack: list[int] = Field(
        default=[],
        description="Pieces in player 1's stack."
    )
    player02_stack: list[int] = Field(
        default=[],
        description="Pieces in player 2's stack."
    )
