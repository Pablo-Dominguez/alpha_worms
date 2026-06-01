
from pydantic import BaseModel, Field
import numpy as np

def rand_dices():
    return [int(x) for x in np.random.default_rng().integers(1, 7, size=8)]

class Dices(BaseModel):
    dices: list[int] = Field(default_factory=rand_dices)
    available_dices: list[int] = Field(default_factory=lambda: [0,1,2,3,4,5,6,7])
    selected_dices: list[int] = Field(default_factory=list)
    grey_out_dices: int = Field(default=0)

class Pieces(BaseModel):
    board_pieces: list[int] = Field(default_factory=lambda: list(range(21, 37)))
    current_player: int = Field(default=1)
    player01_stack: list[int] = Field(default_factory=list)
    player02_stack: list[int] = Field(default_factory=list)