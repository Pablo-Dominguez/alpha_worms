
from alpha_worms.reflex_endpoint.models import Dices, Pieces
from pydantic import BaseModel

class GameDTO(BaseModel):
    dice_map: Dices
    board_map: Pieces
