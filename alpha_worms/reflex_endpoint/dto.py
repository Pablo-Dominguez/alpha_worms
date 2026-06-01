from pydantic import BaseModel, Field
from alpha_worms.reflex_endpoint.models import Dices, Pieces

class GameDTO(BaseModel):
    dice_map: Dices
    board_map: Pieces
    # Wrap the entire type expression in quotes
    previous_dto: "GameDTO | None" = Field(default=None, exclude=True)

# Essential for Pydantic to resolve the string above
GameDTO.model_rebuild()