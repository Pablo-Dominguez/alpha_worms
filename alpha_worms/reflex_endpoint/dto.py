from pydantic import BaseModel
from alpha_worms.reflex_endpoint.models import Dices, Pieces

class GameDTO(BaseModel):
    dice_map: Dices
    board_map: Pieces
    # Use Optional["GameDTO"] or "GameDTO | None" to allow the default None
    previous_dto: "GameDTO" | None = None

# Add this right after the class definition
GameDTO.model_rebuild()