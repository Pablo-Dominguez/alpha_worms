"""
game_recorder.py — Bridge between alpha_worms execute_episode() and worms_logger.

Records one game per activation to the database, turn by turn.
Completely optional: if DATABASE_URL is not set, GameRecorder.create() returns
None and no recording happens.

Usage in execute_episode():
    game_history = Worms.execute_episode(game, mcts=mcts, tensor_engine=te,
                                         game_recorder=recorder)

Usage in worker_play_games():
    recorder = GameRecorder.create() if worker_id == 0 and game_num == 1 else None
"""
import logging
import os

logger = logging.getLogger(__name__)


def _dice_to_char(v: int) -> str:
    """Convert engine dice value to TurnRecorder character.  6 (worm face) → 'w'."""
    return 'w' if v == 6 else str(v)


class GameRecorder:
    """
    Stateful per-game recorder.  Call .step() after each get_next_state(), then
    .finish() when the game ends.  All methods are non-fatal.
    """

    def __init__(self, match_id: str, db):
        self._match_id = match_id
        self._db = db
        self._turn_number = 0   # increments each time a player's turn closes
        self._turn_string = ""  # worms_logger turn string being built for this turn
        self._finished = False  # guard against double-calls to finish()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls) -> "GameRecorder | None":
        """
        Return a ready GameRecorder if DATABASE_URL is set, else None.
        Uses player_id 1 and 2 (the two MCTS agents).
        """
        if not os.environ.get("DATABASE_URL"):
            return None
        try:
            from worms_logger import DatabaseManager
            db = DatabaseManager()
            match_id = db.start_match(player1_id=1, player2_id=2)
            logger.info(f"[GameRecorder] Recording game → match_id={match_id}")
            return cls(match_id, db)
        except Exception as exc:
            logger.warning(f"[GameRecorder] Could not start match: {exc}")
            return None

    # ------------------------------------------------------------------
    # Main hooks
    # ------------------------------------------------------------------

    def step(self, state_before, action: int, state_after) -> None:
        """
        Call once per get_next_state() with the raw (non-canonical) states.

        Reconstructs the worms_logger turn string incrementally:
          - Dice selection  (action 16-23) → appends R(available)K(kept) segment
          - Piece action    (action 0-15)  → closes the turn with :T{n} or :S{n}
          - Explicit roll   (action 24)    → closes with :F if turn ended, else no-op
        Auto-fails triggered inside the engine are detected by comparing
        current_player before and after the call.
        """
        try:
            from worms_logger import TurnRecorder
            self._record_step(state_before, action, state_after, TurnRecorder)
        except Exception as exc:
            logger.warning(f"[GameRecorder] step() error (non-fatal): {exc}")

    def finish(self, reward: float) -> None:
        """
        Call when execute_episode() is about to return.
        - If a partial turn string exists (turn-limit hit mid-turn), logs it with
          a ':LIMIT' suffix so it's identifiable in the DB.
        - Marks the match as finished and records the winner.
        Idempotent: safe to call more than once.
        """
        if self._finished:
            return
        self._finished = True
        try:
            if self._turn_string:
                # Turn limit hit before a terminal action closed this turn
                incomplete = self._turn_string + ":LIMIT"
                self._db.log_turn(self._match_id, self._turn_number, incomplete)
                logger.info(f"[GameRecorder] Logged turn-limit turn: {incomplete}")

            # Persist winner (reward from execute_episode is perspective of player 1)
            # reward > 0 → player 1 wins; < 0 → player 2 wins; ~0 → draw
            if reward > 1e-4:
                winner = 1
            elif reward < -1e-4:
                winner = 2
            else:
                winner = None  # draw / tied

            self._finalize_match(winner)
            logger.info(
                f"[GameRecorder] Finished match {self._match_id} "
                f"({self._turn_number} turns, winner={winner})"
            )
        except Exception as exc:
            logger.warning(f"[GameRecorder] finish() error (non-fatal): {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_step(self, state_before, action: int, state_after, TurnRecorder) -> None:
        board_before = state_before.board_map
        board_after  = state_after.board_map
        dice_before  = state_before.dice_map

        player_switched = board_before.current_player != board_after.current_player

        if action < 16:
            # ── Piece action: terminal, closes this player's turn ──────────────
            piece_num = action + 21
            if piece_num in board_before.board_pieces:
                # Piece was on the shared board → take
                self._turn_string = TurnRecorder.append_take(self._turn_string, piece_num)
            else:
                # Piece was on the opponent's stack → steal
                self._turn_string = TurnRecorder.append_steal(self._turn_string, piece_num)
            self._log_and_reset()

        elif action < 24:
            # ── Dice selection: adds one R(available)K(kept) segment ──────────
            available_indices = dice_before.available_dices
            available_values  = [dice_before.dices[i] for i in available_indices]
            # The engine commits ALL dice showing the same face value in one action
            selected_value = dice_before.dices[action - 16]
            kept_values    = [v for v in available_values if v == selected_value]

            self._turn_string = TurnRecorder.append_roll(
                self._turn_string,
                [_dice_to_char(v) for v in available_values],
                [_dice_to_char(v) for v in kept_values],
            )

            if player_switched:
                # select_dice() auto-threw remaining dice and got a dead reroll → fail
                self._turn_string = TurnRecorder.append_fail(self._turn_string)
                self._log_and_reset()

        else:
            # ── Explicit roll (action 24): player chose to re-roll ────────────
            # No new R(...)K(...) segment for the roll itself; the subsequent
            # select_dice call will capture R(new dice)K(kept).
            if player_switched:
                # throw_dices() internally detected a dead reroll after this roll
                self._turn_string = TurnRecorder.append_fail(self._turn_string)
                self._log_and_reset()

    def _log_and_reset(self) -> None:
        self._db.log_turn(self._match_id, self._turn_number, self._turn_string)
        logger.debug(
            f"[GameRecorder] Logged turn {self._turn_number}: {self._turn_string}"
        )
        self._turn_number += 1
        self._turn_string  = ""

    def _finalize_match(self, winner: int | None) -> None:
        """Update match row with winner + status='finished'."""
        try:
            from worms_logger import DatabaseManager
            from worms_logger.models_db import Match
            db = DatabaseManager()
            db._connect()
            session = db.Session()
            match = session.query(Match).filter(Match.match_id == self._match_id).first()
            if match:
                match.winner = winner
                match.status = "finished"
                session.commit()
            session.close()
        except Exception as exc:
            logger.warning(f"[GameRecorder] Could not finalize match: {exc}")
