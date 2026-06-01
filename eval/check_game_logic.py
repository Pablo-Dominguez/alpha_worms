#!/usr/bin/env python3
"""
eval/check_game_logic.py

Reads recorded games from nn_game_records.db (via worms_logger) and validates
each turn for game-logic consistency:

  1. Turn string is well-formed (parseable R(...)K(...) segments + terminal)
  2. Kept dice are a subset of available dice in each roll step
  3. Each roll step keeps all dice of ONE face value (engine invariant)
  4. No face value is committed twice within the same turn
  5. :T{n} / :S{n}  — accumulated score ≥ n, worm committed, piece in 21–36

Run from the project root:
    poetry run python eval/check_game_logic.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DATABASE_URL", "sqlite:///./nn_game_records.db")

# ---------------------------------------------------------------------------
# Turn-level logic checks
# ---------------------------------------------------------------------------

_ROLL_RE = re.compile(r"^R\(([^)]*)\)K\(([^)]*)\)$")


def _score(chars: list[str]) -> int:
    """Sum dice values: 'w' (worm) = 5, everything else = face int."""
    return sum(5 if c == "w" else int(c) for c in chars)


def check_turn(match_id: str, turn_num: int, raw: str) -> list[str]:
    errors  = []
    prefix  = f"match={match_id[:8]}… turn={turn_num}"

    # Strip :LIMIT suffix left by the turn-limit path in execute_episode.
    # is_limit=True means the turn was cut off by the game step-limit — a
    # terminal action is expected to be absent, so we must not report that
    # as an error.  Roll steps that *are* present are still validated.
    is_limit = raw.endswith(":LIMIT")
    turn_str = raw.removesuffix(":LIMIT") if is_limit else raw
    if not turn_str:
        return errors  # empty mid-turn at game limit — nothing to validate

    from worms_logger import TurnRecorder
    actions = TurnRecorder.parse_actions(turn_str)

    terminal   = actions[-1] if actions and actions[-1].startswith(":") else None
    roll_steps = [a for a in actions if not a.startswith(":")]

    # ── 1. Basic structure ──────────────────────────────────────────────────
    if not terminal and not is_limit:
        errors.append(f"{prefix}: no terminal action found in '{turn_str}'")
    if not roll_steps:
        errors.append(f"{prefix}: no R(...)K(...) roll steps in '{turn_str}'")
        return errors

    # ── 2. Per-step checks ──────────────────────────────────────────────────
    committed: list[str] = []   # face chars committed so far this turn

    for step_idx, step in enumerate(roll_steps):
        m = _ROLL_RE.match(step)
        if not m:
            errors.append(f"{prefix}: malformed roll step '{step}'")
            continue

        available = list(m.group(1))   # one char per die face
        kept      = list(m.group(2))

        if not kept:
            errors.append(f"{prefix}: step {step_idx} kept nothing (empty K)")
            continue

        # Check A — kept values must all be present in available
        avail_copy = available[:]
        for k in kept:
            if k in avail_copy:
                avail_copy.remove(k)
            else:
                errors.append(
                    f"{prefix}: step {step_idx} kept '{k}' but it is not in "
                    f"available='{m.group(1)}'"
                )

        # Check B — engine always selects ALL dice of ONE face value per action
        if len(set(kept)) != 1:
            errors.append(
                f"{prefix}: step {step_idx} kept mixed faces {kept} "
                f"(engine invariant: one face value per selection)"
            )

        # Check C — that face value must not have been committed earlier
        kept_face = kept[0]
        if kept_face in committed:
            errors.append(
                f"{prefix}: step {step_idx} re-committed '{kept_face}' "
                f"(already in committed={committed})"
            )

        committed.extend(kept)

    # ── 3. Terminal checks ──────────────────────────────────────────────────
    if terminal is None or terminal == ":F":
        pass  # fail / truncated — no piece-score constraint to enforce

    elif terminal.startswith((":T", ":S")):
        try:
            piece_num = int(terminal[2:])
        except ValueError:
            errors.append(f"{prefix}: unparseable piece number in '{terminal}'")
            piece_num = None

        if piece_num is not None:
            if not (21 <= piece_num <= 36):
                errors.append(
                    f"{prefix}: piece {piece_num} outside valid range [21, 36]"
                )
            if "w" not in committed:
                errors.append(
                    f"{prefix}: took piece {piece_num} without a worm — "
                    f"committed={committed}"
                )
            total = _score(committed)
            if total < piece_num:
                errors.append(
                    f"{prefix}: score {total} < piece {piece_num} "
                    f"(committed={committed})"
                )
    else:
        errors.append(f"{prefix}: unrecognised terminal '{terminal}'")

    return errors


# ---------------------------------------------------------------------------
# Run checks across all games in the DB
# ---------------------------------------------------------------------------

def check_all_games(matches: list[dict]) -> dict:
    from worms_logger import DatabaseManager
    db = DatabaseManager()
    results = {}
    for match in matches:
        mid    = match["match_id"]
        events = db.get_match_events(mid)
        errs   = []
        for ev in events:
            errs.extend(check_turn(mid, ev["turn_number"], ev["turn_string"]))
        results[mid] = {
            "turns":  len(events),
            "status": match["status"],
            "winner": match["winner"],
            "errors": errs,
        }
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from worms_logger import DatabaseManager
    db = DatabaseManager()

    matches = db.get_all_matches()
    print(f"DB contains {len(matches)} match(es).")

    if not matches:
        print("No games recorded yet — run training with DATABASE_URL set and try again.")
        sys.exit(0)

    print("Running logic checks …\n")
    results = check_all_games(matches)

    total_turns  = sum(r["turns"]        for r in results.values())
    total_errors = sum(len(r["errors"])  for r in results.values())

    print(f"{'─' * 62}")
    print(f"  Checked {len(results)} game(s)   {total_turns} turn(s) total")
    print(f"{'─' * 62}")

    if total_errors == 0:
        print("  ✓  No logic violations found across all turns.")
    else:
        print(f"  ✗  {total_errors} violation(s) found:\n")
        for mid, r in results.items():
            if r["errors"]:
                print(f"  match {mid[:8]}… ({r['turns']} turns, winner={r['winner']}):")
                for e in r["errors"]:
                    print(f"    ! {e}")

    print(f"\n  Per-game summary:")
    print(f"  {'match':<10} {'turns':>5}  {'status':<10}  {'winner':>6}  result")
    print(f"  {'─'*54}")
    for mid, r in results.items():
        status = "OK" if not r["errors"] else f"{len(r['errors'])} error(s)"
        print(
            f"  {mid[:8]}…  {r['turns']:5d}  {r['status']:<10}  "
            f"{str(r['winner']):>6}  {status}"
        )
