#!/usr/bin/env python3
import argparse
import collections
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Iterable, List, Optional

ITERATION_RE = re.compile(r"^---\s*Iteration\s+(\d+)\s*---")
ARENA_RE = re.compile(
    r"Arena results:\s*challenger wins=(\d+),\s*baseline wins=(\d+),\s*draws=(\d+),\s*win_rate=([0-9.]+)\s*\((\d+)\s*decisive\)",
    re.IGNORECASE,
)
SAVE_ITER_RE = re.compile(r"Successfully saved .*iteration_(\d+)_(\d{8})_(\d{6})\.pkl")
HISTORY_FILE_RE = re.compile(r"iteration_(\d+)_(\d{8})_(\d{6})\.pkl$")


@dataclass
class ArenaSummary:
    iteration: int
    challenger_wins: int
    baseline_wins: int
    draws: int
    win_rate: float
    decisive: int
    total_games: int
    saved_time: Optional[datetime] = None

    @property
    def average_time_seconds(self) -> Optional[float]:
        return None


def parse_arena_results(line: str):
    match = ARENA_RE.search(line)
    if not match:
        return None
    challenger, baseline, draws, win_rate, decisive = match.groups()
    challenger, baseline, draws, decisive = map(int, (challenger, baseline, draws, decisive))
    win_rate = float(win_rate)
    return {
        "challenger_wins": challenger,
        "baseline_wins": baseline,
        "draws": draws,
        "win_rate": win_rate,
        "decisive": decisive,
        "total_games": challenger + baseline + draws,
    }


def parse_iteration_save_time(line: str):
    match = SAVE_ITER_RE.search(line)
    if not match:
        return None
    iteration_str, date_str, time_str = match.groups()
    iteration = int(iteration_str)
    saved_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    return iteration, saved_time


def parse_history_filename(filename: str):
    match = HISTORY_FILE_RE.match(filename)
    if not match:
        return None
    iteration_str, date_str, time_str = match.groups()
    iteration = int(iteration_str)
    saved_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    return iteration, saved_time


def find_iteration_saved_time(iteration: int, history_dir: Path) -> Optional[datetime]:
    if not history_dir.exists() or not history_dir.is_dir():
        return None
    best_time = None
    for child in history_dir.iterdir():
        if not child.is_file():
            continue
        parsed = parse_history_filename(child.name)
        if parsed is None:
            continue
        file_iteration, file_time = parsed
        if file_iteration != iteration:
            continue
        if best_time is None or file_time > best_time:
            best_time = file_time
    return best_time


def summarize_stats(records: List[ArenaSummary], windows: Iterable[int]) -> List[str]:
    lines = []
    if not records:
        return ["No arena results found yet."]

    for window in windows:
        if window <= 0:
            continue
        selected = records[-window:]
        if not selected:
            continue
        total_challenger = sum(r.challenger_wins for r in selected)
        total_baseline = sum(r.baseline_wins for r in selected)
        total_draws = sum(r.draws for r in selected)
        total_decisive = sum(r.decisive for r in selected)
        total_games = sum(r.total_games for r in selected)
        average_win_rate = (
            total_challenger / total_decisive if total_decisive > 0 else 0.0
        )
        average_iteration_time = compute_average_iteration_time(selected)

        lines.append(f"Last {len(selected)} arena iterations:")
        lines.append(
            f"  challenger wins={total_challenger}, baseline wins={total_baseline}, draws={total_draws},"
        )
        lines.append(
            f"  total games={total_games}, total decisive={total_decisive}, average win_rate={average_win_rate:.3f}"
        )
        if average_iteration_time is not None:
            lines.append(
                f"  avg iteration completion interval: {average_iteration_time:.1f} seconds"
            )
        else:
            lines.append("  avg iteration completion interval: not available")
        lines.append("")
    return lines


def compute_average_iteration_time(records: List[ArenaSummary]) -> Optional[float]:
    times = [r.saved_time for r in records if r.saved_time is not None]
    if len(times) < 2:
        return None
    deltas = [
        (times[i] - times[i - 1]).total_seconds()
        for i in range(1, len(times))
        if times[i] > times[i - 1]
    ]
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def print_iteration_record(record: ArenaSummary):
    print(f"--- Iteration {record.iteration} arena summary ---")
    print(f" challenger wins: {record.challenger_wins}")
    print(f" baseline wins:   {record.baseline_wins}")
    print(f" draws:           {record.draws}")
    print(f" win_rate:        {record.win_rate:.3f}")
    print(f" decisive:        {record.decisive}")
    print(f" total games:     {record.total_games}")
    if record.saved_time:
        print(f" completed at:    {record.saved_time.isoformat(sep=' ')}")
    print()


def read_new_lines(path: str, last_position: int):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(last_position)
            chunk = f.read()
            current_position = f.tell()
            return chunk.splitlines(), current_position
    except FileNotFoundError:
        return [], last_position


def monitor_log(path: str, interval: float, follow: bool, windows: List[int], history_dir: Path):
    records: Deque[ArenaSummary] = collections.deque(maxlen=max(windows) * 2)
    current_iteration: Optional[int] = None
    last_position = 0

    print(f"Watching {path}")
    print(
        f" mode={'continuous' if follow else f'poll every {interval:.0f}s'}, "
        f"summary windows={windows}, history_dir={history_dir}"
    )
    print("Press Ctrl+C to stop.\n")

    while True:
        lines, current_position = read_new_lines(path, last_position)
        if lines:
            for line in lines:
                match_iter = ITERATION_RE.match(line.strip())
                if match_iter:
                    current_iteration = int(match_iter.group(1))
                    continue

                arena_result = parse_arena_results(line)
                if arena_result and current_iteration is not None:
                    record = ArenaSummary(
                        iteration=current_iteration,
                        saved_time=None,
                        **arena_result,
                    )
                    if history_dir is not None:
                        saved_time = find_iteration_saved_time(current_iteration, history_dir)
                        if saved_time is not None:
                            record.saved_time = saved_time
                    records.append(record)
                    print_iteration_record(record)
                    print("Summary:")
                    for summary_line in summarize_stats(list(records), windows):
                        print(summary_line)
                    continue

                save_info = parse_iteration_save_time(line)
                if save_info:
                    iteration_num, saved_time = save_info
                    for record in reversed(records):
                        if record.iteration == iteration_num:
                            record.saved_time = saved_time
                            break

        if history_dir is not None:
            for record in records:
                if record.saved_time is None:
                    maybe_time = find_iteration_saved_time(record.iteration, history_dir)
                    if maybe_time is not None:
                        record.saved_time = maybe_time

        file_size = os.path.getsize(path) if os.path.exists(path) else 0
        if file_size < current_position:
            current_position = 0
        last_position = current_position

        if not follow:
            time.sleep(interval)
            continue

        time.sleep(interval)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Monitor training log arena results and summarize challenger win stats."
    )
    parser.add_argument("log_file", help="Path to the training log file to watch.")
    parser.add_argument(
        "--interval",
        type=float,
        default=600.0,
        help="Polling interval in seconds when not following; or interval for follow mode. Default: 600",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Continuously poll the file for updates instead of waiting 10 minutes between checks.",
    )
    parser.add_argument(
        "--history-dir",
        default="./checkpoints/history",
        help="Path to checkpoint history folder containing iteration_*.pkl files.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="*",
        default=[5, 10, 25],
        help="Summary windows in number of arena iterations. Default: 5 10 25.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.interval <= 0:
        raise SystemExit("Interval must be a positive number.")
    history_dir = Path(args.history_dir)
    monitor_log(
        args.log_file,
        args.interval,
        args.follow,
        sorted(set(args.windows)),
        history_dir,
    )


if __name__ == "__main__":
    main()
