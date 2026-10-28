"""
Wrapper around the compiled `ai-crosshair` Rust engine binary.

Provides position state queries, batch state evaluation, engine search moves
(`az-puct-1`, `random-uct-50`, `az-puct-50`, `az-puct-400`, `az-puct-2000`, `az-puct-4000`),
and cached `az-puct-400` shadow-oracle move evaluations.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

ENGINE_SPECS: dict[str, tuple[str, int]] = {
    "az-puct-1": ("az-puct", 1),
    "az-puct-50": ("az-puct", 50),
    "az-puct-400": ("az-puct", 400),
    "az-puct-2000": ("az-puct", 2000),
    "az-puct-4000": ("az-puct", 4000),
    "random-uct-50": ("random-uct", 50),
    "random-uct-400": ("random-uct", 400),
}

ENGINE_CONFIGS = ENGINE_SPECS

ENGINE_SHORT_LABELS: dict[str, str] = {
    "az-puct-1": "az1",
    "az-puct-50": "az50",
    "az-puct-400": "az400",
    "az-puct-2000": "az2000",
    "az-puct-4000": "az4000",
    "random-uct-50": "uct50",
    "random-uct-400": "uct400",
}

_SEARCH_CACHE: dict[tuple[tuple[str, ...], str | None], dict[str, Any]] = {}
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_engine_bin() -> Path:
    """Locate the compiled `ai-crosshair` binary."""
    env_bin = os.environ.get("AI_CROSSHAIR_BIN", "").strip()
    if env_bin:
        p = Path(env_bin).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"AI_CROSSHAIR_BIN points to missing file: {p}")

    which_bin = shutil.which("ai-crosshair")
    if which_bin:
        return Path(which_bin).resolve()

    candidates = [
        REPO_ROOT.parent / "ai-crosshair" / "target" / "release" / "ai-crosshair",
        REPO_ROOT.parent / "private-blog" / "game-ai" / "target" / "release" / "ai-crosshair",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(
        "Could not locate `ai-crosshair` binary. Set AI_CROSSHAIR_BIN, add "
        "`ai-crosshair` to PATH, or build `cargo build --release` in `../ai-crosshair`."
    )


def encode_board(board: list[str | None]) -> str:
    """Encode 49-element board array into a compact 49-character string of 'R', 'B', '.'."""
    return "".join(cell if cell in ("R", "B") else "." for cell in board)


def format_board_ascii(board: list[str | None]) -> str:
    """Format 7x7 board into human/LLM-readable ASCII grid with rank/file labels."""
    rows = ["    a   b   c   d   e   f   g", "  +---+---+---+---+---+---+---+"]
    for r in range(7):
        cells = []
        for c in range(7):
            val = board[r * 7 + c]
            cells.append(val if val is not None else ".")
        row_str = " | ".join(cells)
        rows.append(f"{r + 1} | {row_str} | {r + 1}")
    rows.append("  +---+---+---+---+---+---+---+")
    rows.append("    a   b   c   d   e   f   g")
    return "\n".join(rows)


def query_engine(
    moves: list[str],
    search_cmd: str | None = None,
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Query `ai-crosshair` for a position's state, optionally running a search command first.
    Uses an in-memory cache keyed on `(tuple(moves), search_cmd)` so consecutive turns
    sharing the same position and oracle seed never duplicate work.
    """
    cache_key = (tuple(moves), search_cmd)
    if use_cache and cache_key in _SEARCH_CACHE:
        return _SEARCH_CACHE[cache_key]

    engine_bin = resolve_engine_bin()
    pos_line = (
        "position startpos"
        if not moves
        else f"position startpos moves {' '.join(moves)}"
    )
    lines = [pos_line]
    if search_cmd:
        lines.append(search_cmd)
    lines.append("state")
    payload = "\n".join(lines) + "\n"

    proc = subprocess.run(
        [str(engine_bin)],
        input=payload,
        text=True,
        capture_output=True,
        check=True,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("state "):
            state = json.loads(line[len("state ") :])
            if use_cache:
                _SEARCH_CACHE[cache_key] = state
            return state
    raise RuntimeError(f"Engine did not return a state line:\n{proc.stdout}")


def batch_query_states(move_prefixes: list[list[str]]) -> list[dict[str, Any]]:
    """Run a single `ai-crosshair` subprocess to obtain `state` JSON for every move prefix."""
    if not move_prefixes:
        return []
    engine_bin = resolve_engine_bin()
    lines: list[str] = []
    for prefix in move_prefixes:
        if not prefix:
            lines.append("position startpos")
        else:
            lines.append(f"position startpos moves {' '.join(prefix)}")
        lines.append("state")
    payload = "\n".join(lines) + "\n"
    proc = subprocess.run(
        [str(engine_bin)],
        input=payload,
        text=True,
        capture_output=True,
        check=True,
    )
    states: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if line.startswith("state "):
            states.append(json.loads(line[len("state ") :]))
    if len(states) != len(move_prefixes):
        raise RuntimeError(
            f"Expected {len(move_prefixes)} states from engine, got {len(states)}"
        )
    return states


def choose_engine_move(
    moves: list[str],
    engine_key: str,
    game_seed: int,
) -> tuple[str, dict[str, Any]]:
    """
    Compute the engine's move for `engine_key` (e.g. `az-puct-1`, `az-puct-400`, `random-uct-50`).
    Returns `(best_move, decision_dict)`.
    """
    if engine_key not in ENGINE_SPECS:
        raise ValueError(
            f"Unknown engine config '{engine_key}'. Valid: {sorted(ENGINE_SPECS)}"
        )
    strategy, sims = ENGINE_SPECS[engine_key]
    ply = len(moves)
    # Use a deterministic seed distinct from the oracle seed unless engine_key == az-puct-400
    seed = game_seed + ply * 31 if engine_key == "az-puct-400" else game_seed + ply * 17 + sims
    snap = query_engine(
        moves,
        f"search strategy {strategy} simulations {sims} seed {seed}",
    )
    dec = snap["decision"]
    return dec["bestMove"], dec


def evaluate_move_oracle(
    moves_before: list[str],
    chosen_move: str,
    game_seed: int,
    oracle_sims: int = 4000,
) -> dict[str, Any]:
    """
    Evaluate `chosen_move` played at `moves_before` using the `az-puct` shadow oracle
    (default 4,000 simulations).

    Uses a deterministic seed per ply (`game_seed + ply * 31`) so `snap_after` on ply `t`
    is an exact cache hit for `snap_before` on ply `t + 1`, guaranteeing both 2x speedup
    and exact win-probability conservation (`win_prob_before(t+1) == 1 - win_prob_after(t)`).
    """
    ply_before = len(moves_before)
    seed_before = game_seed + ply_before * 31
    snap_before = query_engine(
        moves_before,
        f"search strategy az-puct simulations {oracle_sims} seed {seed_before}",
    )
    side_to_move = snap_before["sideToMove"]
    dec_before = snap_before["decision"]
    root_moves = dec_before["rootMoves"]

    by_prior = sorted(
        root_moves,
        key=lambda m: (m["prior"], m["expectedScore"]),
        reverse=True,
    )
    by_visits = sorted(
        root_moves,
        key=lambda m: (m["visits"], m["expectedScore"], m["prior"]),
        reverse=True,
    )

    policy_rank = next(
        (i + 1 for i, m in enumerate(by_prior) if m["move"] == chosen_move),
        len(root_moves),
    )
    search_rank = next(
        (i + 1 for i, m in enumerate(by_visits) if m["move"] == chosen_move),
        len(root_moves),
    )
    chosen_stats = next((m for m in root_moves if m["move"] == chosen_move), None)
    win_prob_before = float(dec_before["expectedScore"])

    moves_after = list(moves_before) + [chosen_move]
    snap_after_state = query_engine(moves_after)
    if snap_after_state["result"] == "win":
        win_prob_after = 1.0
    else:
        ply_after = len(moves_after)
        seed_after = game_seed + ply_after * 31
        snap_after_search = query_engine(
            moves_after,
            f"search strategy az-puct simulations {oracle_sims} seed {seed_after}",
        )
        # Opponent's expectedScore is from the opponent's perspective
        win_prob_after = 1.0 - float(snap_after_search["decision"]["expectedScore"])

    delta_win_prob = win_prob_after - win_prob_before
    win_prob_loss = max(0.0, -delta_win_prob)

    if side_to_move == "R":
        red_before = win_prob_before
        red_after = win_prob_after
    else:
        red_before = 1.0 - win_prob_before
        red_after = 1.0 - win_prob_after

    top_candidates = [
        {
            "move": m["move"],
            "visits": int(m["visits"]),
            "prior": round(float(m["prior"]), 4),
            "expectedScore": round(float(m["expectedScore"]), 4),
        }
        for m in by_visits[:5]
    ]

    return {
        "az400_best_move": dec_before["bestMove"],
        "az1_top_prior_move": by_prior[0]["move"] if by_prior else None,
        "chosen_move": chosen_move,
        "chosen_prior": round(float(chosen_stats["prior"]), 4) if chosen_stats else 0.0,
        "chosen_visits_400": int(chosen_stats["visits"]) if chosen_stats else 0,
        "chosen_expected_score": (
            round(float(chosen_stats["expectedScore"]), 4) if chosen_stats else 0.0
        ),
        "policy_rank": int(policy_rank),
        "search_rank": int(search_rank),
        "num_legal": len(root_moves),
        "win_prob_before": round(win_prob_before, 4),
        "win_prob_after": round(win_prob_after, 4),
        "delta_win_prob": round(delta_win_prob, 4),
        "win_prob_loss": round(win_prob_loss, 4),
        "red_win_prob_before": round(red_before, 4),
        "red_win_prob_after": round(red_after, 4),
        "network_value_before": (
            round(float(dec_before["networkValue"]), 4)
            if dec_before.get("networkValue") is not None
            else None
        ),
        "top_candidates": top_candidates,
    }
