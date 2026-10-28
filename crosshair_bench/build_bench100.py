#!/usr/bin/env python3
"""
Builds the Crosshair-Bench-100 single-turn tactical suite (`benchmarks/crosshair_bench_100.json`).

Mines high-cliff tactical positions (`best_win_prob - median_win_prob >= 0.20`) from
completed arena games and deterministic `ai-crosshair` self-play trajectories,
evaluating every legal move at each candidate position with `az-puct-400` so that
single-turn model evaluation is instantaneous.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import choose_engine_move, encode_board, query_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH_OUT = REPO_ROOT / "benchmarks" / "crosshair_bench_100.json"
DEFAULT_SOURCE_DIRS = [
    REPO_ROOT / "results" / "matches",
    Path("/home/nick/projects/private-blog/notes/llm-arena/games"),
]


def _classify_position(
    state: dict[str, Any],
    best_move: str,
    best_score: float,
    median_score: float,
) -> str:
    previews_by_move = {p["move"]: p for p in state.get("movePreviews", [])}
    best_prev = previews_by_move.get(best_move, {})
    if best_prev.get("immediateWin") or best_score >= 0.96:
        return "immediate_win_or_save"
    flips = best_prev.get("flips", [])
    if len(flips) >= 3:
        return "cascade_chain_reaction"
    if len(flips) == 2:
        return "crosscut_2x2_tactic"
    if best_score - median_score >= 0.35:
        return "self_routing_trap"
    return "positional_routing"


def evaluate_position_candidates(
    moves: list[str],
    source_id: str,
    seed: int = 42,
    sims: int = 400,
) -> dict[str, Any] | None:
    """
    Run `az-puct` with `sims` simulations on `moves` and package a Bench-100 item
    if the position has at least 4 legal moves and a sharp tactical cliff.
    """
    ply = len(moves)
    pos_seed = seed + ply * 31
    snap = query_engine(
        moves,
        f"search strategy az-puct simulations {sims} seed {pos_seed}",
    )
    if snap["result"] != "ongoing":
        return None
    legal = snap["legalMoves"]
    if len(legal) < 4:
        return None

    dec = snap["decision"]
    root_moves = dec["rootMoves"]
    by_visits = sorted(
        root_moves,
        key=lambda m: (m["visits"], m["expectedScore"], m["prior"]),
        reverse=True,
    )
    by_prior = sorted(
        root_moves,
        key=lambda m: (m["prior"], m["expectedScore"]),
        reverse=True,
    )

    scores = [float(m["expectedScore"]) for m in root_moves]
    best_move = by_visits[0]["move"]
    best_score = float(by_visits[0]["expectedScore"])
    median_score = float(statistics.median(scores))
    cliff_spread = best_score - median_score
    if cliff_spread < 0.14:
        return None

    previews_by_move = {p["move"]: p for p in snap.get("movePreviews", [])}
    move_values: dict[str, dict[str, Any]] = {}
    for rank_idx, m in enumerate(by_visits, start=1):
        mv = m["move"]
        sc = float(m["expectedScore"])
        prev = previews_by_move.get(mv, {})
        move_values[mv] = {
            "rank": rank_idx,
            "expected_score": round(sc, 4),
            "win_prob_loss": round(max(0.0, best_score - sc), 4),
            "visits": int(m["visits"]),
            "prior": round(float(m["prior"]), 4),
            "flips": prev.get("flips", []),
            "immediate_win": bool(prev.get("immediateWin", False)),
        }

    category = _classify_position(snap, best_move, best_score, median_score)
    return {
        "category": category,
        "source_game": source_id,
        "ply": ply,
        "side_to_move": snap["sideToMove"],
        "moves": list(moves),
        "board": encode_board(snap["board"]),
        "active_row": snap["activeRow"],
        "active_col": snap["activeCol"],
        "wildcard": bool(snap["wildcard"]),
        "legal_moves": legal,
        "az400_best_move": best_move,
        "az1_top_prior_move": by_prior[0]["move"],
        "policy_agrees_with_search": best_move == by_prior[0]["move"],
        "win_prob_best": round(best_score, 4),
        "win_prob_median": round(median_score, 4),
        "cliff_spread": round(cliff_spread, 4),
        "move_values": move_values,
    }


def build_suite(target_count: int = 30) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    seen_boards: set[tuple[str, int | None, int | None, str]] = set()

    # 1. Mine positions from completed games where blunders or sharp cliffs occurred
    for d in DEFAULT_SOURCE_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            moves = doc.get("moves") or []
            turns = doc.get("turns") or []
            gid = doc.get("match_id") or doc.get("game_id") or f.stem
            seed = int(doc.get("seed", 42))

            # Prioritize turns with large negative delta_win_prob or midgame turns
            interesting_plies: list[int] = []
            for t in turns:
                ev = t.get("eval")
                if ev and float(ev.get("delta_win_prob", 0.0)) <= -0.06:
                    interesting_plies.append(int(t["ply"]))
            for p in range(6, min(len(moves), 42), 3):
                if p not in interesting_plies:
                    interesting_plies.append(p)

            for ply in interesting_plies:
                if ply >= len(moves):
                    continue
                prefix = moves[:ply]
                item = evaluate_position_candidates(prefix, gid, seed=seed)
                if item is None:
                    continue
                sig = (
                    item["board"],
                    item["active_row"],
                    item["active_col"],
                    item["side_to_move"],
                )
                if sig in seen_boards:
                    continue
                seen_boards.add(sig)
                candidates.append(item)
                if len(candidates) >= target_count * 2:
                    break

    # 2. If needed, supplement with deterministic mixed-strength self-play trajectories
    sp_idx = 0
    while len(candidates) < target_count and sp_idx < 15:
        sp_idx += 1
        sp_seed = 1000 + sp_idx * 97
        moves: list[str] = []
        state = query_engine(moves)
        while state["result"] == "ongoing" and len(moves) < 42:
            eng = "az-puct-400" if state["ply"] % 2 == 0 else "random-uct-50"
            mv, _ = choose_engine_move(moves, eng, sp_seed)
            if 10 <= state["ply"] <= 36 and state["ply"] % 3 == 0:
                item = evaluate_position_candidates(
                    moves, f"selfplay_s{sp_seed}", seed=sp_seed
                )
                if item is not None:
                    sig = (
                        item["board"],
                        item["active_row"],
                        item["active_col"],
                        item["side_to_move"],
                    )
                    if sig not in seen_boards:
                        seen_boards.add(sig)
                        candidates.append(item)
            moves.append(mv)
            state = query_engine(moves)

    # Sort by cliff_spread descending and select top `target_count`
    candidates.sort(
        key=lambda x: (not x["policy_agrees_with_search"], x["cliff_spread"]),
        reverse=True,
    )
    selected = candidates[:target_count]
    for idx, item in enumerate(selected, start=1):
        item["id"] = f"cb100_{idx:03d}"

    cat_counts: dict[str, int] = {}
    for item in selected:
        cat = item["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    return {
        "schema_version": "1.0",
        "benchmark": "Crosshair-Bench-100",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_positions": len(selected),
        "category_counts": cat_counts,
        "positions": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Crosshair-Bench-100 tactical position suite."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=30,
        help="Number of high-cliff positions to include (default: 30, expandable to 100)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_BENCH_OUT),
        help="Output JSON file path",
    )
    args = parser.parse_args()

    suite = build_suite(args.target)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(
        f"Built {suite['total_positions']} high-cliff tactical positions -> {out_path}\n"
        f"Category breakdown: {suite['category_counts']}"
    )


if __name__ == "__main__":
    main()
