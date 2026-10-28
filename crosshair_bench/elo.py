#!/usr/bin/env python3
"""
Anchored Bradley-Terry / BayesElo rating and move-quality aggregator for Crosshair-Bench.

Ingests both LLM-vs-Engine and LLM-vs-LLM matches from `results/matches/*.json`
(and optionally legacy games), anchoring `az-puct-1` at 1200 Elo and computing:
- Win/Loss record & Anchored Elo
- Mean Win-Probability Loss (MWPL) against the `az-puct-400` shadow oracle
- Blunder Rate (ΔWinProb <= -0.10)
- Top-1 and Top-3 Agreement with `az-puct-400`
- Illegal Move Retries, Latency/Turn, Reasoning Tokens/Turn, and Cost/Game ($)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATCHES_DIR = REPO_ROOT / "results" / "matches"
DEFAULT_LEADERBOARD_OUT = REPO_ROOT / "results" / "leaderboard.json"


def _extract_participants(doc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Return `(red_info, blue_info)` for any match document (v1.0 multi-player schema
    or legacy single-LLM schema).
    """
    if "red_player" in doc and "blue_player" in doc:
        rp = doc["red_player"]
        bp = doc["blue_player"]
        red_key = (
            f"{rp['label']} ({rp['prompt_mode']})"
            if rp.get("actor_type") == "llm" and rp.get("prompt_mode")
            else f"engine:{rp['label']}"
        )
        blue_key = (
            f"{bp['label']} ({bp['prompt_mode']})"
            if bp.get("actor_type") == "llm" and bp.get("prompt_mode")
            else f"engine:{bp['label']}"
        )
        return (
            {
                "key": red_key,
                "label": rp["label"],
                "actor_type": rp["actor_type"],
                "prompt_mode": rp.get("prompt_mode"),
                "reasoning_effort": rp.get("reasoning_effort"),
            },
            {
                "key": blue_key,
                "label": bp["label"],
                "actor_type": bp["actor_type"],
                "prompt_mode": bp.get("prompt_mode"),
                "reasoning_effort": bp.get("reasoning_effort"),
            },
        )

    # Legacy single-LLM schema
    llm_side = doc["llm_side"]
    llm_label = doc["model_label"]
    pm = doc.get("prompt_mode", "raw")
    opp = doc["opponent"]
    llm_info = {
        "key": f"{llm_label} ({pm})",
        "label": llm_label,
        "actor_type": "llm",
        "prompt_mode": pm,
        "reasoning_effort": None,
    }
    eng_info = {
        "key": f"engine:{opp}",
        "label": opp,
        "actor_type": "engine",
        "prompt_mode": None,
        "reasoning_effort": None,
    }
    if llm_side == "R":
        return llm_info, eng_info
    return eng_info, llm_info


def fit_anchored_bradley_terry(
    outcomes: list[tuple[str, str, float]],
    anchor_key: str = "engine:az-puct-1",
    anchor_elo: float = 1200.0,
    prior_games: float = 0.25,
    iters: int = 400,
) -> dict[str, float]:
    """
    Fit regularized Bradley-Terry log-strengths via MM / Newton updates and convert
    to standard Elo scale (`400 * log10(exp(theta))`), anchored at `anchor_key = 1200.0`.
    """
    players_set: set[str] = set()
    for p_a, p_b, _ in outcomes:
        players_set.add(p_a)
        players_set.add(p_b)
    if not players_set:
        return {}

    players = sorted(players_set)
    theta: dict[str, float] = {p: 0.0 for p in players}

    # Add weak virtual draws against the anchor (or mean field) so 100% / 0% records stay finite
    virtual_ref = anchor_key if anchor_key in players_set else players[0]
    augmented = list(outcomes)
    for p in players:
        if p != virtual_ref:
            augmented.append((p, virtual_ref, 0.5 * prior_games))
            augmented.append((virtual_ref, p, 0.5 * prior_games))

    for _ in range(iters):
        grad: dict[str, float] = {p: 0.0 for p in players}
        hess: dict[str, float] = {p: 1e-6 for p in players}
        for p_a, p_b, score_a in augmented:
            # Weight can be < 1 for prior games; for real games score_a is in {0.0, 1.0}
            w = prior_games if (0.0 < score_a < 1.0 and score_a != 0.5) else 1.0
            actual_a = score_a if w == 1.0 else 0.5 * w
            diff = max(-20.0, min(20.0, theta[p_a] - theta[p_b]))
            pred_a = 1.0 / (1.0 + math.exp(-diff))
            err = actual_a - w * pred_a
            info = w * pred_a * (1.0 - pred_a)
            grad[p_a] += err
            grad[p_b] -= err
            hess[p_a] += info
            hess[p_b] += info

        max_step = 0.0
        for p in players:
            step = 0.5 * grad[p] / max(0.5, hess[p])
            theta[p] += step
            max_step = max(max_step, abs(step))

        ref_val = theta.get(anchor_key, sum(theta.values()) / len(players))
        for p in players:
            theta[p] -= ref_val
        if max_step < 1e-9:
            break

    scale = 400.0 / math.log(10.0)
    anchor_offset = theta.get(anchor_key, 0.0)
    return {
        p: round(anchor_elo + scale * (theta[p] - anchor_offset), 1)
        for p in players
    }


def compute_leaderboard(match_dirs: list[Path]) -> dict[str, Any]:
    seen_ids: set[str] = set()
    docs: list[dict[str, Any]] = []
    for d in match_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if doc.get("result") in (None, "ongoing"):
                continue
            mid = doc.get("match_id") or doc.get("game_id") or f.stem
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            docs.append(doc)

    outcomes: list[tuple[str, str, float]] = []
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "key": "",
            "label": "",
            "actor_type": "",
            "prompt_mode": None,
            "reasoning_effort": None,
            "games": 0,
            "wins": 0,
            "losses": 0,
            "turns": 0,
            "eval_turns": 0,
            "top1": 0,
            "top3": 0,
            "policy_top1": 0,
            "delta_sum": 0.0,
            "loss_sum": 0.0,
            "blunders": 0,
            "illegals": 0,
            "latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cost_usd": 0.0,
        }
    )

    for doc in docs:
        red_info, blue_info = _extract_participants(doc)
        winner = doc.get("winner")
        if winner not in ("R", "B"):
            continue

        score_r = 1.0 if winner == "R" else 0.0
        outcomes.append((red_info["key"], blue_info["key"], score_r))

        for side, p_info in (("R", red_info), ("B", blue_info)):
            st = stats[p_info["key"]]
            st["key"] = p_info["key"]
            st["label"] = p_info["label"]
            st["actor_type"] = p_info["actor_type"]
            st["prompt_mode"] = p_info["prompt_mode"]
            st["reasoning_effort"] = p_info["reasoning_effort"]
            st["games"] += 1
            if winner == side:
                st["wins"] += 1
            else:
                st["losses"] += 1

            side_turns = [t for t in doc.get("turns", []) if t.get("side") == side]
            st["turns"] += len(side_turns)
            for t in side_turns:
                ev = t.get("eval")
                if ev:
                    st["eval_turns"] += 1
                    if ev.get("search_rank") == 1:
                        st["top1"] += 1
                    if ev.get("search_rank", 99) <= 3:
                        st["top3"] += 1
                    if ev.get("policy_rank") == 1:
                        st["policy_top1"] += 1
                    d = float(ev.get("delta_win_prob", 0.0))
                    st["delta_sum"] += d
                    st["loss_sum"] += max(0.0, -d)
                    if d <= -0.10:
                        st["blunders"] += 1

                tel = t.get("telemetry") or {}
                st["latency_ms"] += float(tel.get("latency_ms", 0.0))
                st["prompt_tokens"] += int(tel.get("prompt_tokens", 0))
                st["completion_tokens"] += int(tel.get("completion_tokens", 0))
                st["reasoning_tokens"] += int(tel.get("reasoning_tokens", 0))
                st["cost_usd"] += float(tel.get("cost_usd", 0.0))
                st["illegals"] += int(tel.get("illegal_retries", 0))

            # Fallback for legacy docs without per-turn illegal_retries
            if "per_side_telemetry" not in doc and p_info["actor_type"] == "llm":
                st["illegals"] += int(doc.get("illegal_attempts", 0))

    elos = fit_anchored_bradley_terry(outcomes)

    rows: list[dict[str, Any]] = []
    for key, st in stats.items():
        g = max(1, st["games"])
        et = max(1, st["eval_turns"])
        t_cnt = max(1, st["turns"])
        has_eval = st["eval_turns"] > 0
        rows.append(
            {
                "key": key,
                "label": st["label"],
                "actor_type": st["actor_type"],
                "prompt_mode": st["prompt_mode"],
                "reasoning_effort": st["reasoning_effort"],
                "elo": elos.get(key, 1200.0),
                "games": st["games"],
                "wins": st["wins"],
                "losses": st["losses"],
                "win_rate": round(st["wins"] / g, 4),
                "turns": st["turns"],
                "eval_turns": st["eval_turns"],
                "top1_rate": round(st["top1"] / et, 4) if has_eval else None,
                "top3_rate": round(st["top3"] / et, 4) if has_eval else None,
                "policy_top1_rate": round(st["policy_top1"] / et, 4) if has_eval else None,
                "avg_delta_win_prob": round(st["delta_sum"] / et, 4) if has_eval else None,
                "mwpl": round(st["loss_sum"] / et, 4) if has_eval else None,
                "blunders": st["blunders"],
                "blunder_rate": round(st["blunders"] / et, 4) if has_eval else None,
                "illegal_attempts": st["illegals"],
                "avg_latency_ms": round(st["latency_ms"] / t_cnt, 1),
                "avg_tokens_per_turn": round(
                    (st["prompt_tokens"] + st["completion_tokens"]) / t_cnt, 1
                ),
                "avg_reasoning_tokens_per_turn": round(
                    st["reasoning_tokens"] / t_cnt, 1
                ),
                "total_cost_usd": round(st["cost_usd"], 5),
                "cost_per_game_usd": round(st["cost_usd"] / g, 5),
            }
        )

    rows.sort(key=lambda r: (r["elo"], r["win_rate"]), reverse=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_matches": len(docs),
        "anchor": {"key": "engine:az-puct-1", "elo": 1200.0},
        "participants": rows,
    }


def print_leaderboard(summary: dict[str, Any]) -> None:
    print(
        f"=== Crosshair-Bench Anchored Leaderboard ({summary['total_matches']} matches, "
        f"anchor: engine:az-puct-1 = 1200 Elo) ==="
    )
    header = (
        f"{'Participant':<34} {'Elo':>6} {'W-L':>7} {'Win%':>6} "
        f"{'Top1%':>6} {'Top3%':>6} {'MWPL':>7} {'Blund%':>7} "
        f"{'Ill':>4} {'AvgMs':>7} {'ReasTok':>8} {'Cost/G($)':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in summary["participants"]:
        wl = f"{r['wins']}-{r['losses']}"
        top1_s = f"{100 * r['top1_rate']:>5.1f}%" if r["top1_rate"] is not None else "    - "
        top3_s = f"{100 * r['top3_rate']:>5.1f}%" if r["top3_rate"] is not None else "    - "
        mwpl_s = f"{r['mwpl']:>7.4f}" if r["mwpl"] is not None else "      -"
        blund_s = (
            f"{100 * r['blunder_rate']:>6.1f}%"
            if r["blunder_rate"] is not None
            else "     - "
        )
        ms_s = f"{r['avg_latency_ms']:>6.0f}ms" if r["avg_latency_ms"] > 0 else "      -"
        rtok_s = (
            f"{r['avg_reasoning_tokens_per_turn']:>8.0f}"
            if r["avg_reasoning_tokens_per_turn"] > 0
            else "       -"
        )
        cost_s = (
            f"${r['cost_per_game_usd']:>8.4f}"
            if r["cost_per_game_usd"] > 0
            else "        -"
        )
        print(
            f"{r['key']:<34} {r['elo']:>6.1f} {wl:>7} {100 * r['win_rate']:>5.1f}% "
            f"{top1_s} {top3_s} {mwpl_s} {blund_s} "
            f"{r['illegal_attempts']:>4} {ms_s} {rtok_s} {cost_s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Anchored Elo and move-quality leaderboard for Crosshair-Bench."
    )
    parser.add_argument(
        "--matches-dir",
        default=str(DEFAULT_MATCHES_DIR),
        help="Comma-separated match directories to aggregate",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_LEADERBOARD_OUT),
        help="Output path for leaderboard JSON",
    )
    args = parser.parse_args()

    dirs = [Path(p.strip()) for p in args.matches_dir.split(",") if p.strip()]
    summary = compute_leaderboard(dirs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_leaderboard(summary)


if __name__ == "__main__":
    main()
