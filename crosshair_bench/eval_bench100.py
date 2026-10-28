#!/usr/bin/env python3
"""
Evaluates an OpenRouter LLM (or local `ai-crosshair` engine config) on the
Crosshair-Bench-100 single-turn tactical position suite.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .build_bench100 import DEFAULT_BENCH_OUT
from .engine import ENGINE_SPECS, choose_engine_move, query_engine
from .openrouter_client import (
    call_openrouter_chat,
    load_openrouter_api_key,
    model_short_label,
    parse_model_spec,
)
from .prompts import PROMPT_MODES, build_turn_prompt, extract_move_and_thoughts


async def evaluate_suite(
    *,
    player_spec: str,
    prompt_mode: str = "raw",
    reasoning_effort: str | None = None,
    bench_path: Path = DEFAULT_BENCH_OUT,
    limit: int | None = None,
    concurrency: int = 4,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    suite = json.loads(bench_path.read_text(encoding="utf-8"))
    positions = suite["positions"]
    if limit is not None:
        positions = positions[:limit]

    is_engine = player_spec.startswith("engine:") or player_spec in ENGINE_SPECS
    eng_key = player_spec.split("engine:", 1)[-1] if is_engine else None
    model_id, parsed_effort = (
        (None, None) if is_engine else parse_model_spec(player_spec)
    )
    eff_effort = reasoning_effort or parsed_effort
    api_key = "" if is_engine else load_openrouter_api_key()
    sem = asyncio.Semaphore(concurrency)

    async def _eval_one(pos: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            moves = pos["moves"]
            state = await asyncio.to_thread(query_engine, moves)
            legal = state["legalMoves"]
            if is_engine:
                assert eng_key is not None
                chosen, _ = await asyncio.to_thread(
                    choose_engine_move, moves, eng_key, 42
                )
                thought = ""
                cost_usd = 0.0
                latency_ms = 0.0
            else:
                assert model_id is not None
                prompt = build_turn_prompt(state, pos["side_to_move"], prompt_mode)
                eff_max = max(max_tokens, 6144) if eff_effort else max_tokens
                resp = await asyncio.to_thread(
                    call_openrouter_chat,
                    api_key=api_key,
                    model_id=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=eff_max,
                    temperature=0.0,
                    reasoning_effort=eff_effort,
                )
                cand, thought, _, _ = extract_move_and_thoughts(
                    resp.get("content", ""), resp.get("native_reasoning")
                )
                chosen = cand if (cand in legal) else None
                cost_usd = float(resp.get("cost_usd", 0.0))
                latency_ms = float(resp.get("latency_ms", 0.0))

            mv_info = pos["move_values"].get(chosen or "", None)
            if mv_info is None:
                rank = len(legal)
                loss = float(pos["win_prob_best"])
                is_illegal = True
            else:
                rank = int(mv_info["rank"])
                loss = float(mv_info["win_prob_loss"])
                is_illegal = False

            return {
                "id": pos["id"],
                "category": pos["category"],
                "chosen_move": chosen,
                "az400_best_move": pos["az400_best_move"],
                "top1": chosen == pos["az400_best_move"],
                "top3": rank <= 3,
                "rank": rank,
                "win_prob_loss": round(loss, 4),
                "blunder": loss >= 0.10,
                "illegal": is_illegal,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "thought": thought[:300],
            }

    results = await asyncio.gather(*[_eval_one(p) for p in positions])
    n = max(1, len(results))
    top1 = sum(1 for r in results if r["top1"])
    top3 = sum(1 for r in results if r["top3"])
    mwpl = sum(r["win_prob_loss"] for r in results) / n
    blunders = sum(1 for r in results if r["blunder"])
    illegals = sum(1 for r in results if r["illegal"])
    total_cost = sum(r["cost_usd"] for r in results)

    label = (
        eng_key
        if is_engine
        else f"{model_short_label(model_id or '', eff_effort)} ({prompt_mode})"
    )
    print(
        f"=== Crosshair-Bench-100 Evaluation: {label} ({len(results)} positions) ===\n"
        f"  Top-1 Accuracy: {100 * top1 / n:.1f}% ({top1}/{len(results)})\n"
        f"  Top-3 Accuracy: {100 * top3 / n:.1f}% ({top3}/{len(results)})\n"
        f"  MWPL:           {mwpl:.4f}\n"
        f"  Blunder Rate:   {100 * blunders / n:.1f}%\n"
        f"  Illegal Rate:   {100 * illegals / n:.1f}%\n"
        f"  Total Cost:     ${total_cost:.4f}"
    )
    return {
        "player": label,
        "positions_evaluated": len(results),
        "top1_rate": round(top1 / n, 4),
        "top3_rate": round(top3 / n, 4),
        "mwpl": round(mwpl, 4),
        "blunder_rate": round(blunders / n, 4),
        "illegal_rate": round(illegals / n, 4),
        "total_cost_usd": round(total_cost, 5),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a model or engine on Crosshair-Bench-100."
    )
    parser.add_argument("--player", required=True, help="Model spec or engine:<key>")
    parser.add_argument(
        "--prompt-mode", choices=list(PROMPT_MODES), default="raw"
    )
    parser.add_argument("--effort", default=None)
    parser.add_argument("--bench", default=str(DEFAULT_BENCH_OUT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    asyncio.run(
        evaluate_suite(
            player_spec=args.player,
            prompt_mode=args.prompt_mode,
            reasoning_effort=args.effort,
            bench_path=Path(args.bench),
            limit=args.limit,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
