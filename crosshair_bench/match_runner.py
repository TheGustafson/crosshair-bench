#!/usr/bin/env python3
"""
Match runner for Crosshair-Bench.

Supports:
- LLM vs Engine (`google/gemini-3.8-flash` vs `engine:az-puct-1`)
- LLM vs LLM (`deepseek/deepseek-v4.1-flash` vs `anthropic/claude-haiku-5.5`)
- Engine vs Engine (`engine:az-puct-400` vs `engine:az-puct-1`)

Every turn (Red and Blue, LLM and Engine) is evaluated by the `az-puct-400`
shadow oracle before and after the move, recording full board state, flips,
active crosshair constraints, top-5 MCTS candidates, thoughts/reasoning traces,
and OpenRouter telemetry into a self-contained match JSON file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import (
    ENGINE_SHORT_LABELS,
    ENGINE_SPECS,
    choose_engine_move,
    encode_board,
    evaluate_move_oracle,
    query_engine,
)
from .openrouter_client import (
    DEFAULT_KEY_FILE,
    call_openrouter_chat,
    load_openrouter_api_key,
    model_short_label,
    parse_model_spec,
)
from .prompts import (
    PROMPT_MODES,
    build_retry_prompt,
    build_turn_prompt,
    extract_move_and_thoughts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATCHES_DIR = REPO_ROOT / "results" / "matches"


@dataclass
class PlayerSpec:
    actor_type: str  # "llm" or "engine"
    actor_id: str  # e.g. "google/gemini-3.8-flash" or "engine:az-puct-1"
    label: str  # e.g. "gemini-3.8-flash" or "az-puct-1"
    model_id: str | None = None
    engine_key: str | None = None
    prompt_mode: str | None = None
    reasoning_effort: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096

    @classmethod
    def from_spec(
        cls,
        spec: str,
        *,
        prompt_mode: str = "raw",
        reasoning_effort: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        label: str | None = None,
    ) -> PlayerSpec:
        s = spec.strip()
        if s.startswith("engine:") or s in ENGINE_SPECS:
            eng_key = s.split("engine:", 1)[-1].strip()
            if eng_key not in ENGINE_SPECS:
                raise ValueError(
                    f"Unknown engine '{eng_key}'. Valid: {sorted(ENGINE_SPECS)}"
                )
            return cls(
                actor_type="engine",
                actor_id=f"engine:{eng_key}",
                label=label or eng_key,
                model_id=None,
                engine_key=eng_key,
                prompt_mode=None,
                reasoning_effort=None,
                temperature=0.0,
                max_tokens=0,
            )

        if prompt_mode not in PROMPT_MODES:
            raise ValueError(
                f"Invalid prompt_mode '{prompt_mode}'. Valid: {PROMPT_MODES}"
            )
        model_id, parsed_effort = parse_model_spec(s)
        eff_effort = reasoning_effort if reasoning_effort is not None else parsed_effort
        short_lbl = label or model_short_label(model_id, eff_effort)
        actor_id = (
            f"{model_id}@{eff_effort}"
            if (eff_effort and eff_effort != "none")
            else model_id
        )
        return cls(
            actor_type="llm",
            actor_id=actor_id,
            label=short_lbl,
            model_id=model_id,
            engine_key=None,
            prompt_mode=prompt_mode,
            reasoning_effort=eff_effort,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-")


def make_default_match_id(red: PlayerSpec, blue: PlayerSpec, seed: int) -> str:
    r_part = _slugify(red.label)
    if red.prompt_mode:
        r_part = f"{r_part}_{red.prompt_mode}"
    b_part = _slugify(blue.label)
    if blue.prompt_mode:
        b_part = f"{b_part}_{blue.prompt_mode}"
    elif blue.engine_key:
        b_part = ENGINE_SHORT_LABELS.get(blue.engine_key, b_part)
    return f"{r_part}_vs_{b_part}_s{seed}"


def _compute_side_metrics(turns: list[dict[str, Any]], side: str) -> dict[str, Any]:
    side_turns = [t for t in turns if t["side"] == side and t.get("eval")]
    n = len(side_turns)
    if n == 0:
        return {
            "turns": 0,
            "top1_count": 0,
            "top3_count": 0,
            "top1_rate": 0.0,
            "top3_rate": 0.0,
            "policy_top1_rate": 0.0,
            "avg_delta_win_prob": 0.0,
            "mwpl": 0.0,
            "blunder_count": 0,
            "blunder_rate": 0.0,
            "worst_ply": 0,
            "worst_delta": 0.0,
        }

    evals = [t["eval"] for t in side_turns]
    top1 = sum(1 for e in evals if e["search_rank"] == 1)
    top3 = sum(1 for e in evals if e["search_rank"] <= 3)
    pol1 = sum(1 for e in evals if e["policy_rank"] == 1)
    deltas = [float(e["delta_win_prob"]) for e in evals]
    losses = [max(0.0, -d) for d in deltas]
    blunders = sum(1 for d in deltas if d <= -0.10)
    worst_turn = min(side_turns, key=lambda t: float(t["eval"]["delta_win_prob"]))

    return {
        "turns": n,
        "top1_count": top1,
        "top3_count": top3,
        "top1_rate": round(top1 / n, 4),
        "top3_rate": round(top3 / n, 4),
        "policy_top1_rate": round(pol1 / n, 4),
        "avg_delta_win_prob": round(sum(deltas) / n, 4),
        "mwpl": round(sum(losses) / n, 4),
        "blunder_count": blunders,
        "blunder_rate": round(blunders / n, 4),
        "worst_ply": int(worst_turn["ply"]),
        "worst_delta": round(float(worst_turn["eval"]["delta_win_prob"]), 4),
    }


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


async def run_match(
    *,
    red_player: PlayerSpec,
    blue_player: PlayerSpec,
    match_id: str | None = None,
    seed: int = 42,
    opening: str = "",
    oracle_sims: int = 400,
    max_illegal_retries: int = 3,
    overwrite: bool = False,
    matches_dir: Path = DEFAULT_MATCHES_DIR,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Run a complete Crosshair match between `red_player` and `blue_player`.
    Records full board state, `az-puct-400` shadow-oracle evaluations, and telemetry
    on every ply, saving atomically to `matches_dir / <match_id>.json`.
    """
    matches_dir.mkdir(parents=True, exist_ok=True)
    mid = match_id or make_default_match_id(red_player, blue_player, seed)
    match_path = matches_dir / f"{mid}.json"

    if match_path.exists() and not overwrite:
        try:
            existing = json.loads(match_path.read_text(encoding="utf-8"))
            if existing.get("result") and existing.get("result") != "ongoing":
                print(
                    f"[skip] {mid} already completed ({existing['result']}, winner={existing.get('winner')})"
                )
                return existing
        except Exception:
            pass

    needs_api_key = (
        red_player.actor_type == "llm" or blue_player.actor_type == "llm"
    )
    resolved_key = api_key or (load_openrouter_api_key() if needs_api_key else "")

    if red_player.actor_type == "llm" and blue_player.actor_type == "llm":
        match_type = "llm_vs_llm"
        primary_llm_side = "R"
        primary_model_label = red_player.label
        primary_model_id = red_player.actor_id
        primary_prompt_mode = red_player.prompt_mode or "raw"
        opponent_label = f"llm:{blue_player.label}"
    elif red_player.actor_type == "llm" and blue_player.actor_type == "engine":
        match_type = "llm_vs_engine"
        primary_llm_side = "R"
        primary_model_label = red_player.label
        primary_model_id = red_player.actor_id
        primary_prompt_mode = red_player.prompt_mode or "raw"
        opponent_label = blue_player.engine_key or blue_player.label
    elif red_player.actor_type == "engine" and blue_player.actor_type == "llm":
        match_type = "llm_vs_engine"
        primary_llm_side = "B"
        primary_model_label = blue_player.label
        primary_model_id = blue_player.actor_id
        primary_prompt_mode = blue_player.prompt_mode or "raw"
        opponent_label = red_player.engine_key or red_player.label
    else:
        match_type = "engine_vs_engine"
        primary_llm_side = "R"
        primary_model_label = red_player.label
        primary_model_id = red_player.actor_id
        primary_prompt_mode = "raw"
        opponent_label = blue_player.engine_key or blue_player.label

    opening_moves = opening.split() if opening else []
    moves: list[str] = list(opening_moves)

    per_side_telemetry: dict[str, dict[str, Any]] = {
        "R": {
            "total_latency_ms": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_reasoning_tokens": 0,
            "total_cost_usd": 0.0,
            "illegal_attempts": 0,
        },
        "B": {
            "total_latency_ms": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_reasoning_tokens": 0,
            "total_cost_usd": 0.0,
            "illegal_attempts": 0,
        },
    }

    match_doc: dict[str, Any] = {
        "schema_version": "1.0",
        "match_id": mid,
        "game_id": mid,
        "match_type": match_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "oracle_sims": oracle_sims,
        "opening": opening_moves,
        "red_player": asdict(red_player),
        "blue_player": asdict(blue_player),
        # Backward-compatible fields for single-LLM analyzers & blog exporters
        "model_label": primary_model_label,
        "model_id": primary_model_id,
        "llm_side": primary_llm_side,
        "prompt_mode": primary_prompt_mode,
        "opponent": opponent_label,
        "moves": moves,
        "illegal_attempts": 0,
        "turns": [],
        "per_side_telemetry": per_side_telemetry,
        "telemetry": {
            "total_latency_ms": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_reasoning_tokens": 0,
            "total_cost_usd": 0.0,
        },
        "result": "ongoing",
        "winner": None,
        "winning_line": [],
    }

    print(
        f"[start] {mid} | Red={red_player.label} ({red_player.prompt_mode or 'engine'}) "
        f"vs Blue={blue_player.label} ({blue_player.prompt_mode or 'engine'})"
    )

    state_before = await asyncio.to_thread(query_engine, moves)

    while state_before["result"] == "ongoing":
        side: str = state_before["sideToMove"]
        current_player = red_player if side == "R" else blue_player
        ply: int = state_before["ply"]
        legal_moves: list[str] = state_before["legalMoves"]

        chosen_move: str | None = None
        combined_thought: str = ""
        visible_thought: str = ""
        reasoning_trace: str | None = None
        engine_decision: dict[str, Any] | None = None

        turn_latency_ms = 0.0
        turn_prompt_tokens = 0
        turn_completion_tokens = 0
        turn_reasoning_tokens = 0
        turn_cost_usd = 0.0
        turn_illegal_retries = 0
        attempts_used = 0
        provider_model = current_player.model_id or current_player.actor_id
        finish_reason = "stop"
        raw_excerpt = ""

        if current_player.actor_type == "engine":
            t0 = time.perf_counter()
            assert current_player.engine_key is not None
            chosen_move, engine_decision = await asyncio.to_thread(
                choose_engine_move,
                list(moves),
                current_player.engine_key,
                seed,
            )
            turn_latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            attempts_used = 1
        else:
            assert current_player.model_id is not None
            prompt_text = build_turn_prompt(
                state_before,
                side,
                current_player.prompt_mode or "raw",
            )
            messages: list[dict[str, str]] = [{"role": "user", "content": prompt_text}]
            eff_max_tokens = current_player.max_tokens
            if current_player.reasoning_effort and current_player.reasoning_effort != "none":
                eff_max_tokens = max(eff_max_tokens, 6144)

            for attempt_idx in range(max_illegal_retries + 1):
                attempts_used += 1
                resp = await asyncio.to_thread(
                    call_openrouter_chat,
                    api_key=resolved_key,
                    model_id=current_player.model_id,
                    messages=messages,
                    max_tokens=eff_max_tokens,
                    temperature=current_player.temperature,
                    reasoning_effort=current_player.reasoning_effort,
                )

                turn_latency_ms += float(resp.get("latency_ms", 0.0))
                turn_prompt_tokens += int(resp.get("prompt_tokens", 0))
                turn_completion_tokens += int(resp.get("completion_tokens", 0))
                turn_reasoning_tokens += int(resp.get("reasoning_tokens", 0))
                turn_cost_usd += float(resp.get("cost_usd", 0.0))
                provider_model = str(resp.get("provider_model") or current_player.model_id)
                finish_reason = str(resp.get("finish_reason") or "stop")
                raw_excerpt = str(resp.get("raw_response_excerpt") or "")

                cand_move, comb_t, vis_t, reas_t = extract_move_and_thoughts(
                    resp.get("content", ""),
                    resp.get("native_reasoning"),
                )
                if cand_move and cand_move in legal_moves:
                    chosen_move = cand_move
                    combined_thought = comb_t
                    visible_thought = vis_t
                    reasoning_trace = reas_t
                    break

                # Illegal or missing move -> record retry
                turn_illegal_retries += 1
                match_doc["illegal_attempts"] += 1
                per_side_telemetry[side]["illegal_attempts"] += 1
                was_truncated = finish_reason.lower() in ("length", "max_tokens")
                if was_truncated:
                    eff_max_tokens = max(eff_max_tokens * 2, 8192)

                assistant_reply = resp.get("content") or "<empty response>"
                messages.append({"role": "assistant", "content": assistant_reply})
                messages.append(
                    {
                        "role": "user",
                        "content": build_retry_prompt(
                            cand_move,
                            legal_moves,
                            was_truncated=was_truncated,
                        ),
                    }
                )

        # Accumulate telemetry
        side_tel = per_side_telemetry[side]
        side_tel["total_latency_ms"] = round(
            side_tel["total_latency_ms"] + turn_latency_ms, 1
        )
        side_tel["total_prompt_tokens"] += turn_prompt_tokens
        side_tel["total_completion_tokens"] += turn_completion_tokens
        side_tel["total_reasoning_tokens"] += turn_reasoning_tokens
        side_tel["total_cost_usd"] = round(
            side_tel["total_cost_usd"] + turn_cost_usd, 6
        )

        match_doc["telemetry"]["total_latency_ms"] = round(
            match_doc["telemetry"]["total_latency_ms"] + turn_latency_ms, 1
        )
        match_doc["telemetry"]["total_prompt_tokens"] += turn_prompt_tokens
        match_doc["telemetry"]["total_completion_tokens"] += turn_completion_tokens
        match_doc["telemetry"]["total_reasoning_tokens"] += turn_reasoning_tokens
        match_doc["telemetry"]["total_cost_usd"] = round(
            match_doc["telemetry"]["total_cost_usd"] + turn_cost_usd, 6
        )

        if chosen_move is None:
            other_side = "B" if side == "R" else "R"
            match_doc["result"] = "forfeit"
            match_doc["winner"] = other_side
            match_doc["per_side_metrics"] = {
                "R": _compute_side_metrics(match_doc["turns"], "R"),
                "B": _compute_side_metrics(match_doc["turns"], "B"),
            }
            _atomic_write_json(match_path, match_doc)
            print(
                f"[forfeit] {mid}: {current_player.label} ({side}) exceeded "
                f"{max_illegal_retries} illegal retries at ply {ply}"
            )
            return match_doc

        # Shadow oracle evaluation (az-puct-400) before & after the move
        eval_metrics = await asyncio.to_thread(
            evaluate_move_oracle,
            list(moves),
            chosen_move,
            seed,
            oracle_sims,
        )

        moves.append(chosen_move)
        state_after = await asyncio.to_thread(query_engine, list(moves))

        turn_record: dict[str, Any] = {
            "ply": ply,
            "actor": current_player.actor_type,
            "actor_type": current_player.actor_type,
            "actor_id": current_player.actor_id,
            "actor_label": current_player.label,
            "side": side,
            "prompt_mode": current_player.prompt_mode,
            "reasoning_effort": current_player.reasoning_effort,
            "move": chosen_move,
            "board_before": encode_board(state_before["board"]),
            "board_after": encode_board(state_after["board"]),
            "flips": state_after["lastFlips"],
            "counts_before": state_before["counts"],
            "counts_after": state_after["counts"],
            "active_row": state_before["activeRow"],
            "active_col": state_before["activeCol"],
            "wildcard": bool(state_before["wildcard"]),
            "next_active_row": state_after["activeRow"],
            "next_active_col": state_after["activeCol"],
            "next_wildcard": bool(state_after["wildcard"]),
            "thought": combined_thought if current_player.actor_type == "llm" else None,
            "visible_thought": visible_thought if current_player.actor_type == "llm" else None,
            "reasoning_trace": reasoning_trace,
            "eval": eval_metrics,
            "telemetry": {
                "latency_ms": round(turn_latency_ms, 1),
                "prompt_tokens": turn_prompt_tokens,
                "completion_tokens": turn_completion_tokens,
                "reasoning_tokens": turn_reasoning_tokens,
                "cost_usd": round(turn_cost_usd, 6),
                "provider_model": provider_model,
                "finish_reason": finish_reason,
                "raw_response_excerpt": raw_excerpt,
                "illegal_retries": turn_illegal_retries,
                "attempts": attempts_used,
            },
        }
        if engine_decision is not None:
            turn_record["expected_score"] = round(
                float(engine_decision["expectedScore"]), 4
            )
            turn_record["network_value"] = (
                round(float(engine_decision["networkValue"]), 4)
                if engine_decision.get("networkValue") is not None
                else None
            )

        match_doc["turns"].append(turn_record)
        match_doc["per_side_metrics"] = {
            "R": _compute_side_metrics(match_doc["turns"], "R"),
            "B": _compute_side_metrics(match_doc["turns"], "B"),
        }

        print(
            f"  [ply {ply:>2} {side}] {current_player.label:<24} -> {chosen_move} "
            f"(az400={eval_metrics['az400_best_move']}, rank=#{eval_metrics['search_rank']}, "
            f"win={eval_metrics['win_prob_before']:.2f}->{eval_metrics['win_prob_after']:.2f}, "
            f"Δ={eval_metrics['delta_win_prob']:+.3f}, "
            f"{turn_latency_ms:>5.0f}ms, reas_tok={turn_reasoning_tokens}, ${turn_cost_usd:.4f})"
        )

        state_before = state_after
        _atomic_write_json(match_path, match_doc)

    match_doc["result"] = state_before["result"]
    match_doc["winner"] = state_before["winner"]
    match_doc["winning_line"] = state_before.get("winningLine", [])
    match_doc["completed_at"] = datetime.now(timezone.utc).isoformat()
    match_doc["per_side_metrics"] = {
        "R": _compute_side_metrics(match_doc["turns"], "R"),
        "B": _compute_side_metrics(match_doc["turns"], "B"),
    }
    _atomic_write_json(match_path, match_doc)

    winner_side = match_doc["winner"]
    winner_player = red_player if winner_side == "R" else blue_player
    tel = match_doc["telemetry"]
    print(
        f"[done]  {mid} | Winner: {winner_side} ({winner_player.label}) in {len(moves)} plies | "
        f"illegal={match_doc['illegal_attempts']} | "
        f"tokens={tel['total_prompt_tokens']}+{tel['total_completion_tokens']} "
        f"(reas={tel['total_reasoning_tokens']}) | cost=${tel['total_cost_usd']:.4f}"
    )
    return match_doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Crosshair-Bench match between any two players (LLM or Engine)."
    )
    parser.add_argument(
        "--red",
        required=True,
        help="Red player spec (e.g. 'google/gemini-3.8-flash', 'openai/gpt-6-luna-pro@medium', or 'engine:az-puct-1')",
    )
    parser.add_argument(
        "--blue",
        required=True,
        help="Blue player spec (e.g. 'engine:az-puct-400', 'anthropic/claude-haiku-5.5', etc.)",
    )
    parser.add_argument(
        "--red-mode",
        choices=list(PROMPT_MODES),
        default="raw",
        help="Prompt mode for Red if Red is an LLM (default: raw)",
    )
    parser.add_argument(
        "--blue-mode",
        choices=list(PROMPT_MODES),
        default="raw",
        help="Prompt mode for Blue if Blue is an LLM (default: raw)",
    )
    parser.add_argument(
        "--red-effort",
        default=None,
        help="Reasoning effort override for Red (low, medium, high, none)",
    )
    parser.add_argument(
        "--blue-effort",
        default=None,
        help="Reasoning effort override for Blue (low, medium, high, none)",
    )
    parser.add_argument("--id", default=None, help="Explicit match ID")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--opening", default="", help="Space-separated opening moves")
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Max tokens per LLM call"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--matches-dir",
        default=str(DEFAULT_MATCHES_DIR),
        help="Directory to store match JSON outputs",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing completed match"
    )
    args = parser.parse_args()

    red_spec = PlayerSpec.from_spec(
        args.red,
        prompt_mode=args.red_mode,
        reasoning_effort=args.red_effort,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    blue_spec = PlayerSpec.from_spec(
        args.blue,
        prompt_mode=args.blue_mode,
        reasoning_effort=args.blue_effort,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    asyncio.run(
        run_match(
            red_player=red_spec,
            blue_player=blue_spec,
            match_id=args.id,
            seed=args.seed,
            opening=args.opening,
            overwrite=args.overwrite,
            matches_dir=Path(args.matches_dir),
        )
    )


if __name__ == "__main__":
    main()
