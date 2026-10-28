#!/usr/bin/env python3
"""
Crash-resilient concurrent batch runner for Crosshair-Bench tournament manifests.

Usage:
  python3 -m crosshair_bench.run_batch manifests/batch_00_engine_anchors.json --concurrency 8

Features:
- Concurrent execution via `asyncio.Semaphore(concurrency)`
- Automatic per-match skip if `results/matches/<match_id>.json` is already complete
- Automatic per-ply checkpoint resume if `results/checkpoints/<match_id>.partial.json` exists
- Live atomic progress updates to `results/batches/<batch_id>_status.json`
- Automatic `results/leaderboard.json` recomputation upon batch completion
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .elo import DEFAULT_LEADERBOARD_OUT, compute_leaderboard, print_leaderboard
from .match_runner import (
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_MATCHES_DIR,
    DEFAULT_TURNS_LOG_PATH,
    PlayerSpec,
    run_match,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BATCHES_STATUS_DIR = REPO_ROOT / "results" / "batches"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


async def execute_batch(
    manifest_path: Path,
    *,
    concurrency: int = 8,
    limit: int | None = None,
    overwrite: bool = False,
    verbose_plies: bool = False,
    matches_dir: Path = DEFAULT_MATCHES_DIR,
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR,
    turns_log_path: Path = DEFAULT_TURNS_LOG_PATH,
    status_dir: Path = DEFAULT_BATCHES_STATUS_DIR,
    leaderboard_out: Path = DEFAULT_LEADERBOARD_OUT,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch_id = manifest.get("batch_id") or manifest_path.stem
    title = manifest.get("title") or batch_id
    all_items: list[dict[str, Any]] = manifest["matches"]
    raw_items = all_items[:limit] if limit is not None else list(all_items)

    # Schedule slowest models first, interleaved across providers so no single upstream
    # endpoint receives a burst of 20 simultaneous connections at t=0.
    def _schedule_key(idx_and_item: tuple[int, dict[str, Any]]) -> tuple[int, int, int, int]:
        orig_idx, it = idx_and_item
        specs_str = f"{it.get('red', '')} {it.get('blue', '')}"
        is_slowest = any(
            s in specs_str
            for s in (
                "openai/gpt-6.1-sol@high",
                "deepseek/deepseek-v4-pro-0813@high",
                "google/gemini-4-argon@high",
                "deepseek/deepseek-v4.1-flash-1017@high",
            )
        )
        game_round = (int(it.get("opening_index", 1)) - 1) * 2 + int(
            it.get("swap_index", 1)
        )
        return (0 if is_slowest else 1, game_round, orig_idx % 7, orig_idx)

    items = [
        it for _, it in sorted(enumerate(raw_items), key=_schedule_key)
    ]
    total = len(items)

    status_path = status_dir / f"{batch_id}_status.json"
    started_at = datetime.now(timezone.utc).isoformat()
    t_batch_start = time.perf_counter()

    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=max(32, concurrency * 2)))

    completed_count = 0
    failed_count = 0
    batch_cost_usd = 0.0
    batch_prompt_tokens = 0
    batch_completion_tokens = 0
    batch_reasoning_tokens = 0
    match_summaries: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    lock = asyncio.Lock()

    def build_status_doc(state_str: str) -> dict[str, Any]:
        rel_manifest = (
            str(manifest_path.relative_to(REPO_ROOT))
            if manifest_path.is_relative_to(REPO_ROOT)
            else str(manifest_path)
        )
        return {
            "batch_id": batch_id,
            "title": title,
            "manifest_path": rel_manifest,
            "state": state_str,
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(time.perf_counter() - t_batch_start, 2),
            "concurrency": concurrency,
            "total_matches": total,
            "completed_matches": completed_count,
            "failed_matches": failed_count,
            "remaining_matches": total - completed_count - failed_count,
            "batch_cost_usd": round(batch_cost_usd, 6),
            "batch_prompt_tokens": batch_prompt_tokens,
            "batch_completion_tokens": batch_completion_tokens,
            "batch_reasoning_tokens": batch_reasoning_tokens,
            "failures": failures,
            "matches": [
                match_summaries[m["match_id"]]
                for m in items
                if m["match_id"] in match_summaries
            ],
        }

    _atomic_write_json(status_path, build_status_doc("running"))
    print(
        f"=== Starting {title} ({total} matches, concurrency={concurrency}) ===",
        flush=True,
    )

    sem = asyncio.Semaphore(concurrency)

    async def worker(idx: int, item: dict[str, Any]) -> None:
        nonlocal completed_count, failed_count, batch_cost_usd
        nonlocal batch_prompt_tokens, batch_completion_tokens, batch_reasoning_tokens

        mid = item["match_id"]
        red_spec = PlayerSpec.from_spec(
            item["red"],
            prompt_mode=item.get("red_mode") or "raw",
        )
        blue_spec = PlayerSpec.from_spec(
            item["blue"],
            prompt_mode=item.get("blue_mode") or "raw",
        )
        seed = int(item.get("seed", 42))
        opening = str(item.get("opening", ""))
        skip_oracle = bool(item.get("skip_shadow_oracle", False))

        async with sem:
            t0 = time.perf_counter()
            try:
                doc = await run_match(
                    red_player=red_spec,
                    blue_player=blue_spec,
                    match_id=mid,
                    seed=seed,
                    opening=opening,
                    skip_shadow_oracle=skip_oracle,
                    overwrite=overwrite,
                    quiet=not verbose_plies,
                    matches_dir=matches_dir,
                    checkpoints_dir=checkpoints_dir,
                    turns_log_path=turns_log_path,
                )
                elapsed_s = round(time.perf_counter() - t0, 2)
                tel = doc.get("telemetry") or {}
                m_cost = float(tel.get("total_cost_usd", 0.0))
                m_ptok = int(tel.get("total_prompt_tokens", 0))
                m_ctok = int(tel.get("total_completion_tokens", 0))
                m_rtok = int(tel.get("total_reasoning_tokens", 0))
                winner_side = doc.get("winner")
                winner_label = (
                    red_spec.label
                    if winner_side == "R"
                    else (blue_spec.label if winner_side == "B" else "none")
                )
                plies = len(doc.get("moves", []))

                async with lock:
                    completed_count += 1
                    batch_cost_usd += m_cost
                    batch_prompt_tokens += m_ptok
                    batch_completion_tokens += m_ctok
                    batch_reasoning_tokens += m_rtok
                    match_summaries[mid] = {
                        "match_id": mid,
                        "red": red_spec.label,
                        "blue": blue_spec.label,
                        "opening": opening,
                        "result": doc.get("result"),
                        "winner": winner_side,
                        "winner_label": winner_label,
                        "plies": plies,
                        "cost_usd": round(m_cost, 6),
                        "elapsed_sec": elapsed_s,
                    }
                    done_now = completed_count
                    cost_now = batch_cost_usd
                    _atomic_write_json(status_path, build_status_doc("running"))

                print(
                    f"[{done_now:>{len(str(total))}}/{total}] DONE {mid} | "
                    f"winner={winner_side} ({winner_label}) | plies={plies:>2} | "
                    f"time={elapsed_s:>5.1f}s | batch_cost=${cost_now:.4f}",
                    flush=True,
                )
            except Exception as exc:
                async with lock:
                    failed_count += 1
                    failures.append(
                        {
                            "match_id": mid,
                            "error": str(exc),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    _atomic_write_json(status_path, build_status_doc("running"))
                print(
                    f"[FAIL] {mid} ({idx}/{total}): {exc}",
                    flush=True,
                )

    await asyncio.gather(*(worker(i + 1, item) for i, item in enumerate(items)))

    final_state = "completed" if failed_count == 0 else "completed_with_errors"
    final_status = build_status_doc(final_state)
    _atomic_write_json(status_path, final_status)

    print(
        f"\n=== Batch {batch_id} {final_state.upper()}: "
        f"{completed_count}/{total} completed, {failed_count} failed, "
        f"elapsed={final_status['elapsed_sec']:.1f}s, total_cost=${batch_cost_usd:.4f} ===\n",
        flush=True,
    )

    summary = compute_leaderboard([matches_dir])
    leaderboard_out.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_out.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print_leaderboard(summary)
    return final_status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Crosshair-Bench batch manifest with concurrency and per-ply checkpointing."
    )
    parser.add_argument(
        "manifest",
        help="Path to batch manifest JSON (e.g. manifests/batch_00_engine_anchors.json)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum concurrent matches (default: 8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of matches from the manifest to run",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if completed match JSON already exists",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-ply logs in addition to per-match completion lines",
    )
    parser.add_argument(
        "--matches-dir",
        default=str(DEFAULT_MATCHES_DIR),
        help="Directory to store completed match JSON outputs",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=str(DEFAULT_CHECKPOINTS_DIR),
        help="Directory to store per-ply in-progress checkpoints",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    status = asyncio.run(
        execute_batch(
            manifest_path,
            concurrency=args.concurrency,
            limit=args.limit,
            overwrite=args.overwrite,
            verbose_plies=args.verbose,
            matches_dir=Path(args.matches_dir),
            checkpoints_dir=Path(args.checkpoints_dir),
        )
    )
    if status["failed_matches"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
