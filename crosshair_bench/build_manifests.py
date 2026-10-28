#!/usr/bin/env python3
"""
Deterministic batch manifest generator for the Crosshair-Bench 5-Company Tournament.

Generates JSON manifests in `manifests/`:
- `batch_00_engine_anchors.json`: 6 engine pairs x 16 games = 96 games ($0.00 local CPU)
- `batch_01a_vs_uct50.json`: 20 LLM entrants vs `random-uct-50` x 4 games = 80 games
- `batch_01b_vs_az400.json`: 20 LLM entrants vs `az-puct-400` x 4 games = 80 games
- `batch_01c_vs_az4000.json`: 20 LLM entrants vs `az-puct-4000` x 4 games = 80 games
- `batch_01_vs_engines.json`: Combined 240 LLM-vs-Engine games (01a + 01b + 01c)
- `batch_02_thinking_delta.json`: 10 intra-model (@high vs @low/@none) + 5 intra-company
  (Flagship@low/@none vs Flash@high) pairs x 4 games = 60 games
- `batch_03_flagship_rr.json`: 10 pairs of 5 Company Flagships (@high) x 4 games = 40 games
- `batch_04_flash_rr.json`: 10 pairs of 5 Company Flash models (@high) x 4 games = 40 games
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import ENGINE_SHORT_LABELS
from .roster import (
    ALL_CALIBRATION_ENGINES,
    CALIBRATION_OPENINGS_8,
    COMPANY_DEFINITIONS,
    ENTRANTS_BY_SPEC,
    LLM_ENTRANTS,
    TOURNAMENT_ENGINES,
    TOURNAMENT_OPENINGS_2,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = REPO_ROOT / "manifests"


def _actor_short(spec: str) -> str:
    if spec.startswith("engine:"):
        ek = spec.split("engine:", 1)[1]
        return ENGINE_SHORT_LABELS.get(ek, ek)
    if spec in ENTRANTS_BY_SPEC:
        return ENTRANTS_BY_SPEC[spec].short_id
    return spec.replace("/", "_").replace("@", "_")


def _build_paired_games(
    *,
    batch_prefix: str,
    pair_slug: str,
    player_a: str,
    player_b: str,
    openings: tuple[str, ...],
    base_seed: int,
    prompt_mode: str = "raw",
    skip_shadow_oracle: bool = False,
    category: str = "",
) -> list[dict[str, Any]]:
    """
    Create `2 * len(openings)` color-balanced games between `player_a` and `player_b`.
    For each opening `op`, Game 1 has `player_a` as Red and `player_b` as Blue, and
    Game 2 has `player_b` as Red and `player_a` as Blue, with identical seed per opening pair.
    """
    games: list[dict[str, Any]] = []
    short_a = _actor_short(player_a)
    short_b = _actor_short(player_b)
    game_num = 0
    for op_idx, opening in enumerate(openings, start=1):
        op_seed = base_seed + op_idx * 100
        for swap_idx, (red_spec, blue_spec, red_s, blue_s) in enumerate(
            (
                (player_a, player_b, short_a, short_b),
                (player_b, player_a, short_b, short_a),
            ),
            start=1,
        ):
            game_num += 1
            mid = f"{batch_prefix}_{pair_slug}_g{game_num:02d}_{red_s}_vs_{blue_s}"
            games.append(
                {
                    "match_id": mid,
                    "red": red_spec,
                    "blue": blue_spec,
                    "red_mode": None if red_spec.startswith("engine:") else prompt_mode,
                    "blue_mode": None if blue_spec.startswith("engine:") else prompt_mode,
                    "opening": opening,
                    "opening_index": op_idx,
                    "swap_index": swap_idx,
                    "seed": op_seed,
                    "skip_shadow_oracle": skip_shadow_oracle,
                    "category": category,
                }
            )
    return games


def _write_manifest(
    filename: str,
    batch_id: str,
    title: str,
    description: str,
    matches: list[dict[str, Any]],
) -> Path:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS_DIR / filename
    doc = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "title": title,
        "description": description,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_matches": len(matches),
        "matches": matches,
    }
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] Wrote {path.relative_to(REPO_ROOT)} ({len(matches)} matches)")
    return path


def build_all_manifests() -> None:
    # -------------------------------------------------------------------------
    # Batch 0: Engine Anchor Calibration (6 pairs x 16 games = 96 local CPU games)
    # -------------------------------------------------------------------------
    b00_matches: list[dict[str, Any]] = []
    engine_pairs = list(itertools.combinations(ALL_CALIBRATION_ENGINES, 2))
    for pair_idx, (eng_a, eng_b) in enumerate(engine_pairs, start=1):
        sa = _actor_short(eng_a)
        sb = _actor_short(eng_b)
        b00_matches.extend(
            _build_paired_games(
                batch_prefix="b00",
                pair_slug=f"{sa}_{sb}",
                player_a=eng_a,
                player_b=eng_b,
                openings=CALIBRATION_OPENINGS_8,
                base_seed=10000 + pair_idx * 1000,
                skip_shadow_oracle=True,
                category="engine_calibration",
            )
        )
    _write_manifest(
        "batch_00_engine_anchors.json",
        "batch_00_engine_anchors",
        "Batch 0: Engine Anchor Calibration Round-Robin",
        "All 6 pairs among engine:az-puct-1, engine:random-uct-50, engine:az-puct-400, and engine:az-puct-4000 across 8 balanced 2-ply openings with Red/Blue swap (96 local CPU games).",
        b00_matches,
    )

    # -------------------------------------------------------------------------
    # Batch 1: All 20 LLM Entrants vs 3 Tournament Engines (240 games total)
    # -------------------------------------------------------------------------
    b01_by_engine: dict[str, list[dict[str, Any]]] = {e: [] for e in TOURNAMENT_ENGINES}
    for eng_idx, eng_spec in enumerate(TOURNAMENT_ENGINES, start=1):
        eng_short = _actor_short(eng_spec)
        for ent_idx, entrant in enumerate(LLM_ENTRANTS, start=1):
            pair_games = _build_paired_games(
                batch_prefix="b01",
                pair_slug=f"{entrant.short_id}_vs_{eng_short}",
                player_a=entrant.spec,
                player_b=eng_spec,
                openings=TOURNAMENT_OPENINGS_2,
                base_seed=20000 + eng_idx * 5000 + ent_idx * 100,
                skip_shadow_oracle=False,
                category=f"vs_{eng_short}",
            )
            b01_by_engine[eng_spec].extend(pair_games)

    _write_manifest(
        "batch_01a_vs_uct50.json",
        "batch_01a_vs_uct50",
        "Batch 1A: 20 LLM Entrants vs engine:random-uct-50",
        "All 20 LLM entrants (5 companies x 2 models x 2 thinking levels) vs engine:random-uct-50 (4 balanced games each = 80 games).",
        b01_by_engine["engine:random-uct-50"],
    )
    _write_manifest(
        "batch_01b_vs_az400.json",
        "batch_01b_vs_az400",
        "Batch 1B: 20 LLM Entrants vs engine:az-puct-400",
        "All 20 LLM entrants (5 companies x 2 models x 2 thinking levels) vs engine:az-puct-400 (4 balanced games each = 80 games).",
        b01_by_engine["engine:az-puct-400"],
    )
    _write_manifest(
        "batch_01c_vs_az4000.json",
        "batch_01c_vs_az4000",
        "Batch 1C: 20 LLM Entrants vs engine:az-puct-4000",
        "All 20 LLM entrants (5 companies x 2 models x 2 thinking levels) vs engine:az-puct-4000 (4 balanced games each = 80 games).",
        b01_by_engine["engine:az-puct-4000"],
    )
    for item in b01_by_engine["engine:az-puct-1"]:
        item["skip_shadow_oracle"] = True
    _write_manifest(
        "batch_01d_vs_az1.json",
        "batch_01d_vs_az1",
        "Batch 1D: 20 LLM Entrants vs engine:az-puct-1",
        "All 20 LLM entrants (5 companies x 2 models x 2 thinking levels) vs engine:az-puct-1 (4 balanced games each = 80 games).",
        b01_by_engine["engine:az-puct-1"],
    )

    b01_all = (
        b01_by_engine["engine:random-uct-50"]
        + b01_by_engine["engine:az-puct-1"]
        + b01_by_engine["engine:az-puct-400"]
        + b01_by_engine["engine:az-puct-4000"]
    )
    _write_manifest(
        "batch_01_vs_engines.json",
        "batch_01_vs_engines",
        "Batch 1: All 20 LLM Entrants vs 4 Tournament Engines",
        "Combined 320-game LLM-vs-Engine calibration stage (Batch 1A + Batch 1D + Batch 1B + Batch 1C).",
        b01_all,
    )

    # -------------------------------------------------------------------------
    # Batch 2: Thinking-Budget Delta & Intra-Company Tier Crossover (60 games)
    #   Part A: 10 Intra-Model pairs (@high vs @low/@none) x 4 games = 40 games
    #   Part B: 5 Intra-Company pairs (Flagship@low/@none vs Flash@high) x 4 games = 20 games
    # -------------------------------------------------------------------------
    b02_matches: list[dict[str, Any]] = []
    pair_counter = 0
    for comp in COMPANY_DEFINITIONS:
        # 1. Flagship @high vs Flagship @base
        flag_high = f"{comp.flagship_model_id}@high"
        flag_base = f"{comp.flagship_model_id}@{comp.flagship_base_effort}"
        pair_counter += 1
        b02_matches.extend(
            _build_paired_games(
                batch_prefix="b02",
                pair_slug=f"{comp.flagship_short_base}_high_vs_{comp.flagship_base_effort}",
                player_a=flag_high,
                player_b=flag_base,
                openings=TOURNAMENT_OPENINGS_2,
                base_seed=40000 + pair_counter * 500,
                skip_shadow_oracle=False,
                category="intra_model_thinking_delta",
            )
        )

        # 2. Flash @high vs Flash @base
        flash_high = f"{comp.flash_model_id}@high"
        flash_base = f"{comp.flash_model_id}@{comp.flash_base_effort}"
        pair_counter += 1
        b02_matches.extend(
            _build_paired_games(
                batch_prefix="b02",
                pair_slug=f"{comp.flash_short_base}_high_vs_{comp.flash_base_effort}",
                player_a=flash_high,
                player_b=flash_base,
                openings=TOURNAMENT_OPENINGS_2,
                base_seed=40000 + pair_counter * 500,
                skip_shadow_oracle=False,
                category="intra_model_thinking_delta",
            )
        )

    for comp in COMPANY_DEFINITIONS:
        # 3. Intra-company crossover: Flagship @base vs Flash @high
        flag_base = f"{comp.flagship_model_id}@{comp.flagship_base_effort}"
        flash_high = f"{comp.flash_model_id}@high"
        pair_counter += 1
        b02_matches.extend(
            _build_paired_games(
                batch_prefix="b02",
                pair_slug=f"{comp.flagship_short_base}_{comp.flagship_base_effort}_vs_{comp.flash_short_base}_high",
                player_a=flag_base,
                player_b=flash_high,
                openings=TOURNAMENT_OPENINGS_2,
                base_seed=40000 + pair_counter * 500,
                skip_shadow_oracle=False,
                category="intra_company_crossover",
            )
        )

    _write_manifest(
        "batch_02_thinking_delta.json",
        "batch_02_thinking_delta",
        "Batch 2: Intra-Model Thinking Delta & Intra-Company Tier Crossover",
        "10 intra-model (@high vs @low/@none) pairs + 5 intra-company (Flagship@base vs Flash@high) pairs x 4 balanced games = 60 games.",
        b02_matches,
    )

    # -------------------------------------------------------------------------
    # Batch 3: 5-Company Flagship Round-Robin at @high (10 pairs x 4 games = 40 games)
    # -------------------------------------------------------------------------
    b03_matches: list[dict[str, Any]] = []
    flagship_high_entrants = [
        e for e in LLM_ENTRANTS if e.tier == "flagship" and e.thinking_tier == "high"
    ]
    for pair_idx, (ent_a, ent_b) in enumerate(
        itertools.combinations(flagship_high_entrants, 2), start=1
    ):
        b03_matches.extend(
            _build_paired_games(
                batch_prefix="b03",
                pair_slug=f"{ent_a.company_slug}_vs_{ent_b.company_slug}",
                player_a=ent_a.spec,
                player_b=ent_b.spec,
                openings=TOURNAMENT_OPENINGS_2,
                base_seed=60000 + pair_idx * 500,
                skip_shadow_oracle=False,
                category="flagship_high_rr",
            )
        )
    _write_manifest(
        "batch_03_flagship_rr.json",
        "batch_03_flagship_rr",
        "Batch 3: 5-Company Flagship Round-Robin (@high)",
        "All 10 head-to-head pairs among the 5 company flagship models at @high x 4 balanced games = 40 games.",
        b03_matches,
    )

    # -------------------------------------------------------------------------
    # Batch 4: 5-Company Flash Round-Robin at @high (10 pairs x 4 games = 40 games)
    # -------------------------------------------------------------------------
    b04_matches: list[dict[str, Any]] = []
    flash_high_entrants = [
        e for e in LLM_ENTRANTS if e.tier == "flash" and e.thinking_tier == "high"
    ]
    for pair_idx, (ent_a, ent_b) in enumerate(
        itertools.combinations(flash_high_entrants, 2), start=1
    ):
        b04_matches.extend(
            _build_paired_games(
                batch_prefix="b04",
                pair_slug=f"{ent_a.company_slug}_vs_{ent_b.company_slug}",
                player_a=ent_a.spec,
                player_b=ent_b.spec,
                openings=TOURNAMENT_OPENINGS_2,
                base_seed=80000 + pair_idx * 500,
                skip_shadow_oracle=False,
                category="flash_high_rr",
            )
        )
    _write_manifest(
        "batch_04_flash_rr.json",
        "batch_04_flash_rr",
        "Batch 4: 5-Company Flash Round-Robin (@high)",
        "All 10 head-to-head pairs among the 5 company flash/cost-efficient models at @high x 4 balanced games = 40 games.",
        b04_matches,
    )


if __name__ == "__main__":
    build_all_manifests()
