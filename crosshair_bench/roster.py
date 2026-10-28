"""
Canonical 7-Company (28 LLM Entrants + 4 Engine Actors) Tournament Roster for Crosshair-Bench.

Structure:
- 7 Frontier AI Labs: OpenAI, Anthropic, Google, DeepSeek, Qwen, Meta, xAI
- 2 Model Tiers per Lab: Flagship ("best") and Fast/Cost-Efficient ("flash")
- 2 Thinking Regimes per Model: Base (`@low` or `@none`) and Extended Thinking (`@high`)
- Total LLM Entrants: 7 * 2 * 2 = 28
- Engine Actors:
  * Calibration Anchor (1200.0 Elo): `engine:az-puct-1`
  * Tournament Engine 1: `engine:random-uct-50`
  * Tournament Engine 2: `engine:az-puct-400`
  * Tournament Engine 3: `engine:az-puct-4000`
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMEntrant:
    company: str
    company_slug: str
    tier: str  # "flagship" or "flash"
    thinking_tier: str  # "base" or "high"
    model_id: str
    reasoning_effort: str  # "low", "none", or "high"
    spec: str  # e.g. "openai/gpt-6.1-sol@high"
    short_id: str  # concise slug for match IDs, e.g. "gpt61sol_high"


@dataclass(frozen=True)
class CompanyModels:
    company: str
    company_slug: str
    flagship_model_id: str
    flagship_short_base: str
    flagship_base_effort: str
    flash_model_id: str
    flash_short_base: str
    flash_base_effort: str


COMPANY_DEFINITIONS: tuple[CompanyModels, ...] = (
    CompanyModels(
        company="OpenAI",
        company_slug="openai",
        flagship_model_id="openai/gpt-6.1-sol",
        flagship_short_base="gpt61sol",
        flagship_base_effort="low",
        flash_model_id="openai/gpt-6-luna",
        flash_short_base="gpt6luna",
        flash_base_effort="low",
    ),
    CompanyModels(
        company="Anthropic",
        company_slug="anthropic",
        flagship_model_id="anthropic/claude-opus-5.5",
        flagship_short_base="opus55",
        flagship_base_effort="low",
        flash_model_id="anthropic/claude-haiku-5.5",
        flash_short_base="haiku55",
        flash_base_effort="low",
    ),
    CompanyModels(
        company="Google",
        company_slug="google",
        flagship_model_id="google/gemini-4-argon",
        flagship_short_base="gemini4argon",
        flagship_base_effort="low",
        flash_model_id="google/gemini-3.8-flash",
        flash_short_base="gemini38flash",
        flash_base_effort="low",
    ),
    CompanyModels(
        company="DeepSeek",
        company_slug="deepseek",
        flagship_model_id="deepseek/deepseek-v4-pro-0813",
        flagship_short_base="deepseekv4pro",
        flagship_base_effort="none",
        flash_model_id="deepseek/deepseek-v4.1-flash-1017",
        flash_short_base="deepseekv41flash",
        flash_base_effort="none",
    ),
    CompanyModels(
        company="Qwen",
        company_slug="qwen",
        flagship_model_id="qwen/qwen3.8-max-0902",
        flagship_short_base="qwen38max",
        flagship_base_effort="none",
        flash_model_id="qwen/qwen3.8-flash",
        flash_short_base="qwen38flash",
        flash_base_effort="none",
    ),
    CompanyModels(
        company="Meta",
        company_slug="meta",
        flagship_model_id="meta/muse-wonder",
        flagship_short_base="musewonder",
        flagship_base_effort="low",
        flash_model_id="meta/muse-spark-open",
        flash_short_base="musespark",
        flash_base_effort="low",
    ),
    CompanyModels(
        company="xAI",
        company_slug="xai",
        flagship_model_id="x-ai/grok-4.7",
        flagship_short_base="grok47",
        flagship_base_effort="low",
        flash_model_id="x-ai/grok-4.3",
        flash_short_base="grok43",
        flash_base_effort="low",
    ),
)


def _build_llm_roster() -> tuple[LLMEntrant, ...]:
    entrants: list[LLMEntrant] = []
    for comp in COMPANY_DEFINITIONS:
        for tier, mid, sbase, base_eff in (
            (
                "flagship",
                comp.flagship_model_id,
                comp.flagship_short_base,
                comp.flagship_base_effort,
            ),
            (
                "flash",
                comp.flash_model_id,
                comp.flash_short_base,
                comp.flash_base_effort,
            ),
        ):
            for t_tier, eff in (("base", base_eff), ("high", "high")):
                entrants.append(
                    LLMEntrant(
                        company=comp.company,
                        company_slug=comp.company_slug,
                        tier=tier,
                        thinking_tier=t_tier,
                        model_id=mid,
                        reasoning_effort=eff,
                        spec=f"{mid}@{eff}",
                        short_id=f"{sbase}_{eff}",
                    )
                )
    return tuple(entrants)


LLM_ENTRANTS: tuple[LLMEntrant, ...] = _build_llm_roster()
ENTRANTS_BY_SPEC: dict[str, LLMEntrant] = {e.spec: e for e in LLM_ENTRANTS}

CALIBRATION_ANCHOR_ENGINE = "engine:az-puct-1"
TOURNAMENT_ENGINES: tuple[str, ...] = (
    "engine:random-uct-50",
    "engine:az-puct-400",
    "engine:az-puct-4000",
)
ALL_CALIBRATION_ENGINES: tuple[str, ...] = (
    "engine:az-puct-1",
    "engine:random-uct-50",
    "engine:az-puct-400",
    "engine:az-puct-4000",
)

# Balanced 2-ply openings verified by az-puct-400 (Red win probability ~ 0.49 - 0.51).
# In every 4-game pairing, each of the 2 primary openings is played twice (once with
# Player A as Red / Player B as Blue, and once with Player B as Red / Player A as Blue).
TOURNAMENT_OPENINGS_2: tuple[str, ...] = (
    "d4 f4",
    "c4 f4",
)

# 8 balanced 2-ply openings for the 16-game-per-pair Engine Anchor Calibration (Batch 0).
CALIBRATION_OPENINGS_8: tuple[str, ...] = (
    "d4 f4",
    "c4 f4",
    "d4 d2",
    "d4 b4",
    "e4 c4",
    "c4 e4",
    "d3 d5",
    "d5 d3",
)
