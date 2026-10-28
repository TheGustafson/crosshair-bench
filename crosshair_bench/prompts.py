"""
Prompt builders (`raw`, `assisted`, `strategic`) and `<move>coord</move>` response
parsers for Crosshair-Bench.
"""

from __future__ import annotations

import re
from typing import Any

from .engine import format_board_ascii

PROMPT_MODES = ("raw", "assisted", "strategic")

RULES_TEXT = """\
=== CROSSHAIR RULES ===
- Board: 7x7 grid. Columns (files) are a..g (left to right). Rows (ranks) are 1..7 (top to bottom, where 1 is the top edge and 7 is the bottom edge).
- Win Goal (Hex-style connection):
  * Red (R, moves first) wins immediately by connecting the TOP edge (row 1) to the BOTTOM edge (row 7) with a 4-orthogonally connected chain of Red stones (up/down/left/right; diagonals do NOT connect).
  * Blue (B, moves second) wins immediately by connecting the LEFT edge (col a) to the RIGHT edge (col g) with a 4-orthogonally connected chain of Blue stones.
- Move Constraint (Crosshair routing):
  * After a stone is placed at square X, the next player MUST play on an empty square in the SAME ROW or SAME COLUMN as X.
  * Wildcard: If every square in that row and column is already occupied (or on turn 1), the player may play on ANY empty square on the board.
- Flips & Cascades (Zero-draw guarantee):
  * After placing a stone of your color C, enemy stones flip to C via two patterns that repeat in a chain reaction (cascade) until no more flips occur:
    1. Linear Sandwich: Your stone and another C stone orthogonally sandwich a single enemy stone along a row or column (`C [E] C` -> `C C C`).
    2. 2x2 Crosscut: A 2x2 block of squares contains two diagonal C stones and two diagonal enemy stones (`C E / E C` or `E C / C E`) -> both enemy stones flip to C.
"""

TACTICAL_PRIMER_TEXT = """\
=== TACTICAL PRIMER ===
1. Orthogonal only: Diagonal stones do NOT connect your road. You need an up/down/left/right path between your two target edges.
2. Don't walk into a pinch: Since your move at (col, row) forces the opponent into that SAME col or row next turn, never place a stone adjacent to an enemy stone if the square on the other side of you in that same row/col is empty (they will immediately sandwich and flip you!).
3. Watch 2x2 crosscuts: In any 2x2 square, if two diagonal corners are enemy stones and one corner is yours, never let the opponent play the 4th corner—completing a 2x2 diagonal checkerboard flips BOTH of your stones.
4. Starve their routing: Prefer moves that flip enemy stones AND route the opponent into a row/column where they have no flips and cannot extend their main chain."""

MOVE_TAG_RE = re.compile(r"<move>\s*([a-g][1-7])\s*</move>", re.IGNORECASE)
FALLBACK_MOVE_PATTERNS = [
    re.compile(r"final\s+move\s*[:\-]?\s*\*?\*?([a-g][1-7])\*?\*?", re.IGNORECASE),
    re.compile(r"chosen\s+move\s*[:\-]?\s*\*?\*?([a-g][1-7])\*?\*?", re.IGNORECASE),
    re.compile(r"move\s*[:\-]\s*\*?\*?([a-g][1-7])\*?\*?", re.IGNORECASE),
    re.compile(r"\*\*([a-g][1-7])\*\*", re.IGNORECASE),
    re.compile(r"\b([a-g][1-7])\b", re.IGNORECASE),
]


def build_turn_prompt(
    state: dict[str, Any],
    side: str,
    prompt_mode: str = "raw",
) -> str:
    """
    Build the stateless single-turn prompt for `side` ('R' or 'B') in `prompt_mode`
    ('raw', 'assisted', or 'strategic').
    """
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Invalid prompt_mode '{prompt_mode}'. Valid: {PROMPT_MODES}")

    side_name = (
        "Red (connect TOP row 1 to BOTTOM row 7)"
        if side == "R"
        else "Blue (connect LEFT col a to RIGHT col g)"
    )
    board = state["board"]
    red_stones: list[str] = []
    blue_stones: list[str] = []
    for idx, cell in enumerate(board):
        coord = f"{chr(ord('a') + (idx % 7))}{(idx // 7) + 1}"
        if cell == "R":
            red_stones.append(coord)
        elif cell == "B":
            blue_stones.append(coord)

    out: list[str] = [RULES_TEXT.rstrip()]
    if prompt_mode == "strategic":
        out.append("\n" + TACTICAL_PRIMER_TEXT)

    out.append(
        f"\nYou are playing as {side} — {side_name}. Orthogonal connections only (up/down/left/right)."
    )
    out.append(f"\n=== TURN {state['ply'] + 1} (Ply {state['ply']}) ===")
    if state.get("history"):
        out.append(
            f"Move history ({len(state['history'])} plies): {' '.join(state['history'])}"
        )
    if state.get("lastMove"):
        flips_note = (
            f" (flipped: {', '.join(state['lastFlips'])})"
            if state.get("lastFlips")
            else " (no flips)"
        )
        out.append(f"Opponent's last move: {state['lastMove']}{flips_note}")

    out.append(
        "\nCurrent Board (1=top edge, 7=bottom edge, a=left edge, g=right edge):"
    )
    out.append(format_board_ascii(board))
    out.append(
        f"\nRed stones ({len(red_stones)}): {', '.join(red_stones) if red_stones else 'none'}"
    )
    out.append(
        f"Blue stones ({len(blue_stones)}): {', '.join(blue_stones) if blue_stones else 'none'}"
    )

    if state["wildcard"]:
        out.append("Active constraint: WILDCARD (any empty square is legal).")
    else:
        row_num = state["activeRow"] + 1
        col_let = chr(ord("a") + state["activeCol"])
        out.append(f"Active crosshair constraint: Row {row_num} or Column {col_let}.")

    legal = state["legalMoves"]
    out.append(f"Legal moves ({len(legal)}): {', '.join(legal)}")

    if prompt_mode in ("assisted", "strategic"):
        out.append("\n=== TACTICAL PREVIEWS FOR EACH LEGAL MOVE ===")
        for prev in state["movePreviews"]:
            mv = prev["move"]
            parts = []
            if prev["immediateWin"]:
                parts.append("IMMEDIATE WIN!")
            if prev["flips"]:
                parts.append(
                    f"flips [{', '.join(prev['flips'])}] ({len(prev['flips'])} stones)"
                )
            else:
                parts.append("flips none")
            if prev["nextWildcard"]:
                parts.append("WARNING: grants opponent WILDCARD")
            else:
                nr = prev["nextActiveRow"] + 1
                nc = chr(ord("a") + prev["nextActiveCol"])
                parts.append(f"routes opponent to Row {nr} / Col {nc}")
            out.append(f"  - {mv}: {'; '.join(parts)}")

    out.append(
        "\nBriefly evaluate the top 2–3 candidate moves, checking flips and where your move routes the opponent next (keep your analysis concise, strictly under 200 words; do NOT exhaustively analyze every legal move).\n"
        "End your response with your final chosen legal coordinate inside XML tags, like:\n"
        "<move>d4</move>"
    )
    return "\n".join(out)


def build_retry_prompt(
    bad_move: str | None,
    legal_moves: list[str],
    *,
    was_truncated: bool = False,
) -> str:
    """Build a concise user retry prompt when the previous response had an illegal or missing move."""
    if was_truncated and not bad_move:
        return (
            "You reached the output token limit before emitting your `<move>...</move>` tag.\n"
            f"Choose from the legal moves ({', '.join(legal_moves)}) and respond IMMEDIATELY with ONLY your chosen coordinate inside XML tags (no extra commentary), e.g.:\n"
            f"<move>{legal_moves[0]}</move>"
        )

    bad_desc = f"'{bad_move}'" if bad_move else "<no coordinate found>"
    return (
        f"ERROR: Your move {bad_desc} is ILLEGAL in this position.\n"
        f"You MUST choose one of the legal moves: {', '.join(legal_moves)}\n"
        f"Do not write a long essay—respond immediately with your chosen legal coordinate inside XML tags, e.g. <move>{legal_moves[0]}</move>."
    )


def extract_move_and_thoughts(
    content: str | None,
    native_reasoning: str | None = None,
    *,
    finish_reason: str = "stop",
) -> tuple[str | None, str, str, str | None]:
    """
    Extract `(move, combined_thought, visible_thought, reasoning_trace)` from an LLM response.
    Prefers the last `<move>([a-g][1-7])</move>` tag.
    Crucially, if `finish_reason` is `'length'` or `'max_tokens'`, disables loose coordinate
    fallbacks (`**coord**` / bare `\\b[a-g][1-7]\\b`) so a response cut off mid-sentence
    never accidentally plays a random square mentioned in its incomplete analysis.
    """
    text = (content or "").strip()
    move: str | None = None
    visible_thought = text
    was_truncated = finish_reason.lower() in ("length", "max_tokens")

    tag_matches = list(MOVE_TAG_RE.finditer(text))
    if tag_matches:
        last_match = tag_matches[-1]
        move = last_match.group(1).lower()
        visible_thought = (
            text[: last_match.start()] + text[last_match.end() :]
        ).strip()
    else:
        # If truncated, only accept explicit "Final move: X" / "Chosen move: X" labels
        allowed_patterns = (
            FALLBACK_MOVE_PATTERNS[:2] if was_truncated else FALLBACK_MOVE_PATTERNS
        )
        for idx, pat in enumerate(allowed_patterns):
            search_region = text if idx < 4 else "\n".join(text.splitlines()[-3:])
            matches = list(pat.finditer(search_region))
            if matches:
                move = matches[-1].group(1).lower()
                break

    reasoning_clean = (
        native_reasoning.strip()
        if (native_reasoning and native_reasoning.strip())
        else None
    )

    # Fallback: if content was empty or had no move, check native_reasoning for explicit <move> tag
    if move is None and reasoning_clean:
        r_tags = list(MOVE_TAG_RE.finditer(reasoning_clean))
        if r_tags:
            move = r_tags[-1].group(1).lower()
        elif not was_truncated:
            for pat in FALLBACK_MOVE_PATTERNS[:3]:
                matches = list(pat.finditer(reasoning_clean))
                if matches:
                    move = matches[-1].group(1).lower()
                    break

    thought_parts: list[str] = []
    if reasoning_clean:
        thought_parts.append(reasoning_clean)
    if visible_thought and visible_thought != reasoning_clean:
        thought_parts.append(visible_thought)
    combined_thought = "\n\n".join(thought_parts).strip()
    if len(combined_thought) > 2500:
        combined_thought = combined_thought[:2497] + "..."

    return move, combined_thought, visible_thought, reasoning_clean
