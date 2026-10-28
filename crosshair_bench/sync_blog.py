#!/usr/bin/env python3
"""
Syncs completed matches and leaderboard data between `crosshair-bench` and a sibling
`private-blog` checkout (`PRIVATE_BLOG_ROOT` or `../private-blog`), then regenerates
`src/data/llmArenaGames.json` for the interactive blog visuals.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .elo import compute_leaderboard, print_leaderboard

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_MATCHES_DIR = REPO_ROOT / "results" / "matches"
LEADERBOARD_PATH = REPO_ROOT / "results" / "leaderboard.json"

BLOG_ROOT = (
    Path(os.environ.get("PRIVATE_BLOG_ROOT", str(REPO_ROOT.parent / "private-blog")))
    .expanduser()
    .resolve()
)
BLOG_GAMES_DIR = BLOG_ROOT / "notes" / "llm-arena" / "games"
BLOG_LEADERBOARD_PATH = BLOG_ROOT / "notes" / "llm-arena" / "leaderboard.json"
BLOG_EXPORT_SCRIPT = BLOG_ROOT / "notes" / "llm-arena" / "export_blog_data.py"


def sync_matches(
    *,
    import_existing_blog_games: bool = True,
    export_to_blog: bool = True,
) -> None:
    BENCH_MATCHES_DIR.mkdir(parents=True, exist_ok=True)
    BLOG_GAMES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Import any completed games from private-blog/notes/llm-arena/games not yet in crosshair-bench
    imported = 0
    if import_existing_blog_games and BLOG_GAMES_DIR.exists():
        for src in sorted(BLOG_GAMES_DIR.glob("*.json")):
            dst = BENCH_MATCHES_DIR / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                imported += 1

    # 2. Copy all completed LLM matches from crosshair-bench to private-blog/notes/llm-arena/games
    exported = 0
    if export_to_blog:
        for src in sorted(BENCH_MATCHES_DIR.glob("*.json")):
            try:
                doc = json.loads(src.read_text(encoding="utf-8"))
            except Exception:
                continue
            if doc.get("match_type") == "engine_vs_engine":
                continue
            dst = BLOG_GAMES_DIR / src.name
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src, dst)
                exported += 1

    print(
        f"[sync] Imported {imported} legacy games into crosshair-bench; "
        f"exported {exported} updated LLM matches to {BLOG_GAMES_DIR}"
    )

    # 3. Recompute leaderboard across all matches (including Batch 0 engine calibration)
    summary = compute_leaderboard([BENCH_MATCHES_DIR])
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_PATH.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if export_to_blog:
        BLOG_LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LEADERBOARD_PATH, BLOG_LEADERBOARD_PATH)
    print_leaderboard(summary)

    # 4. Run private-blog's export_blog_data.py
    if export_to_blog and BLOG_EXPORT_SCRIPT.is_file():
        subprocess.run([sys.executable, str(BLOG_EXPORT_SCRIPT)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync matches with private-blog and update leaderboard + blog JSON."
    )
    parser.add_argument(
        "--no-import",
        action="store_true",
        help="Do not import legacy games from private-blog",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not copy matches or run export_blog_data.py in private-blog",
    )
    args = parser.parse_args()
    sync_matches(
        import_existing_blog_games=not args.no_import,
        export_to_blog=not args.no_export,
    )


if __name__ == "__main__":
    main()
