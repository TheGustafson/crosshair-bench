# Crosshair-Bench

**Crosshair-Bench** evaluates LLMs on [Crosshair](https://thegustafson.com/games/crosshair), a 7×7 connection-and-flip board game invented on October 25, 2026.

Because the game is brand new, zero games, opening books, or strategy guides exist in pretraining data. Red connects top to bottom; Blue connects left to right; orthogonal sandwiches and 2×2 diagonal crosscuts flip enemy stones in chain reactions. Every game finishes within 49 plies with zero draws, and each move restricts the next player to the same row or column (~8 to 12 legal moves per turn).

- **Live Leaderboard & Game Explorer:** [thegustafson.com/benchmark](https://thegustafson.com/benchmark)
- **Technical Writeup:** [Crosshair-Bench: Evaluating Frontier Models on an Unseen Game](https://thegustafson.com/blog/llms-vs-crosshair)
- **Play Crosshair:** [thegustafson.com/games/crosshair](https://thegustafson.com/games/crosshair)
- **Rust AlphaZero Engine (`ai-crosshair`):** [github.com/TheGustafson/ai-crosshair](https://github.com/TheGustafson/ai-crosshair) ([training writeup](https://thegustafson.com/blog/crosshair-alphazero))

## How It Works

### 1. Full-Game League (`crosshair_bench.match_runner` & `crosshair_bench.run_batch`)

Models receive a stateless prompt on each turn via OpenRouter containing the rules, the 7×7 ASCII board (`a1` to `g7`), stone coordinate lists, the active row/column constraint, and the explicit list of legal moves, and return `<move>coord</move>`.

Entrants play paired games (swapping Red and Blue from balanced 2-ply openings) against four deterministic [`ai-crosshair`](https://github.com/TheGustafson/ai-crosshair) engine baselines and against each other:

- **`random-uct-50` (`1078.4 Elo`):** 50-simulation uniform-rollout MCTS with no neural network.
- **`az-puct-1` (`1200.0 Elo` anchor):** The raw 114,290-parameter (`457 KB`) AlphaZero policy network with tree search turned off (`1` simulation, zero lookahead).
- **`az-puct-400` (`1730.4 Elo`):** 400-simulation AlphaZero PUCT search.
- **`az-puct-4000` (`2096.8 Elo`):** 4,000-simulation AlphaZero PUCT search.

On every turn of every game, a 4,000-simulation AlphaZero shadow oracle evaluates the board before and after the chosen move to record win-probability trajectories, Mean Win-Probability Loss (`MWPL`), Blunder Rate ($\Delta\text{WinProb} \le -0.10$), and Top-1 / Top-3 agreement with `az-puct-4000`. All completed game logs live in `results/matches/`.

### 2. Single-Turn Tactical Suite (`benchmarks/crosshair_bench_100.json`)

`Crosshair-Bench-100` contains 100 high-leverage positions mined from tournament games where the best move and median legal move differ by at least 14% win probability. Every legal move in every position is pre-graded by `ai-crosshair`, allowing fast single-turn evaluation of any model in under two minutes without running local MCTS searches during grading.

## Results (556-Game Tournament)

20 LLM configurations (`5 labs × 2 model tiers × 2 reasoning budgets`, all in unassisted `raw` mode) were evaluated alongside the 4 engine baselines across 556 complete games:

- **Batch 0 (`96 games`):** Engine calibration round-robin (`random-uct-50`, `az-puct-1`, `az-puct-400`, `az-puct-4000`; 16 games per pair).
- **Batch 1 (`320 games`):** All 20 LLM configurations vs. `random-uct-50`, `az-puct-1`, `az-puct-400`, and `az-puct-4000` (4 games per pair).
- **Batch 2 (`60 games`):** Same-model thinking ablation (`@high` vs. `@low`/`@none`) and cross-tier showdown (`Flash @high` vs. `Flagship @low`/`@none`).
- **Batch 3 (`40 games`):** 5×5 Flagship `@high` round-robin.
- **Batch 4 (`40 games`):** 5×5 Flash `@high` round-robin.

### 24-Entrant Anchored BayesElo Leaderboard

Ratings are fit jointly across all 556 tournament games via regularized Bradley-Terry maximum likelihood anchored at `az-puct-1 = 1200.0 Elo`.

| Rank | Model / Engine | Elo | Overall W–L | vs `uct50` | vs `az1` | vs `az400` | vs `az4000` | AZ-4000 Top-1% | AZ-4000 Top-3% | MWPL | Blunder% | Cost/Game |
| ---: | :--- | ---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `engine:az-puct-4000` | **2096.8** | 126–2 | - | - | - | - | 100.0% | 100.0% | 0.010 | 0.8% | $0.000 |
| 2 | `engine:az-puct-400` | **1730.4** | 110–18 | - | - | - | - | 78.1% | 99.0% | 0.019 | 2.2% | $0.000 |
| 3 | `gpt-6.1-sol@high` | **1464.0** | 24–12 | 3–1 | 3–1 | 2–2 | 1–3 | 46.0% | 77.5% | 0.036 | 6.8% | $0.627 |
| 4 | `gemini-4-argon@high` | **1409.5** | 23–13 | 3–1 | 2–2 | 1–3 | 0–4 | 47.0% | 76.6% | 0.037 | 8.2% | $0.408 |
| 5 | `gpt-6.1-sol@low` | **1296.8** | 10–14 | 4–0 | 2–2 | 0–4 | 0–4 | 41.5% | 69.3% | 0.053 | 12.2% | $0.152 |
| 6 | `engine:az-puct-1` (anchor) | **1200.0** | 65–63 | - | - | - | - | 52.8% | 89.1% | 0.051 | 10.6% | $0.000 |
| 7 | `gemini-3.8-flash@high` | **1156.6** | 19–21 | 1–3 | 1–3 | 0–4 | 0–4 | 41.6% | 67.8% | 0.044 | 12.7% | $0.189 |
| 8 | `gpt-6-luna@high` | **1144.3** | 18–22 | 2–2 | 2–2 | 0–4 | 0–4 | 35.2% | 63.9% | 0.055 | 14.4% | $0.022 |
| 9 | `gemini-3.8-flash@low` | **1139.1** | 6–14 | 3–1 | 1–3 | 0–4 | 0–4 | 39.5% | 58.3% | 0.051 | 15.2% | $0.042 |
| 10 | `claude-opus-5.5@high` | **1124.1** | 13–23 | 3–1 | 2–2 | 0–4 | 0–4 | 36.1% | 62.9% | 0.056 | 15.9% | $0.537 |
| 11 | `claude-haiku-5.5@high` | **1116.4** | 18–22 | 3–1 | 1–3 | 0–4 | 0–4 | 38.6% | 58.4% | 0.061 | 22.1% | $0.018 |
| 12 | `gemini-4-argon@low` | **1106.5** | 6–18 | 3–1 | 0–4 | 0–4 | 0–4 | 41.3% | 72.1% | 0.052 | 14.2% | $0.101 |
| 13 | `engine:random-uct-50` | **1078.4** | 47–81 | - | - | - | - | 27.3% | 54.9% | 0.057 | 16.9% | $0.000 |
| 14 | `claude-haiku-5.5@low` | **1071.2** | 5–15 | 3–1 | 1–3 | 0–4 | 0–4 | 33.3% | 56.6% | 0.056 | 17.7% | $0.010 |
| 15 | `deepseek-v4-pro@high` | **1066.4** | 12–24 | 2–2 | 1–3 | 0–4 | 0–4 | 38.4% | 59.9% | 0.060 | 17.8% | $0.109 |
| 16 | `deepseek-v4.1-flash@none` | **1048.4** | 5–15 | 1–3 | 2–2 | 0–4 | 0–4 | 23.8% | 55.5% | 0.068 | 17.2% | $0.004 |
| 17 | `deepseek-v4.1-flash@high` | **1046.8** | 16–24 | 3–1 | 0–4 | 0–4 | 0–4 | 40.2% | 65.2% | 0.057 | 16.8% | $0.044 |
| 18 | `gpt-6-luna@low` | **1021.1** | 4–16 | 0–4 | 1–3 | 0–4 | 0–4 | 28.0% | 55.2% | 0.064 | 15.8% | $0.007 |
| 19 | `claude-opus-5.5@low` | **994.1** | 5–19 | 2–2 | 1–3 | 0–4 | 0–4 | 33.3% | 54.5% | 0.075 | 24.0% | $0.320 |
| 20 | `qwen3.8-max@high` | **955.4** | 8–28 | 1–3 | 2–2 | 0–4 | 0–4 | 28.5% | 60.9% | 0.059 | 18.1% | $0.165 |
| 21 | `qwen3.8-flash@high` | **862.2** | 9–31 | 1–3 | 0–4 | 0–4 | 0–4 | 27.8% | 57.6% | 0.068 | 18.4% | $0.013 |
| 22 | `qwen3.8-flash@none` | **843.2** | 3–17 | 0–4 | 1–3 | 0–4 | 0–4 | 26.0% | 48.4% | 0.064 | 19.6% | $0.009 |
| 23 | `qwen3.8-max@low` | **755.9** | 3–21 | 1–3 | 1–3 | 0–4 | 0–4 | 27.0% | 59.4% | 0.064 | 18.0% | $0.189 |
| 24 | `deepseek-v4-pro@none` | **656.5** | 1–23 | 1–3 | 0–4 | 0–4 | 0–4 | 19.3% | 46.7% | 0.067 | 19.6% | $0.014 |

### Flash `@high` vs. Flagship `@low`/`@none` (`16–4`)

In direct 4-game head-to-head matches within each lab, smaller Flash models with `@high` thinking beat their own labs' low-effort Flagships `16–4` (`80.0%`):

| Lab | Flash `@high` | Flagship `@low`/`@none` | Head-to-Head | Elo Delta | Cost/Game (`Flash` vs `Flagship`) |
| :--- | :--- | :--- | :---: | ---: | :--- |
| **Anthropic** | `claude-haiku-5.5@high` | `claude-opus-5.5@low` | **4–0** | +122.3 | $0.018 vs $0.320 (`17.5×` cheaper) |
| **DeepSeek** | `deepseek-v4.1-flash@high` | `deepseek-v4-pro@none` | **4–0** | +390.3 | $0.044 vs $0.014 |
| **Qwen** | `qwen3.8-flash@high` | `qwen3.8-max@low` | **4–0** | +106.3 | $0.013 vs $0.189 (`14.0×` cheaper) |
| **Google** | `gemini-3.8-flash@high` | `gemini-4-argon@low` | **2–2** | +50.1 | $0.189 vs $0.101 |
| **OpenAI** | `gpt-6-luna@high` | `gpt-6.1-sol@low` | **2–2** | -152.5 | $0.022 vs $0.152 (`6.8×` cheaper) |

### 5×5 Round-Robin Standings (Batch 3 & Batch 4)

**Batch 3: Flagship `@high` Round-Robin (`40 games`)**

| Rank | Model | RR W–L | Win% | vs `gemini-4-argon` | vs `gpt-6.1-sol` | vs `claude-opus-5.5` | vs `deepseek-v4-pro` | vs `qwen3.8-max` |
| ---: | :--- | :---: | ---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `gemini-4-argon@high` | **14–2** | 87.5% | - | 2–2 | 4–0 | 4–0 | 4–0 |
| 2 | `gpt-6.1-sol@high` | **13–3** | 81.2% | 2–2 | - | 3–1 | 4–0 | 4–0 |
| 3 | `claude-opus-5.5@high` | **6–10** | 37.5% | 0–4 | 1–3 | - | 2–2 | 3–1 |
| 4 | `deepseek-v4-pro@high` | **5–11** | 31.2% | 0–4 | 0–4 | 2–2 | - | 3–1 |
| 5 | `qwen3.8-max@high` | **2–14** | 12.5% | 0–4 | 0–4 | 1–3 | 1–3 | - |

**Batch 4: Flash `@high` Round-Robin (`40 games`)**

| Rank | Model | RR W–L | Win% | vs `gemini-3.8-flash` | vs `gpt-6-luna` | vs `claude-haiku-5.5` | vs `deepseek-v4.1-flash` | vs `qwen3.8-flash` |
| ---: | :--- | :---: | ---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `gemini-3.8-flash@high` | **13–3** | 81.2% | - | 2–2 | 4–0 | 3–1 | 4–0 |
| 2 | `gpt-6-luna@high` | **11–5** | 68.8% | 2–2 | - | 2–2 | 3–1 | 4–0 |
| 3 | `claude-haiku-5.5@high` | **7–9** | 43.8% | 0–4 | 2–2 | - | 2–2 | 3–1 |
| 4 | `deepseek-v4.1-flash@high` | **7–9** | 43.8% | 1–3 | 1–3 | 2–2 | - | 3–1 |
| 5 | `qwen3.8-flash@high` | **2–14** | 12.5% | 0–4 | 0–4 | 1–3 | 1–3 | - |

## Quickstart

Requires Python 3.10+ (zero third-party pip dependencies) and the compiled [`ai-crosshair`](https://github.com/TheGustafson/ai-crosshair) binary on `PATH` (or set `AI_CROSSHAIR_BIN=/path/to/ai-crosshair`). For LLM evaluation via OpenRouter, set `OPENROUTER_API_KEY` in your environment or store it in `~/.config/crosshair-arena/openrouter.env` (`chmod 600`).

```bash
# 1. Run a single match (LLM vs Engine or LLM vs LLM)
python3 -m crosshair_bench.match_runner \
  --red google/gemini-4-argon@high --red-mode raw \
  --blue engine:az-puct-400 \
  --opening "d4 f4" \
  --id sample_gemini4argon_vs_az400

# 2. Run a tournament batch manifest with concurrency and per-ply checkpoint resume
python3 -m crosshair_bench.run_batch manifests/batch_03_flagship_rr.json --concurrency 24

# 3. Evaluate a model on the 100-position single-turn tactical suite (Crosshair-Bench-100)
python3 -m crosshair_bench.eval_bench100 \
  --player google/gemini-3.8-flash@high \
  --concurrency 16

# 4. Recompute the anchored BayesElo leaderboard from results/matches/
python3 -m crosshair_bench.elo
```

## License

MIT License. See [LICENSE](LICENSE).
