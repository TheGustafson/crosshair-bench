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

- **`random-uct-50` (`1075.9 Elo`):** 50-simulation uniform-rollout MCTS with no neural network.
- **`az-puct-1` (`1200.0 Elo` anchor):** The raw 114,290-parameter (`457 KB`) AlphaZero policy network with tree search turned off (`1` simulation, zero lookahead).
- **`az-puct-400` (`1742.9 Elo`):** 400-simulation AlphaZero PUCT search.
- **`az-puct-4000` (`2110.0 Elo`):** 4,000-simulation AlphaZero PUCT search.

On every turn of every game, a 4,000-simulation AlphaZero shadow oracle evaluates the board before and after the chosen move to record win-probability trajectories, Mean Win-Probability Loss (`MWPL`), Blunder Rate ($\Delta\text{WinProb} \le -0.10$), and Top-1 / Top-3 agreement with `az-puct-4000`. All completed game logs live in `results/matches/`.

### 2. Single-Turn Tactical Suite (`benchmarks/crosshair_bench_100.json`)

`Crosshair-Bench-100` contains 100 high-leverage positions mined from tournament games where the best move and median legal move differ by at least 14% win probability. Every legal move in every position is pre-graded by `ai-crosshair`, allowing fast single-turn evaluation of any model in under two minutes without running local MCTS searches during grading.

## Results (696-Game Tournament)

20 LLM configurations (`5 providers × 2 model tiers × 2 reasoning budgets`, all in unassisted `raw` mode, `44` games per model) were evaluated alongside the 4 engine baselines (`128` games per engine) across 696 complete games:

- **Batch 0 (`96 games`):** Engine calibration round-robin (`random-uct-50`, `az-puct-1`, `az-puct-400`, `az-puct-4000`; 16 games per pair).
- **Batch 1 (`320 games`):** All 20 LLM configurations vs. `random-uct-50`, `az-puct-1`, `az-puct-400`, and `az-puct-4000` (4 games per pair).
- **Batch 2 (`60 games`):** Same-model thinking ablation (`@high` vs. `@low`/`@none`) and cross-tier showdown (`Flash @high` vs. `Flagship @low`/`@none`).
- **Batch 3 (`40 games`):** 5×5 Flagship `@high` round-robin.
- **Batch 4 (`40 games`):** 5×5 Flash `@high` round-robin.
- **Batch 5 (`40 games`):** 5×5 Flagship `@low`/`@none` round-robin.
- **Batch 6 (`40 games`):** 5×5 Flash `@low`/`@none` round-robin.
- **Batch 7 (`60 games`):** Remaining intra-provider round-robin matchups (`Flagship @high` vs. `Flash @high`, `Flagship @low` vs. `Flash @low`, and `Flagship @high` vs. `Flash @low`).

### 24-Entrant Anchored BayesElo Leaderboard

Ratings are fit jointly across all 696 tournament games via regularized Bradley-Terry maximum likelihood anchored at `az-puct-1 = 1200.0 Elo`.

| Rank | Model / Engine | Elo | Overall W–L | vs `uct50` | vs `az1` | vs `az400` | vs `az4000` | AZ-4000 Top-1% | AZ-4000 Top-3% | MWPL | Blunder% | Cost/Game |
| ---: | :--- | ---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `engine:az-puct-4000` | **2110.0** | 126–2 | - | - | - | - | 100.0% | 100.0% | 0.010 | 0.8% | $0.000 |
| 2 | `engine:az-puct-400` | **1742.9** | 110–18 | - | - | - | - | 78.1% | 99.0% | 0.019 | 2.2% | $0.000 |
| 3 | `gpt-6.1-sol@high` | **1510.9** | 32–12 | 3–1 | 3–1 | 2–2 | 1–3 | 47.5% | 77.6% | 0.033 | 6.4% | $0.596 |
| 4 | `gemini-4-argon@high` | **1449.2** | 30–14 | 3–1 | 2–2 | 1–3 | 0–4 | 48.6% | 77.4% | 0.036 | 7.2% | $0.402 |
| 5 | `gpt-6.1-sol@low` | **1272.8** | 26–18 | 4–0 | 2–2 | 0–4 | 0–4 | 45.0% | 71.7% | 0.050 | 11.3% | $0.132 |
| 6 | `engine:az-puct-1` (anchor) | **1200.0** | 65–63 | - | - | - | - | 52.8% | 89.1% | 0.051 | 10.6% | $0.000 |
| 7 | `claude-opus-5.5@high` | **1189.0** | 20–24 | 3–1 | 2–2 | 0–4 | 0–4 | 37.9% | 64.7% | 0.054 | 15.7% | $0.524 |
| 8 | `gemini-4-argon@low` | **1182.8** | 22–22 | 3–1 | 0–4 | 0–4 | 0–4 | 44.7% | 74.4% | 0.045 | 12.4% | $0.086 |
| 9 | `gemini-3.8-flash@high` | **1140.0** | 19–25 | 1–3 | 1–3 | 0–4 | 0–4 | 41.4% | 66.7% | 0.045 | 12.7% | $0.189 |
| 10 | `gpt-6-luna@high` | **1127.2** | 18–26 | 2–2 | 2–2 | 0–4 | 0–4 | 34.8% | 63.2% | 0.056 | 14.9% | $0.022 |
| 11 | `deepseek-v4-pro@high` | **1112.2** | 18–26 | 2–2 | 1–3 | 0–4 | 0–4 | 39.3% | 60.7% | 0.060 | 18.1% | $0.105 |
| 12 | `gemini-3.8-flash@low` | **1107.8** | 19–25 | 3–1 | 1–3 | 0–4 | 0–4 | 40.8% | 61.9% | 0.057 | 17.2% | $0.040 |
| 13 | `claude-haiku-5.5@high` | **1087.2** | 19–25 | 3–1 | 1–3 | 0–4 | 0–4 | 39.0% | 59.4% | 0.061 | 21.6% | $0.019 |
| 14 | `engine:random-uct-50` | **1075.9** | 47–81 | - | - | - | - | 27.3% | 54.9% | 0.057 | 16.9% | $0.000 |
| 15 | `gpt-6-luna@low` | **1056.4** | 16–28 | 0–4 | 1–3 | 0–4 | 0–4 | 31.6% | 60.5% | 0.063 | 16.2% | $0.007 |
| 16 | `deepseek-v4.1-flash@high` | **1028.1** | 18–26 | 3–1 | 0–4 | 0–4 | 0–4 | 39.4% | 65.0% | 0.059 | 17.8% | $0.043 |
| 17 | `qwen3.8-max@high` | **1021.0** | 15–29 | 1–3 | 2–2 | 0–4 | 0–4 | 29.5% | 61.9% | 0.061 | 18.7% | $0.170 |
| 18 | `claude-opus-5.5@low` | **989.6** | 15–29 | 2–2 | 1–3 | 0–4 | 0–4 | 36.3% | 57.6% | 0.073 | 23.9% | $0.285 |
| 19 | `claude-haiku-5.5@low` | **986.2** | 15–29 | 3–1 | 1–3 | 0–4 | 0–4 | 34.8% | 56.8% | 0.070 | 24.4% | $0.010 |
| 20 | `deepseek-v4.1-flash@none` | **900.1** | 13–31 | 1–3 | 2–2 | 0–4 | 0–4 | 25.8% | 55.1% | 0.076 | 21.4% | $0.003 |
| 21 | `qwen3.8-flash@high` | **849.1** | 10–34 | 1–3 | 0–4 | 0–4 | 0–4 | 28.0% | 57.9% | 0.068 | 18.1% | $0.014 |
| 22 | `qwen3.8-flash@none` | **816.8** | 10–34 | 0–4 | 1–3 | 0–4 | 0–4 | 28.6% | 52.7% | 0.082 | 25.8% | $0.010 |
| 23 | `qwen3.8-max@low` | **776.7** | 9–35 | 1–3 | 1–3 | 0–4 | 0–4 | 27.6% | 61.1% | 0.069 | 20.2% | $0.170 |
| 24 | `deepseek-v4-pro@none` | **661.6** | 4–40 | 1–3 | 0–4 | 0–4 | 0–4 | 21.8% | 49.2% | 0.081 | 23.9% | $0.015 |

### Flash `@high` vs. Flagship `@low`/`@none` (`16–4`)

In direct 4-game head-to-head matches within each provider, smaller Flash models with `@high` thinking beat their own providers' low-effort Flagships `16–4` (`80.0%`):

| Provider | Flash `@high` | Flagship `@low`/`@none` | Head-to-Head | Elo Delta | Cost/Game (`Flash` vs `Flagship`) |
| :--- | :--- | :--- | :---: | ---: | :--- |
| **Anthropic** | `claude-haiku-5.5@high` | `claude-opus-5.5@low` | **4–0** | +97.6 | $0.019 vs $0.285 (`14.7×` cheaper) |
| **DeepSeek** | `deepseek-v4.1-flash@high` | `deepseek-v4-pro@none` | **4–0** | +366.5 | $0.043 vs $0.015 |
| **Qwen** | `qwen3.8-flash@high` | `qwen3.8-max@low` | **4–0** | +72.4 | $0.014 vs $0.170 (`12.6×` cheaper) |
| **Google** | `gemini-3.8-flash@high` | `gemini-4-argon@low` | **2–2** | -42.8 | $0.189 vs $0.086 |
| **OpenAI** | `gpt-6-luna@high` | `gpt-6.1-sol@low` | **2–2** | -145.6 | $0.022 vs $0.132 (`6.0×` cheaper) |

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
