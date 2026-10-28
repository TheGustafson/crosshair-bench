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

- **`random-uct-50` (`1135.4 Elo`):** 50-simulation uniform-rollout MCTS with no neural network.
- **`az-puct-1` (`1200.0 Elo` anchor):** The raw 114,290-parameter (`457 KB`) AlphaZero policy network with tree search turned off (`1` simulation, zero lookahead).
- **`az-puct-400` (`1887.7 Elo`):** 400-simulation AlphaZero PUCT search.
- **`az-puct-4000` (`2343.4 Elo`):** 4,000-simulation AlphaZero PUCT search.

On every turn of every LLM game, a 400-simulation AlphaZero shadow oracle evaluates the board before and after the chosen move to record win-probability trajectories, Mean Win-Probability Loss (`MWPL`), Blunder Rate ($\Delta\text{WinProb} \le -0.10$), and Top-1 / Top-3 agreement with `az-puct-400`. All completed game logs live in `results/matches/`.

### 2. Single-Turn Tactical Suite (`benchmarks/crosshair_bench_100.json`)

`Crosshair-Bench-100` contains 100 high-leverage positions mined from tournament games where the best move and median legal move differ by at least 14% win probability. Every legal move in every position is pre-graded by `az-puct-400`, allowing fast single-turn evaluation of any model in under two minutes without running local MCTS searches during grading.

## Results (495-Game Tournament)

20 LLM configurations (`5 labs × 2 model tiers × 2 reasoning budgets`, all in unassisted `raw` mode) were evaluated alongside the 4 engine baselines across 495 complete games:

- **Batch 0 (`96 games`):** Engine calibration round-robin (`random-uct-50`, `az-puct-1`, `az-puct-400`, `az-puct-4000`).
- **Batch 1 (`240 games`):** All 20 LLM configurations vs. `random-uct-50`, `az-puct-400`, and `az-puct-4000` (4 games per pair).
- **Batch 2 (`60 games`):** Same-model thinking ablation (`@high` vs. `@low`/`@none`) and cross-tier showdown (`Flash @high` vs. `Flagship @low`/`@none`).
- **Batch 3 (`40 games`):** 5×5 Flagship `@high` round-robin.
- **Batch 4 (`40 games`):** 5×5 Flash `@high` round-robin.
- **Pilot (`19 games`):** Prompt scaffolding ablation (`raw`, `assisted`, `strategic`).

### 24-Entrant Anchored BayesElo Leaderboard

Ratings are fit jointly across all 495 games via regularized Bradley-Terry maximum likelihood anchored at `az-puct-1 = 1200.0 Elo`.

| Rank | Model / Engine | Elo | Overall W–L | vs `uct50` | vs `az400` | vs `az4000` | AZ Top-1% | AZ Top-3% | MWPL | Blunder% | Cost/Game |
| ---: | :--- | ---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `engine:az-puct-4000` | **2343.4** | 127–1 | - | - | - | 82.9% | 96.5% | 0.013 | 0.0% | $0.000 |
| 2 | `engine:az-puct-400` | **1887.7** | 117–18 | - | - | - | 99.9% | 100.0% | 0.018 | 0.6% | $0.000 |
| 3 | `gemini-4-argon@high` | **1440.4** | 19–13 | 4–0 | 1–3 | 0–4 | 42.2% | 75.1% | 0.052 | 11.6% | $0.390 |
| 4 | `gpt-6.1-sol@low` | **1408.3** | 9–11 | 4–0 | 0–4 | 0–4 | 39.9% | 65.6% | 0.049 | 12.0% | $0.137 |
| 5 | `gpt-6.1-sol@high` | **1397.5** | 17–15 | 4–0 | 0–4 | 0–4 | 40.4% | 68.3% | 0.051 | 12.8% | $0.488 |
| 6 | `gemini-3.8-flash@high` | **1327.8** | 20–16 | 1–3 | 0–4 | 0–4 | 46.3% | 68.0% | 0.053 | 14.4% | $0.166 |
| 7 | `gemini-4-argon@low` | **1290.0** | 6–14 | 3–1 | 0–4 | 0–4 | 38.8% | 67.2% | 0.063 | 16.7% | $0.095 |
| 8 | `claude-opus-5.5@high` | **1258.9** | 14–18 | 3–1 | 0–4 | 0–4 | 36.0% | 59.2% | 0.058 | 17.2% | $0.551 |
| 9 | `claude-haiku-5.5@low` | **1255.3** | 5–11 | 3–1 | 0–4 | 0–4 | 38.3% | 56.9% | 0.060 | 21.3% | $0.010 |
| 10 | `claude-haiku-5.5@high` | **1222.7** | 16–20 | 2–2 | 0–4 | 0–4 | 38.0% | 57.6% | 0.063 | 21.9% | $0.019 |
| 11 | `deepseek-v4.1-flash@high` | **1211.2** | 18–18 | 2–2 | 0–4 | 0–4 | 35.1% | 62.4% | 0.056 | 16.6% | $0.043 |
| 12 | `engine:az-puct-1` (anchor) | **1200.0** | 12–45 | - | - | - | 54.2% | 91.7% | 0.088 | 12.5% | $0.000 |
| 13 | `gpt-6-luna@high` | **1157.5** | 13–23 | 2–2 | 0–4 | 0–4 | 36.5% | 65.2% | 0.062 | 17.0% | $0.021 |
| 14 | `deepseek-v4-pro@high` | **1147.7** | 11–21 | 3–1 | 0–4 | 0–4 | 38.7% | 63.7% | 0.062 | 15.7% | $0.321 |
| 15 | `engine:random-uct-50` | **1135.4** | 46–83 | - | - | - | 29.0% | 57.0% | 0.066 | 20.7% | $0.000 |
| 16 | `gemini-3.8-flash@low` | **1135.1** | 3–13 | 3–1 | 0–4 | 0–4 | 32.2% | 59.9% | 0.062 | 14.8% | $0.037 |
| 17 | `claude-opus-5.5@low` | **1085.3** | 4–16 | 2–2 | 0–4 | 0–4 | 31.8% | 53.6% | 0.075 | 22.2% | $0.312 |
| 18 | `qwen3.8-flash@high` | **1035.6** | 11–25 | 1–3 | 0–4 | 0–4 | 28.1% | 61.5% | 0.067 | 19.7% | $0.013 |
| 19 | `qwen3.8-max@high` | **978.0** | 6–26 | 1–3 | 0–4 | 0–4 | 31.6% | 59.9% | 0.061 | 18.8% | $0.183 |
| 20 | `gpt-6-luna@low` | **969.5** | 2–14 | 0–4 | 0–4 | 0–4 | 29.5% | 60.7% | 0.068 | 17.6% | $0.008 |
| 21 | `qwen3.8-flash@none` | **908.9** | 2–14 | 0–4 | 0–4 | 0–4 | 21.9% | 45.7% | 0.075 | 23.7% | $0.008 |
| 22 | `deepseek-v4-pro@none` | **899.4** | 2–18 | 1–3 | 0–4 | 0–4 | 20.4% | 48.2% | 0.067 | 21.2% | $0.021 |
| 23 | `qwen3.8-max@none` | **866.2** | 3–17 | 1–3 | 0–4 | 0–4 | 30.1% | 57.1% | 0.058 | 16.3% | $0.172 |
| 24 | `deepseek-v4.1-flash@none` | **864.7** | 1–15 | 1–3 | 0–4 | 0–4 | 17.1% | 51.7% | 0.068 | 19.3% | $0.003 |

### Flash `@high` vs. Flagship `@low`/`@none` (`16–4`)

In direct 4-game head-to-head matches within each lab, smaller Flash models with `@high` thinking beat their own labs' low-effort Flagships `16–4` (`80.0%`):

| Lab | Flash `@high` | Flagship `@low`/`@none` | Head-to-Head | Elo Delta | Cost/Game (`Flash` vs `Flagship`) |
| :--- | :--- | :--- | :---: | ---: | :--- |
| **Anthropic** | `claude-haiku-5.5@high` | `claude-opus-5.5@low` | **4–0** | +137.4 | $0.019 vs $0.312 (`16.5×` cheaper) |
| **DeepSeek** | `deepseek-v4.1-flash@high` | `deepseek-v4-pro@none` | **4–0** | +311.8 | $0.043 vs $0.021 |
| **Qwen** | `qwen3.8-flash@high` | `qwen3.8-max@none` | **4–0** | +169.4 | $0.013 vs $0.172 (`13.1×` cheaper) |
| **Google** | `gemini-3.8-flash@high` | `gemini-4-argon@low` | **2–2** | +37.8 | $0.166 vs $0.095 |
| **OpenAI** | `gpt-6-luna@high` | `gpt-6.1-sol@low` | **2–2** | -250.8 | $0.021 vs $0.137 (`6.6×` cheaper) |

### 5×5 Round-Robin Standings (Batch 3 & Batch 4)

**Batch 3: Flagship `@high` Round-Robin (`40 games`)**

| Rank | Model | RR W–L | Win% | vs `gpt-6.1-sol` | vs `gemini-4-argon` | vs `claude-opus-5.5` | vs `deepseek-v4-pro` | vs `qwen3.8-max` |
| ---: | :--- | :---: | ---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `gpt-6.1-sol@high` | **12–4** | 75.0% | - | 2–2 | 4–0 | 2–2 | 4–0 |
| 2 | `gemini-4-argon@high` | **11–5** | 68.8% | 2–2 | - | 1–3 | 4–0 | 4–0 |
| 3 | `claude-opus-5.5@high` | **9–7** | 56.3% | 0–4 | 3–1 | - | 2–2 | 4–0 |
| 4 | `deepseek-v4-pro@high` | **5–11** | 31.3% | 2–2 | 0–4 | 2–2 | - | 1–3 |
| 5 | `qwen3.8-max@high` | **3–13** | 18.8% | 0–4 | 0–4 | 0–4 | 3–1 | - |

**Batch 4: Flash `@high` Round-Robin (`40 games`)**

| Rank | Model | RR W–L | Win% | vs `gemini-3.8-flash` | vs `claude-haiku-5.5` | vs `deepseek-v4.1-flash` | vs `gpt-6-luna` | vs `qwen3.8-flash` |
| ---: | :--- | :---: | ---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `gemini-3.8-flash@high` | **13–3** | 81.3% | - | 3–1 | 4–0 | 4–0 | 2–2 |
| 2 | `claude-haiku-5.5@high` | **8–8** | 50.0% | 1–3 | - | 2–2 | 2–2 | 3–1 |
| 3 | `deepseek-v4.1-flash@high` | **8–8** | 50.0% | 0–4 | 2–2 | - | 2–2 | 4–0 |
| 4 | `gpt-6-luna@high` | **7–9** | 43.8% | 0–4 | 2–2 | 2–2 | - | 3–1 |
| 5 | `qwen3.8-flash@high` | **4–12** | 25.0% | 2–2 | 1–3 | 0–4 | 1–3 | - |

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
