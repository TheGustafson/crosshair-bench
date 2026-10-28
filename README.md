# Crosshair-Bench

**Crosshair-Bench** is a zero-contamination spatial reasoning, constraint-routing, and multi-step lookahead benchmark for frontier Large Language Models built on the abstract board game **Crosshair** and its compiled Rust AlphaZero engine (`ai-crosshair`).

Unlike Chess, Go, Othello, or Hex—where frontier models can lean on millions of memorized games, opening books, and human commentary in pretraining corpora—Crosshair was designed from scratch with zero prior literature. Every move forces the opponent into the **same row or column** just played, and stone placements trigger recursive **linear sandwiches** and **2×2 diagonal crosscut flips** until reaching a fixed point.

---

## Two-Part Benchmark Architecture

### Part 1: Crosshair-League (`crosshair_bench.match_runner` & `crosshair_bench.elo`)
A full-game competitive arena supporting **LLM vs Engine**, **LLM vs LLM**, and **Engine vs Engine** matches across three prompt regimes:
- **`raw`**: Complete rules, ASCII board, coordinate lists, active crosshair constraint, and legal move list. Tests pure 2D spatial parsing and forward state simulation.
- **`assisted`**: Appends exact 1-ply engine-computed tactical previews for each legal move (stones flipped, immediate win flag, and next active row/column or wildcard grant). Isolates multi-step lookahead from 1-ply board-reading errors.
- **`strategic`**: Prepends a 4-rule domain heuristic primer (orthogonal connectivity, avoiding 1-ply pinches, defending 2×2 crosscuts, and starving opponent routing) alongside 1-ply previews.

Every match is anchored to deterministic `ai-crosshair` baselines:
- `engine:az-puct-1` — Raw 114K-parameter policy prior ($P(s,a)$ with zero MCTS lookahead; **anchored at 1200 Elo**)
- `engine:random-uct-50` — 50-simulation uniform-rollout MCTS without neural guidance
- `engine:az-puct-400` — 400-simulation AlphaZero PUCT search (**also used as the per-turn shadow oracle**)
- `engine:az-puct-2000` — 2000-simulation AlphaZero PUCT tournament anchor

On **every ply** of every game (both Red and Blue), the harness queries the `az-puct-400` shadow oracle before and after the move, recording:
- Win-probability trajectory ($V_{\text{before}} \to V_{\text{after}}$) and per-turn $\Delta\text{WinProb}$
- Mean Win-Probability Loss (**MWPL**) and **Blunder Rate** ($\Delta\text{WinProb} \le -0.10$)
- Top-1 and Top-3 move agreement with `az-puct-400` and policy prior agreement with `az-puct-1`
- Top-5 MCTS candidate moves with visit counts, policy priors, and expected scores
- Full OpenRouter telemetry (`latency_ms`, `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `cost_usd`, `provider_model`, `finish_reason`, `illegal_retries`)

### Part 2: Crosshair-Bench-100 (`crosshair_bench.build_bench100` & `crosshair_bench.eval_bench100`)
A fast, reproducible single-turn tactical suite (`benchmarks/crosshair_bench_100.json`) of high-cliff positions ($\max_a Q(s,a) - \text{median}_a Q(s,a) \ge 0.18$) mined from arena games and mixed-strength self-play. Every legal move in every position is pre-graded by `az-puct-400`, enabling zero-simulation evaluation of any LLM's single-turn Top-1 accuracy, MWPL, and blunder rate across tactical categories (`immediate_win_or_save`, `self_routing_trap`, `crosscut_2x2_tactic`, `cascade_chain_reaction`, `positional_routing`).

---

## Setup & API Key Security

1. Ensure the compiled `ai-crosshair` release binary is available on `PATH`, at `/home/nick/projects/private-blog/game-ai/target/release/ai-crosshair`, or specified via `AI_CROSSHAIR_BIN`.
2. **Never store API keys inside the repository.** Store your OpenRouter key at `~/.config/crosshair-arena/openrouter.env` (`chmod 600`):
   ```bash
   install -d -m 700 ~/.config/crosshair-arena
   install -m 600 /dev/null ~/.config/crosshair-arena/openrouter.env
   echo 'OPENROUTER_API_KEY=sk-or-v1-...' > ~/.config/crosshair-arena/openrouter.env
   ```
   Or export `OPENROUTER_API_KEY` in your environment.

---

## Quickstart Commands

### Run a Single Match (LLM vs Engine or LLM vs LLM)
Model specs accept an optional `@<effort>` suffix (`minimal`, `low`, `medium`, `high`, `none`) to configure OpenRouter's `reasoning` parameter:

```bash
# LLM (Red, raw) vs Engine (Blue, az-puct-1)
python3 -m crosshair_bench.match_runner \
  --red google/gemini-3.8-flash --red-mode raw \
  --blue engine:az-puct-1 \
  --id smoke1_gemini38flash_raw_vs_az1_R

# LLM (Red, assisted, medium reasoning) vs Engine (Blue, az-puct-400)
python3 -m crosshair_bench.match_runner \
  --red openai/gpt-6-luna-pro@medium --red-mode assisted \
  --blue engine:az-puct-400 \
  --id smoke2_gpt6lunapro_assisted_vs_az400_R

# LLM vs LLM Head-to-Head (raw mode)
python3 -m crosshair_bench.match_runner \
  --red deepseek/deepseek-v4.1-flash --red-mode raw \
  --blue anthropic/claude-haiku-5.5 --blue-mode raw \
  --id smoke3_deepseekv41flash_vs_haiku55_raw
```

### Compute Anchored Elo & Move-Quality Leaderboard
```bash
python3 -m crosshair_bench.elo
```

### Build & Evaluate Crosshair-Bench-100
```bash
python3 -m crosshair_bench.build_bench100 --target 30
python3 -m crosshair_bench.eval_bench100 --player engine:az-puct-1
python3 -m crosshair_bench.eval_bench100 --player google/gemini-3.8-flash --prompt-mode raw --limit 10
```

### Sync Matches & Leaderboard to Blog
```bash
python3 -m crosshair_bench.sync_blog
```
