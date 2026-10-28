# Crosshair-Bench

**Crosshair-Bench** is a zero-contamination spatial reasoning, constraint-routing, and multi-step lookahead benchmark for frontier Large Language Models built on the abstract board game **Crosshair** and its compiled Rust AlphaZero engine (`ai-crosshair`).

Unlike Chess, Go, Othello, or Hex, where frontier models can lean on millions of memorized games, opening books, and human commentary in pretraining corpora, Crosshair was designed from scratch with zero prior literature. Every move forces the opponent into the **same row or column** just played, and stone placements trigger recursive **linear sandwiches** and **2x2 diagonal crosscut flips** until reaching a fixed point.

---

## Two-Part Benchmark Architecture

### Part 1: Crosshair-League (`crosshair_bench.match_runner`, `crosshair_bench.run_batch`, `crosshair_bench.elo`)
A full-game competitive arena supporting **LLM vs Engine**, **LLM vs LLM**, and **Engine vs Engine** matches across three prompt regimes:
- **`raw`**: Complete rules, ASCII board, coordinate lists, active crosshair constraint, and legal move list. Tests pure 2D spatial parsing and forward state simulation.
- **`assisted`**: Appends exact 1-ply engine-computed tactical previews for each legal move (stones flipped, immediate win flag, and next active row/column or wildcard grant). Isolates multi-step lookahead from 1-ply board-reading errors.
- **`strategic`**: Prepends a 4-rule domain heuristic primer (orthogonal connectivity, avoiding 1-ply pinches, defending 2x2 crosscuts, and starving opponent routing) alongside 1-ply previews.

#### 5-Company Tournament Roster (20 LLM Entrants + 4 Engine Actors)
Defined in `crosshair_bench/roster.py`:
- **OpenAI:** `openai/gpt-6.1-sol@low`, `openai/gpt-6.1-sol@high`, `openai/gpt-6-luna@low`, `openai/gpt-6-luna@high`
- **Anthropic:** `anthropic/claude-opus-5.5@low`, `anthropic/claude-opus-5.5@high`, `anthropic/claude-haiku-5.5@low`, `anthropic/claude-haiku-5.5@high`
- **Google:** `google/gemini-4-argon@low`, `google/gemini-4-argon@high`, `google/gemini-3.8-flash@low`, `google/gemini-3.8-flash@high`
- **DeepSeek:** `deepseek/deepseek-v4-pro-0813@none`, `deepseek/deepseek-v4-pro-0813@high`, `deepseek/deepseek-v4.1-flash-1017@none`, `deepseek/deepseek-v4.1-flash-1017@high`
- **Qwen:** `qwen/qwen3.8-max-0902@none`, `qwen/qwen3.8-max-0902@high`, `qwen/qwen3.8-flash@none`, `qwen/qwen3.8-flash@high`

Every match is anchored to deterministic `ai-crosshair` baselines (calibrated via the 96-game Batch 0 round-robin):
- `engine:az-puct-1`: Raw 114K-parameter policy prior ($P(s,a)$ with zero MCTS lookahead; **anchored at 1200.0 Elo**)
- `engine:random-uct-50`: 50-simulation uniform-rollout MCTS without neural guidance (**1133.8 Elo**)
- `engine:az-puct-400`: 400-simulation AlphaZero PUCT search (**1753.7 Elo**, also used as the per-turn shadow oracle)
- `engine:az-puct-4000`: 4000-simulation AlphaZero PUCT tournament ceiling anchor (**2196.2 Elo**)

On **every ply** of every LLM game (both Red and Blue), the harness queries the `az-puct-400` shadow oracle before and after the move, recording:
- Win-probability trajectory ($V_{\text{before}} \to V_{\text{after}}$) and per-turn $\Delta\text{WinProb}$
- Mean Win-Probability Loss (**MWPL**) and **Blunder Rate** ($\Delta\text{WinProb} \le -0.10$)
- Top-1 and Top-3 move agreement with `az-puct-400` and policy prior agreement with `az-puct-1`
- Top-5 MCTS candidate moves with visit counts, policy priors, and expected scores
- Full OpenRouter telemetry (`latency_ms`, `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `cost_usd`, `provider_model`, `finish_reason`, `illegal_retries`)
- Atomic per-turn checkpoints (`results/checkpoints/<match_id>.partial.json`) and append-only turn logs (`results/turns_log.jsonl`) so interrupted runs resume without losing a single completed turn.

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

### Run Tournament Batches (`manifests/`)
```bash
# Regenerate all deterministic batch manifests
python3 -m crosshair_bench.build_manifests

# Batch 0: Engine Anchor Calibration (96 local CPU games, $0.00)
python3 -m crosshair_bench.run_batch manifests/batch_00_engine_anchors.json --concurrency 8

# Batch 1: 20 LLM Entrants vs 3 Tournament Engines (80 games per sub-batch, 240 total)
python3 -m crosshair_bench.run_batch manifests/batch_01a_vs_uct50.json --concurrency 24
python3 -m crosshair_bench.run_batch manifests/batch_01b_vs_az400.json --concurrency 24
python3 -m crosshair_bench.run_batch manifests/batch_01c_vs_az4000.json --concurrency 24

# Batch 2: Intra-Model Thinking Delta & Intra-Company Crossover (60 games)
python3 -m crosshair_bench.run_batch manifests/batch_02_thinking_delta.json --concurrency 24

# Batch 3: 5-Company Flagship Round-Robin at @high (40 games)
python3 -m crosshair_bench.run_batch manifests/batch_03_flagship_rr.json --concurrency 24

# Batch 4: 5-Company Flash Round-Robin at @high (40 games)
python3 -m crosshair_bench.run_batch manifests/batch_04_flash_rr.json --concurrency 24
```

### Run a Single Match (LLM vs Engine or LLM vs LLM)
Model specs accept an optional `@<effort>` suffix (`minimal`, `low`, `medium`, `high`, `none`) to configure OpenRouter's `reasoning` parameter:

```bash
# LLM (Red, raw) vs Engine (Blue, az-puct-400)
python3 -m crosshair_bench.match_runner \
  --red openai/gpt-6.1-sol@high --red-mode raw \
  --blue engine:az-puct-400 \
  --opening "d4 f4" \
  --id sample_gpt61sol_high_vs_az400

# LLM vs LLM Head-to-Head (raw mode)
python3 -m crosshair_bench.match_runner \
  --red anthropic/claude-opus-5.5@high --red-mode raw \
  --blue google/gemini-4-argon@high --blue-mode raw \
  --opening "d4 f4" \
  --id sample_opus55_high_vs_gemini4argon_high
```

### Compute Anchored Elo & Move-Quality Leaderboard
```bash
python3 -m crosshair_bench.elo
```

### Sync Matches & Leaderboard to Blog
```bash
python3 -m crosshair_bench.sync_blog
```
