# mt5-ml-bot

A MetaTrader 5 forex bot: LightGBM long/short models predict direction on M5 bars, and Thompson-sampling bandits
pick the ATR stop and probability thresholds. Built as a first machine-learning project and kept for learning.

## Status: research only, do not fund

No strategy in this repo has passed a leak-free, pre-registered holdout on an XM Standard Ultra Low account
(`#` symbols). Do not run it on a real account. The live code still works as plumbing (`main.py` defaults to
dry-run; `mql5/H1Bridge.mq5` defaults to dry-run and refuses real accounts).

## What I found

### 1. The original backtests were inflated by look-ahead

The October 2025 version joined H1 features to M5 bars at the H1 bar's **open** time. Each M5 row could therefore
see that hour's final close, which covers most of the 12-bar label window. Live, that H1 bar is still forming, so
the real model never had this information. That is why demo and backtest results looked good and real money lost.

Out-of-sample AUC, EURUSD M5, 12-bar direction label, three walk-forward folds:

| Features | Fold 1 | Fold 2 | Fold 3 |
|---|---|---|---|
| Leak-free (H1 joined at close time) | 0.556 | 0.522 | 0.544 |
| Old open-time H1 join | 0.643 | 0.613 | 0.679 |

A second leak (5-bar fractals built with `shift(-1)`/`shift(-2)`) existed too but changed AUC by less than 0.01.
Both are fixed (commit d57d52f) and guarded by `tests/test_no_lookahead.py`.

### 2. Without the leak there is no edge after costs

All tests below charge the real recorded spread (shorts pay it at exit, on the ask), slippage, and, where stated,
swap. Pass rules were written down before each run.

| Strategy | Test data | Result per trade | Verdict |
|---|---|---|---|
| M5 LightGBM, 12-bar hold (EURUSD, GBPUSD, AUDUSD, USDJPY; GOLD had no edge even before costs) | walk-forward 2024-26 | -0.9 to -1.4 pips | fail |
| H1 LightGBM, 24-bar hold, 2x ATR stop (USDJPY) | holdout 2010-23 | -1.26 pips | fail |
| LightGBM, p > 0.40, TP 4x ATR / SL 2x ATR (USDJPY H1) | holdout 2010-23 | -0.07 R and -0.05 R (two labels) | fail |
| EMA21/50 crossover, TP 4x ATR / SL 2x ATR (USDJPY H1) | holdout 2010-23 | +0.03 R, 95% CI -0.04 to +0.09 | fail |
| Same EMA rule on fresh pairs (GOLD, GBPJPY, EURJPY H1) | holdout 2010-23 | -0.06 R, +0.01 R, +0.02 R | fail (0 of 3) |
| Breakouts, calendar effects, daily trend following, cheaper-cost what-ifs | 2010-26 | none cleared its pass rule | fail |

More than 300 variants were tried. The model's raw edge is a fraction of a pip, smaller than the 1-1.3 pip spread.
With TP 4x ATR / SL 2x ATR, the win rate stayed near 31%, below the 33% a random entry needs just to break even.
Breakeven moves, trailing stops and bandit-tuned thresholds change how wins and losses are spread out; they cannot
create an edge when the entries have none.

### 3. Lessons

- Join higher-timeframe bars at their close time, never their open time.
- An AUC above about 0.6 on M5 forex direction is a sign of a leak, not a good model.
- A backtest is only evidence if its data was not used to pick the strategy. Keep a holdout and write the pass rule
  before running.
- Charge the spread where it is actually paid: shorts exit on the ask, and spreads widen at the daily rollover.
- A few weeks of demo profit cannot confirm an edge; the noise in 40 trades is far larger than a one-pip edge.

## Reproducing the research

Data: export bars from MT5 (View > Symbols > Bars > Export Bars, after raising Tools > Options > Charts >
Max bars in chart), then convert:

```bash
source .venv/bin/activate
python scripts/import_mt5_export.py data/raw_export data/historical_data          # 2024+ exports
python scripts/import_mt5_export.py data/raw_export/old data/historical_old       # 2010+ exports
python scripts/walkforward.py --symbols EURUSD GBPUSD                              # M5 walk-forward
python scripts/holdout_h1.py --symbol USDJPY                                       # H1 ML holdout
python scripts/resad_recipe.py --mode holdout                                      # ATR barrier exits
python scripts/resad_trend.py                                                      # EMA21/50 crossover
./scripts/check.sh                                                                 # flake8 + pytest
```

`data/` and `results/` are git-ignored. `MetaTrader5` has no Linux wheel; install requirements with
`grep -v '^MetaTrader5' requirements.txt` and use `data_source: csv`.

## Disclaimer

Educational code. Trading leveraged CFDs loses money for most retail accounts. Nothing here is financial advice.
