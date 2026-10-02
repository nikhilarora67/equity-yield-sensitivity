# Equity Sensitivity to Treasury Yield Changes

A reproducible research project that studies which U.S. equities exhibit persistent sensitivity to changes in Treasury yields after controlling for broad equity-market exposure, then tests whether those exposures can be used in a walk-forward portfolio construction exercise.

The project is organized as two stages:

1. **Cross-sectional screening and validation** — estimate stock-level rate sensitivities across the 5Y, 10Y, and 30Y Treasury maturities, rank candidates using training data only, and test whether the relationships remain stable out of sample.
2. **Walk-forward strategy diagnostic** — re-estimate 10Y sensitivities using only past data, impose a one-week gap, hold the selected basket for a quarter, hedge broad equity beta with SPY, and evaluate realized performance after a simple transaction-cost assumption.

This repository is an analytical research project, not an investment recommendation.


## Latest run: key findings

The latest reproducible run used daily data through **September 16, 2026** for the screening stage and weekly data through **September 18, 2026** for the walk-forward diagnostic.

The screening stage found economically interpretable and out-of-sample-stable rate exposures rather than a uniformly successful set of candidates. Among the strongest validated negative exposures:

| Stock | Sector | Yield | Estimated +10bp effect | Holdout added R² |
|---|---|---:|---:|---:|
| DHI | Consumer Discretionary | 30Y | -1.23% | 0.178 |
| SHW | Materials | 30Y | -0.70% | 0.102 |
| NEM | Materials | 5Y | -0.74% | 0.078 |
| LOW | Consumer Discretionary | 30Y | -0.61% | 0.073 |

Positive sensitivity was concentrated heavily in energy names, including HAL, FANG, SLB, VLO, COP, EOG, and BKR. Several stocks that looked strong in-sample retained the expected sign in the holdout but failed the stricter predictive check because the yield factor did not improve holdout error versus the market-only model.

The walk-forward basket retained meaningful rate exposure out of sample, but the unconditional long strategy was not profitable:

| Metric | Result |
|---|---:|
| Predicted +10bp effect | -1.16% |
| Realized +10bp effect | -0.72% |
| Weeks moving opposite to yields | 65.1% |
| Correlation with SPY before hedge | 0.71 |
| Correlation with SPY after hedge | 0.24 |
| Realized residual SPY beta | 0.24 |
| Annualized arithmetic mean return | -2.25% |
| Annualized volatility | 18.72% |
| Worst drawdown | -40.93% |

The main conclusion is deliberately narrower than “this is a profitable strategy”: **persistent yield exposure can be identified and validated out of sample, but identifying an exposure is not the same as forecasting the direction of the factor.** The walk-forward test is therefore used as a diagnostic of exposure persistence and implementation, not as evidence of a standalone trading edge.

## Research question

The central question is simple:

> After controlling for the broad equity market, which stocks have returns that are consistently sensitive to changes in Treasury yields, and do those estimated sensitivities remain useful when tested strictly out of sample?

This is not the same as asking which stocks merely correlate with rates. The screening model attempts to isolate a stock-specific yield relationship after accounting for contemporaneous SPY exposure.

## Model

For stock *i* at time *t*, the main screening regression is:

\[
r_{i,t}^{excess} = \alpha_i + \beta_{m,i} r_{SPY,t}^{excess} + \beta_{y,i}\Delta y_t + \varepsilon_{i,t}
\]

where:

- \(r_{i,t}^{excess}\) is the stock's excess return,
- \(r_{SPY,t}^{excess}\) is SPY's excess return,
- \(\Delta y_t\) is the change in the selected Treasury yield in basis points,
- \(\beta_{m,i}\) is the stock's market beta,
- \(\beta_{y,i}\) is the estimated return sensitivity to a one-basis-point yield move.

The reported `+10bp effect (%)` is simply \(10\beta_{y,i}\).

## Stage 1: screening and robustness checks

`src/screen_yield_sensitivity.py`:

- estimates separate 5Y, 10Y, and 30Y models for each stock;
- uses excess returns and SPY as a market control;
- uses HAC/Newey-West standard errors;
- compares the two-factor model with a market-only model through incremental \(R^2\);
- applies Benjamini-Yekutieli false-discovery-rate correction across the large set of hypothesis tests;
- ranks stocks using the training sample only;
- checks sign stability across multiple training subperiods;
- removes the largest stock-return and yield-move observations to test sensitivity to extremes;
- evaluates sign persistence and incremental predictive fit in a chronological holdout sample, measured as the reduction in squared error versus the market-only benchmark;
- compares estimates across daily, weekly, and monthly cadences.

The holdout begins on **March 16, 2026**. The goal is to keep selection and ranking decisions separate from the final validation period.

## Stage 2: walk-forward diagnostic

`src/walk_forward_backtest.py` focuses on the 10Y Treasury and asks a different question: what happens if the rate-sensitive basket is repeatedly selected using only information available at the time?

For each rebalance:

1. estimate stock exposures using the prior **104 weeks**;
2. leave a **one-week gap** between estimation and holding;
3. select the **10 stocks with the most negative conservative yield exposure**;
4. freeze those estimates for the next **13 weeks**;
5. equal-weight the stock basket;
6. hedge contemporaneous broad-equity exposure using the previously estimated SPY betas;
7. repeat the process through the sample.

The code separately labels a fixed housing basket selected with hindsight. That output is included only as a diagnostic benchmark and is **not** presented as a tradable strategy.

## Data

The code uses public data sources:

- **Equity prices:** Yahoo Finance through `yfinance`
- **Treasury yields:** Federal Reserve Economic Data (FRED)
  - DGS5 — 5-Year Treasury Constant Maturity Rate
  - DGS10 — 10-Year Treasury Constant Maturity Rate
  - DGS30 — 30-Year Treasury Constant Maturity Rate
  - DGS3MO — 3-Month Treasury Constant Maturity Rate, used as a cash-rate proxy
- **Market factor:** SPY

Market data are downloaded by the scripts and cached under `data/cache/`.

### Equity universe

The universe is an explicit input at:

`data/universe/equity_universe.csv`

The included file contains **218 unique tickers spanning all 11 sectors** (20 names in each sector except Communication Services, which has 18). It includes `ticker`, `company`, `sector`, `sub_industry`, and `cap_rank`; the research code uses `ticker` and `sector`.

Keeping the universe explicit matters because silently rebuilding a historical sample from today's index constituents can introduce survivorship bias. This file should therefore be understood as the supplied research universe, not as a claim of point-in-time historical S&P 500 membership.

## Repository structure

```text
equity-yield-sensitivity/
├── README.md
├── requirements.txt
├── run_all.py
├── src/
│   ├── data_loader.py
│   ├── screen_yield_sensitivity.py
│   └── walk_forward_backtest.py
├── data/
│   ├── cache/
│   └── universe/
│       └── equity_universe.csv
├── results/
│   ├── README.md
│   └── figures/
└── tests/
    ├── test_screening.py
    ├── test_backtest.py
    └── test_universe.py
```

## Running the project

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then install packages:

```bash
pip install -r requirements.txt
```

The intended research universe is already included at `data/universe/equity_universe.csv`. Run the tests from the repository root:

```bash
pytest
```

Equivalent:

```bash
python -m pytest
```

Run the screening stage:

```bash
python -m src.screen_yield_sensitivity
```

Run the walk-forward stage:

```bash
python -m src.walk_forward_backtest
```

Or run both:

```bash
python run_all.py
```

## Outputs

The screening stage writes:

- all stock/maturity sensitivity estimates;
- top negative and positive sensitivity tables;
- sector-level median effects;
- SPY's own yield sensitivity by cadence;
- one-factor and two-factor SPY betas;
- yield-effect estimates by cadence;
- a chart of the strongest negative training-selected sensitivities.

The walk-forward stage writes:

- summary statistics;
- yearly attribution;
- P&L summaries under several positioning rules;
- yearly P&L;
- basket composition at each rebalance;
- sector mix;
- cumulative-P&L figures.

## Interpretation

A negative yield beta means that, after controlling for SPY, the stock has historically tended to underperform when the relevant Treasury yield rises and outperform when it falls. That relationship is an estimated exposure, not proof of a causal mechanism.

The research therefore emphasizes robustness rather than the raw coefficient alone. A candidate is more credible when the sign is stable across subperiods, survives the removal of extreme observations, remains present in the holdout sample, and adds explanatory value beyond the market-only model.

## Important limitations

This project is intentionally transparent about what it does **not** establish.

- **No causal claim.** The regressions identify conditional relationships, not structural causality.
- **Universe dependence.** Results depend on the supplied 218-stock sector-balanced research universe and can be affected by survivorship or selection bias; it is not a point-in-time historical constituent file.
- **Daily/weekly data.** The study does not model intraday reaction functions or execution around macro releases.
- **Simplified transaction costs.** The backtest charges a fixed basis-point cost to estimated stock-basket and SPY-hedge turnover, including basket replacements, but does not model spread, market impact, borrow costs, or venue-specific execution.
- **Close-to-close alignment.** Public daily data cannot perfectly reproduce the information set or tradable price at every Treasury-data release.
- **Model instability.** Equity duration can change with fundamentals, sector composition, leverage, and the macro regime.
- **Omitted factors.** SPY plus one yield maturity is intentionally interpretable but incomplete; richer factor models could change the estimated yield coefficient.
- **Hindsight benchmark.** The fixed housing basket is explicitly labeled as selected with hindsight and is not evidence of tradable performance.
- **Backtest is a diagnostic.** A relationship between yield exposure and subsequent returns does not by itself create a directional rates forecast.

## Possible extensions

The next useful extensions would be driven by better data rather than by adding complexity for its own sake:

- intraday prices around Treasury and macro releases;
- a richer equity factor model to separate rates exposure from industry/style exposures;
- Treasury futures or swaps instead of cash-yield changes for a directly tradable rates factor;
- realistic bid/ask, market-impact, and borrow-cost modeling;
- point-in-time index membership and fundamentals;
- dynamic exposure models that allow yield sensitivity to vary through time.

## Research principles

The project follows three rules throughout:

1. **Select on the past, evaluate on the future.**
2. **Label hindsight and non-tradable benchmarks explicitly.**
3. **Treat a statistically interesting relationship as a hypothesis to stress-test, not as a finished trading edge.**
