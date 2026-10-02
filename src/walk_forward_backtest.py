from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data_loader import ROOT, load_adjusted_close, load_fred, load_universe

MARKET = "SPY"
YIELD_SERIES = "DGS10"
RISK_FREE_SERIES = "DGS3MO"

START_DATE = "2017-01-01"
END_DATE = "2026-09-17"
CADENCE = "W-FRI"
ESTIMATION_WEEKS = 104
GAP_WEEKS = 1
REBALANCE_WEEKS = 13
BASKET_SIZE = 10
HAC_LAGS = 4

# Deliberately labeled as a hindsight benchmark, not a tradable strategy.
FIXED_HOUSING_BASKET = ["DHI", "LOW", "SHW", "PSA", "EXR"]

NOTIONAL = 1_000_000
COST_BPS = 5
TREND_WEEKS = 13

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_weekly_data(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    prices = load_adjusted_close(
        tickers,
        START_DATE,
        END_DATE,
        market=MARKET,
        refresh=refresh,
        cache_name="adjusted_close_backtest.csv",
    )
    yields = load_fred(YIELD_SERIES, refresh=refresh).reindex(prices.index)

    calendar_days = prices.index.to_series().diff().dt.days
    cash_return = (
        load_fred(RISK_FREE_SERIES, refresh=refresh).reindex(prices.index).ffill().shift(1)
        * calendar_days
        / 36500
    )
    cash_index = (1 + cash_return.fillna(0)).cumprod()

    bond_market_open = yields.notna()
    prices = prices[bond_market_open].resample(CADENCE).last()
    yields = yields[bond_market_open].resample(CADENCE).last()
    cash_index = cash_index[bond_market_open].resample(CADENCE).last()

    weekly = 100 * prices.pct_change(fill_method=None).sub(cash_index.pct_change(), axis=0)
    weekly["yield_change"] = 100 * yields.diff()
    return weekly.dropna(subset=[MARKET, "yield_change"])


def estimate_exposures(window: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    design = sm.add_constant(window[[MARKET, "yield_change"]])
    rows = {}
    for ticker in tickers:
        if ticker not in window.columns or window[ticker].isna().any():
            continue
        model = sm.OLS(window[ticker], design).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}, use_t=True
        )
        rows[ticker] = {
            "spy_beta": model.params[MARKET],
            "yield_beta": model.params["yield_change"],
            "ci_high": model.conf_int().loc["yield_change", 1],
        }
    return pd.DataFrame(rows).T


def most_negative(exposures: pd.DataFrame) -> pd.DataFrame:
    return exposures.nsmallest(BASKET_SIZE, "ci_high")


def keep_all(exposures: pd.DataFrame) -> pd.DataFrame:
    return exposures


def walk_forward(weekly: pd.DataFrame, tickers: list[str], choose_basket):
    periods = []
    baskets = {}
    previous_names: set[str] = set()

    for start in range(ESTIMATION_WEEKS + GAP_WEEKS, len(weekly), REBALANCE_WEEKS):
        window = weekly.iloc[start - GAP_WEEKS - ESTIMATION_WEEKS : start - GAP_WEEKS]
        holding = weekly.iloc[start : start + REBALANCE_WEEKS]
        basket = choose_basket(estimate_exposures(window, tickers))
        if basket.empty or holding.empty:
            continue

        stock_returns = holding[basket.index]
        hedged_returns = stock_returns - np.outer(holding[MARKET], basket["spy_beta"])
        period = pd.DataFrame(
            {
                "hedged": hedged_returns.mean(axis=1),
                "unhedged": stock_returns.mean(axis=1),
                "predicted": holding["yield_change"] * basket["yield_beta"].mean(),
                "yield_beta": basket["yield_beta"].mean(),
                "spy_beta": basket["spy_beta"].mean(),
                "yield_change": holding["yield_change"],
                "spy": holding[MARKET],
                "quarter": len(periods),
                "names_replaced": 0.0,
            }
        )
        replacement_share = 0.0
        if previous_names:
            replacement_share = len(set(basket.index) - previous_names) / len(basket)
        period.iloc[0, period.columns.get_loc("names_replaced")] = replacement_share
        previous_names = set(basket.index)
        baskets[window.index[-1].date()] = list(basket.index)
        periods.append(period)

    if not periods:
        raise RuntimeError("Walk-forward procedure produced no holding periods.")
    return pd.concat(periods), baskets


def summarize(results: pd.DataFrame) -> pd.Series:
    hedged = results["hedged"]
    growth = (1 + hedged / 100).cumprod()
    realized = sm.OLS(hedged, sm.add_constant(results[["spy", "yield_change"]])).fit()
    yields_fell = results["yield_change"] < 0
    yields_rose = results["yield_change"] > 0
    moved_opposite = (hedged * results["yield_change"] < 0)[yields_fell | yields_rose]

    return pd.Series(
        {
            "First week": results.index[0].date(),
            "Last week": results.index[-1].date(),
            "Weeks": len(results),
            "Annualized mean return, long hedged basket (%)": 52 * hedged.mean(),
            "Annualized volatility (%)": np.sqrt(52) * hedged.std(),
            "Return / volatility": np.sqrt(52) * hedged.mean() / hedged.std(),
            "Worst drawdown (%)": 100 * (growth / growth.cummax() - 1).min(),
            "Correlation with SPY, unhedged": results["unhedged"].corr(results["spy"]),
            "Correlation with SPY, hedged": hedged.corr(results["spy"]),
            "Leftover SPY beta after hedge": realized.params["spy"],
            "Predicted +10bp effect (%)": 10 * results["yield_beta"].mean(),
            "Realized +10bp effect (%)": 10 * realized.params["yield_change"],
            "Share of hedged variance explained by predicted": hedged.corr(results["predicted"]) ** 2,
            "Mean weekly return when yields fell (%)": hedged[yields_fell].mean(),
            "Mean weekly return when yields rose (%)": hedged[yields_rose].mean(),
            "Weeks moving opposite to yields (%)": 100 * moved_opposite.mean(),
        }
    )


def yearly_attribution(results: pd.DataFrame) -> pd.DataFrame:
    table = results.groupby(results.index.year)[["yield_change", "spy", "hedged", "predicted"]].sum()
    table["other"] = table["hedged"] - table["predicted"]
    return table.rename(
        columns={
            "yield_change": "Yield change (bp)",
            "spy": "SPY excess (%)",
            "hedged": "Long hedged basket (%)",
            "predicted": "Predicted from yields (%)",
            "other": "Everything else (%)",
        }
    )


def positions_by_rule(results: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    past_trend = weekly["yield_change"].rolling(TREND_WEEKS).sum().shift(1).reindex(results.index)
    quarter_move = results.groupby("quarter")["yield_change"].transform("sum")
    return pd.DataFrame(
        {
            "Always long (view: yields fall)": 1.0,
            f"Trend rule (short if yields rose over prior {TREND_WEEKS} weeks)": -np.sign(past_trend),
            "CEILING: knows each quarter's yield direction in advance (not tradable)": -np.sign(quarter_move),
        },
        index=results.index,
    )


def transaction_costs(results: pd.DataFrame, position: pd.Series) -> pd.Series:
    """Approximate one-way trading costs in dollars.

    The model charges for gross basket turnover, the SPY hedge, and basket-name
    replacement. It is intentionally simple: it does not model spread, market
    impact, borrow, or intraday execution.
    """
    position_change = position.diff().fillna(position).abs()

    # Stock-basket turnover from entering, exiting, or flipping the strategy.
    stock_turnover = position_change.copy()

    # If the strategy stays on in the same direction through a rebalance, charge
    # for selling removed names and buying their replacements. A full entry/exit
    # or sign flip already turns over the whole basket and should not be double-counted.
    same_active_position = (position.abs() > 0) & (position_change == 0)
    replacement_turnover = (
        2 * results["names_replaced"] * same_active_position.astype(float)
    )

    # The SPY hedge target is -position * beta. This captures turnover caused by
    # strategy sign changes as well as changes in the estimated hedge ratio.
    target_spy_hedge = -position * results["spy_beta"]
    hedge_turnover = target_spy_hedge.diff().fillna(target_spy_hedge).abs()

    traded_multiple = stock_turnover + replacement_turnover + hedge_turnover
    return NOTIONAL * traded_multiple * COST_BPS / 10_000


def pnl_report(results: pd.DataFrame, weekly: pd.DataFrame):
    positions = positions_by_rule(results, weekly)
    summary = {}
    yearly = {}
    series = {}

    for rule, position in positions.items():
        gross = NOTIONAL * position * results["hedged"] / 100
        costs = transaction_costs(results, position)
        net = gross - costs
        cumulative = net.cumsum()
        by_quarter = net.groupby(results["quarter"]).sum()
        years = len(net) / 52
        summary[rule] = {
            "Total P&L ($)": net.sum(),
            "Of which trading costs ($)": -costs.sum(),
            "Average per year ($)": net.sum() / years,
            "Return on notional per year (%)": 100 * net.sum() / years / NOTIONAL,
            "Volatility per year (%)": 100 * np.sqrt(52) * net.std() / NOTIONAL,
            "Return / volatility": np.sqrt(52) * net.mean() / net.std(),
            "Worst drawdown ($)": (cumulative - cumulative.cummax()).min(),
            "Profitable quarters (%)": 100 * (by_quarter > 0).mean(),
        }
        yearly[rule] = net.groupby(net.index.year).sum()
        series[rule] = cumulative

    return pd.DataFrame(summary), pd.DataFrame(yearly), pd.DataFrame(series)


def baskets_table(baskets: dict) -> pd.DataFrame:
    rows = []
    for rebalance_date, names in baskets.items():
        for rank, ticker in enumerate(names, start=1):
            rows.append({"estimation_end": rebalance_date, "rank": rank, "ticker": ticker})
    return pd.DataFrame(rows)


def save_report(prefix: str, results: pd.DataFrame, baskets: dict, sectors: pd.Series, weekly: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = summarize(results)
    attribution = yearly_attribution(results)
    pnl_summary, yearly_pnl, cumulative = pnl_report(results, weekly)
    basket_history = baskets_table(baskets)

    summary.to_frame("value").to_csv(RESULTS_DIR / f"{prefix}_summary.csv")
    attribution.to_csv(RESULTS_DIR / f"{prefix}_yearly_attribution.csv")
    pnl_summary.to_csv(RESULTS_DIR / f"{prefix}_pnl_summary.csv")
    yearly_pnl.to_csv(RESULTS_DIR / f"{prefix}_yearly_pnl.csv")
    basket_history.to_csv(RESULTS_DIR / f"{prefix}_basket_history.csv", index=False)

    slots = pd.Series([ticker for names in baskets.values() for ticker in names], dtype="object")
    if not slots.empty:
        sector_mix = 100 * slots.map(sectors).value_counts(normalize=True)
        sector_mix.rename("share_of_slots_pct").to_csv(RESULTS_DIR / f"{prefix}_sector_mix.csv")

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        for column in cumulative.columns:
            ax.plot(cumulative.index, cumulative[column], label=column)
        ax.set_ylabel("Cumulative net P&L ($)")
        ax.set_title(prefix.replace("_", " ").title())
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{prefix}_cumulative_pnl.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"Figure generation skipped: {exc}")


def main(refresh: bool = False) -> None:
    sectors = load_universe()
    weekly = load_weekly_data(sectors.index.tolist(), refresh=refresh)
    stocks = [ticker for ticker in sectors.index if ticker in weekly.columns]

    pd.set_option("display.width", 250)
    print(
        f"Weekly data: {weekly.index[0].date()} to {weekly.index[-1].date()} "
        f"({len(weekly)} weeks, {len(stocks)} stocks)"
    )
    print(
        f"Each quarter: estimate on {ESTIMATION_WEEKS} weeks, wait {GAP_WEEKS} week, "
        f"then hold for {REBALANCE_WEEKS} weeks with estimates frozen."
    )

    results, baskets = walk_forward(weekly, stocks, most_negative)
    save_report("walk_forward", results, baskets, sectors, weekly)
    print("\nWALK-FORWARD SUMMARY\n")
    print(summarize(results).to_string())

    available_fixed = [ticker for ticker in FIXED_HOUSING_BASKET if ticker in stocks]
    if len(available_fixed) >= 2:
        fixed_results, fixed_baskets = walk_forward(weekly, available_fixed, keep_all)
        save_report("hindsight_fixed_housing_benchmark", fixed_results, fixed_baskets, sectors, weekly)
        print(
            "\nSaved a clearly labeled hindsight benchmark as a diagnostic only; "
            "it is not presented as a tradable strategy."
        )

    print(f"\nSaved outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
