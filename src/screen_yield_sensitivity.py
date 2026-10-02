from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .data_loader import ROOT, load_adjusted_close, load_fred, load_universe

MARKET = "SPY"
YIELD_SERIES = {"5Y": "DGS5", "10Y": "DGS10", "30Y": "DGS30"}
RISK_FREE_SERIES = "DGS3MO"

START_DATE = "2022-01-01"
END_DATE = "2026-09-17"
TEST_START = "2026-03-16"

SHORTLIST_SIZE = 10
HAC_LAGS = 5
MIN_OBS_TO_FIT = 126
MIN_OBS_HOLDOUT = 60
MIN_OBS_TO_RANK = 504
MIN_COVERAGE = 0.90
FDR_LEVEL = 0.05
EXTREME_DAYS = 5
CADENCES = {
    "daily": {"rule": None, "hac_lags": 5, "min_obs": 126},
    "weekly": {"rule": "W-FRI", "hac_lags": 4, "min_obs": 52},
    "monthly": {"rule": "ME", "hac_lags": 3, "min_obs": 24},
}

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def daily_risk_free(prices: pd.DataFrame, refresh: bool = False) -> pd.Series:
    calendar_days = prices.index.to_series().diff().dt.days
    annual_rate = load_fred(RISK_FREE_SERIES, refresh=refresh).reindex(prices.index).ffill().shift(1)
    return annual_rate * calendar_days / 365


def build_model_data(prices: pd.DataFrame, rule: str | None = None, refresh: bool = False) -> pd.DataFrame:
    cash_index = (1 + daily_risk_free(prices, refresh=refresh).fillna(0) / 100).cumprod()
    yields = pd.concat(
        [load_fred(series, refresh=refresh) for series in YIELD_SERIES.values()], axis=1
    ).reindex(prices.index)

    if rule:
        bond_market_open = yields.notna().all(axis=1)
        prices = prices[bond_market_open].resample(rule).last()
        yields = yields[bond_market_open].resample(rule).last()
        cash_index = cash_index[bond_market_open].resample(rule).last()

    returns = 100 * prices.pct_change(fill_method=None)
    risk_free = 100 * cash_index.pct_change()
    excess_returns = returns.sub(risk_free, axis=0)

    # FRED Treasury yields are reported in percentage points. Multiplying the
    # daily change by 100 converts percentage-point changes to basis points.
    yield_changes = 100 * yields.diff()
    yield_changes.columns = list(YIELD_SERIES)

    return pd.concat([excess_returns, yield_changes], axis=1).dropna(
        subset=[MARKET, *YIELD_SERIES]
    )


def fit(
    frame: pd.DataFrame,
    ticker: str,
    factors: list[str],
    min_obs: int = MIN_OBS_TO_FIT,
    hac_lags: int = HAC_LAGS,
):
    sample = frame[[ticker, *factors]].dropna()
    if len(sample) < min_obs:
        return None
    design = sm.add_constant(sample[factors], has_constant="add")
    return sm.OLS(sample[ticker], design).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}, use_t=True
    )


def screen(train: pd.DataFrame, stocks: list[str], sectors: pd.Series) -> pd.DataFrame:
    rows: list[dict] = []
    for ticker in stocks:
        market_only = fit(train, ticker, [MARKET])
        if market_only is None:
            continue
        for maturity in YIELD_SERIES:
            model = fit(train, ticker, [MARKET, maturity])
            if model is None:
                continue
            ci_low, ci_high = model.conf_int().loc[maturity]
            rows.append(
                {
                    "ticker": ticker,
                    "sector": sectors[ticker],
                    "maturity": maturity,
                    "n_obs": int(model.nobs),
                    "spy_beta": model.params[MARKET],
                    "yield_beta": model.params[maturity],
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": model.pvalues[maturity],
                    "r2": model.rsquared,
                    "added_r2": model.rsquared - market_only.rsquared,
                }
            )
    return pd.DataFrame(rows)


def rank(results: pd.DataFrame, n_train: int) -> pd.DataFrame:
    results = results.copy()
    valid_p = results["p_value"].notna()
    results["p_adjusted"] = np.nan
    if valid_p.any():
        results.loc[valid_p, "p_adjusted"] = multipletests(
            results.loc[valid_p, "p_value"], method="fdr_by"
        )[1]

    ci_excludes_zero = (results["ci_low"] > 0) | (results["ci_high"] < 0)
    ci_edge_nearest_zero = results[["ci_low", "ci_high"]].abs().min(axis=1)
    results["conservative_beta"] = np.where(ci_excludes_zero, ci_edge_nearest_zero, 0.0)
    results["abs_beta"] = results["yield_beta"].abs()
    results["passes_screen"] = (
        (results["n_obs"] >= MIN_OBS_TO_RANK)
        & (results["n_obs"] >= MIN_COVERAGE * n_train)
        & (results["p_adjusted"] <= FDR_LEVEL)
        & ci_excludes_zero
    )

    return results.sort_values(
        ["passes_screen", "conservative_beta", "abs_beta", "added_r2", "ticker", "maturity"],
        ascending=[False, False, False, False, True, True],
    )


def same_sign(model, maturity: str, reference_beta: float) -> bool:
    return bool(model is not None and model.params[maturity] * reference_beta > 0)


def validate(candidate: pd.Series, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    ticker = candidate["ticker"]
    maturity = candidate["maturity"]
    beta = candidate["yield_beta"]
    factors = [MARKET, maturity]

    midpoint = train.index[len(train) // 2]
    windows = [train[train.index < midpoint], train[train.index >= midpoint], train.tail(252)]
    stable_sign = all(same_sign(fit(window, ticker, factors), maturity, beta) for window in windows)

    sample = train[[ticker, *factors]].dropna()
    survives_extremes = True
    for column in [ticker, maturity]:
        extreme_dates = sample[column].abs().nlargest(EXTREME_DAYS).index
        trimmed_model = fit(sample.drop(index=extreme_dates), ticker, factors)
        survives_extremes = survives_extremes and same_sign(trimmed_model, maturity, beta)

    holdout = test[[ticker, *factors]].dropna()
    holdout_model = fit(holdout, ticker, factors, MIN_OBS_HOLDOUT)
    holdout_added_r2 = np.nan
    if holdout_model is not None:
        two_factor = fit(train, ticker, factors)
        market_only = fit(train, ticker, [MARKET])
        actual = holdout[ticker]
        two_factor_errors = actual - two_factor.predict(
            sm.add_constant(holdout[factors], has_constant="add")
        )
        market_only_errors = actual - market_only.predict(
            sm.add_constant(holdout[[MARKET]], has_constant="add")
        )
        market_only_sse = (market_only_errors**2).sum()
        two_factor_sse = (two_factor_errors**2).sum()
        if market_only_sse > 0:
            # Incremental predictive fit versus the market-only benchmark.
            # Positive values mean the yield factor reduced holdout squared error.
            holdout_added_r2 = 1 - two_factor_sse / market_only_sse

    return {
        "stable_sign": stable_sign,
        "survives_extremes": survives_extremes,
        "holdout_sign": same_sign(holdout_model, maturity, beta),
        "holdout_added_r2": holdout_added_r2,
    }


def shortlist_table(shortlist: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    shortlist = shortlist.reset_index(drop=True)
    checks = pd.DataFrame(
        [validate(candidate, train, test) for _, candidate in shortlist.iterrows()]
    )
    table = pd.concat([shortlist, checks], axis=1)
    table["Rank"] = table.index + 1
    table["Screen"] = np.where(table["passes_screen"], "PASS", "EXPLORATORY")
    table["+10bp effect (%)"] = 10 * table["yield_beta"]
    table["Quant checks pass"] = (
        table["passes_screen"]
        & table["stable_sign"]
        & table["survives_extremes"]
        & table["holdout_sign"]
        & (table["holdout_added_r2"] > 0)
    )
    table = table.rename(
        columns={
            "ticker": "Stock",
            "sector": "Sector",
            "maturity": "Yield",
            "spy_beta": "SPY beta",
            "r2": "R2",
            "added_r2": "Added R2",
            "stable_sign": "Stable train sign",
            "holdout_sign": "Same test sign",
            "survives_extremes": "Extremes OK",
            "holdout_added_r2": "Holdout added R2",
        }
    )
    columns = [
        "Rank",
        "Stock",
        "Sector",
        "Yield",
        "Screen",
        "SPY beta",
        "+10bp effect (%)",
        "R2",
        "Added R2",
        "Stable train sign",
        "Same test sign",
        "Quant checks pass",
        "Extremes OK",
        "Holdout added R2",
    ]
    return table[columns].round(4)


def sector_summary(results: pd.DataFrame, best_per_stock: pd.DataFrame) -> pd.DataFrame:
    effects = results.assign(effect=10 * results["yield_beta"])
    medians = effects.pivot_table(index="sector", columns="maturity", values="effect", aggfunc="median")
    medians = medians[list(YIELD_SERIES)].add_prefix("Median +10bp effect, ")
    counts = best_per_stock.groupby("sector").agg(
        Stocks=("ticker", "size"), Passing=("passes_screen", "sum")
    )
    return counts.join(medians).sort_values(medians.columns[-1]).round(3)


def cadence_tables(prices: pd.DataFrame, shortlisted: pd.DataFrame, refresh: bool = False):
    training_prices = prices[prices.index < TEST_START]
    market_rows: dict = {}
    beta_rows: dict = {}
    effect_rows: dict = {}

    for cadence, settings in CADENCES.items():
        data = build_model_data(training_prices, settings["rule"], refresh=refresh)
        limits = {"min_obs": settings["min_obs"], "hac_lags": settings["hac_lags"]}

        for maturity in YIELD_SERIES:
            market_model = fit(data, MARKET, [maturity], **limits)
            if market_model is None:
                continue
            row = market_rows.setdefault(maturity, {})
            row[f"+10bp effect on SPY, {cadence}"] = 10 * market_model.params[maturity]
            row[f"Correlation, {cadence}"] = data[MARKET].corr(data[maturity])

        for ticker, maturity in shortlisted["maturity"].items():
            one_factor = fit(data, ticker, [MARKET], **limits)
            two_factor = fit(data, ticker, [MARKET, maturity], **limits)
            if one_factor is None or two_factor is None:
                continue
            betas = beta_rows.setdefault(ticker, {})
            betas[f"1F {cadence}"] = one_factor.params[MARKET]
            betas[f"2F {cadence}"] = two_factor.params[MARKET]
            effects = effect_rows.setdefault(ticker, {})
            effects[f"Effect {cadence}"] = 10 * two_factor.params[maturity]
            effects[f"Std err {cadence}"] = 10 * two_factor.bse[maturity]

    cadences = list(CADENCES)
    market_columns = [f"+10bp effect on SPY, {c}" for c in cadences] + [
        f"Correlation, {c}" for c in cadences
    ]
    beta_columns = [f"1F {c}" for c in cadences] + [f"2F {c}" for c in cadences]
    effect_columns = [f"Effect {c}" for c in cadences] + [f"Std err {c}" for c in cadences]

    market_table = pd.DataFrame(market_rows).T.reindex(columns=market_columns)
    beta_table = pd.DataFrame(beta_rows).T.reindex(columns=beta_columns)
    effect_table = pd.DataFrame(effect_rows).T.reindex(columns=effect_columns)
    effect_table.insert(0, "Yield", shortlisted["maturity"])
    return market_table, beta_table, effect_table


def save_outputs(
    ranked: pd.DataFrame,
    sector_table: pd.DataFrame,
    negative_table: pd.DataFrame,
    positive_table: pd.DataFrame,
    market_table: pd.DataFrame,
    beta_table: pd.DataFrame,
    effect_table: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ranked.to_csv(RESULTS_DIR / "all_sensitivity_estimates.csv", index=False)
    sector_table.to_csv(RESULTS_DIR / "sector_summary.csv")
    negative_table.to_csv(RESULTS_DIR / "top_negative_sensitivity.csv", index=False)
    positive_table.to_csv(RESULTS_DIR / "top_positive_sensitivity.csv", index=False)
    market_table.to_csv(RESULTS_DIR / "spy_yield_sensitivity_by_cadence.csv")
    beta_table.to_csv(RESULTS_DIR / "spy_beta_by_cadence.csv")
    effect_table.to_csv(RESULTS_DIR / "yield_effect_by_cadence.csv")

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        chart = negative_table.head(10).copy()
        if not chart.empty:
            labels = chart["Stock"] + " (" + chart["Yield"] + ")"
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(labels[::-1], chart["+10bp effect (%)"][::-1])
            ax.set_xlabel("Estimated excess-return effect of a +10bp yield move (%)")
            ax.set_title("Most Negative Yield Sensitivities (training-selected)")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "top_negative_sensitivity.png", dpi=180)
            plt.close(fig)
    except Exception as exc:  # plotting is helpful but should not break the research run
        print(f"Figure generation skipped: {exc}")


def main(refresh: bool = False) -> None:
    sectors = load_universe()
    prices = load_adjusted_close(
        sectors.index.tolist(), START_DATE, END_DATE, market=MARKET, refresh=refresh
    )
    stocks = [ticker for ticker in sectors.index if ticker in prices.columns]
    data = build_model_data(prices, refresh=refresh)
    train = data[data.index < TEST_START]
    test = data[data.index >= TEST_START]

    results = screen(train, stocks, sectors)
    ranked = rank(results, len(train))
    best_per_stock = ranked.drop_duplicates("ticker")
    most_negative = best_per_stock[best_per_stock["yield_beta"] < 0].head(SHORTLIST_SIZE)
    most_positive = best_per_stock[best_per_stock["yield_beta"] > 0].head(SHORTLIST_SIZE)

    negative_table = shortlist_table(most_negative, train, test)
    positive_table = shortlist_table(most_positive, train, test)
    sector_table = sector_summary(results, best_per_stock)

    shortlisted = pd.concat([most_negative, most_positive]).set_index("ticker")
    market_table, beta_table, effect_table = cadence_tables(prices, shortlisted, refresh=refresh)

    save_outputs(
        ranked,
        sector_table,
        negative_table,
        positive_table,
        market_table,
        beta_table,
        effect_table,
    )

    pd.set_option("display.width", 250)
    print(f"Universe: {len(sectors)} stocks requested, {len(best_per_stock)} with enough history to model")
    print(f"Training: {train.index.min().date()} to {train.index.max().date()} ({len(train)} rows)")
    print(f"Holdout:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} rows)")
    print("\nMOST NEGATIVE YIELD SENSITIVITY\n")
    print(negative_table.to_string(index=False))
    print("\nMOST POSITIVE YIELD SENSITIVITY\n")
    print(positive_table.to_string(index=False))
    print(f"\nSaved outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
