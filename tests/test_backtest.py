import pandas as pd

from src.walk_forward_backtest import most_negative, transaction_costs


def test_most_negative_uses_conservative_ci_edge():
    exposures = pd.DataFrame(
        {
            "yield_beta": [-0.5, -0.3, -0.7],
            "ci_high": [-0.2, -0.1, -0.4],
            "spy_beta": [1.0, 1.0, 1.0],
        },
        index=["A", "B", "C"],
    )
    selected = most_negative(exposures)
    assert selected.index[0] == "C"


def test_transaction_costs_use_absolute_hedge_notional():
    results = pd.DataFrame(
        {
            "spy_beta": [-1.2, -1.2],
            "names_replaced": [0.0, 0.0],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="W"),
    )
    position = pd.Series([1.0, 1.0], index=results.index)
    costs = transaction_costs(results, position)
    assert costs.iloc[0] > 0
    assert costs.iloc[1] == 0
