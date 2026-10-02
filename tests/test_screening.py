import numpy as np
import pandas as pd

from src.screen_yield_sensitivity import fit, same_sign


def test_fit_recovers_negative_yield_sensitivity():
    rng = np.random.default_rng(7)
    n = 400
    market = rng.normal(0, 1, n)
    yield_change = rng.normal(0, 4, n)
    noise = rng.normal(0, 0.05, n)
    stock = 0.9 * market - 0.08 * yield_change + noise

    frame = pd.DataFrame({"SPY": market, "10Y": yield_change, "TEST": stock})
    model = fit(frame, "TEST", ["SPY", "10Y"], min_obs=100, hac_lags=2)

    assert model is not None
    assert model.params["10Y"] < 0
    assert abs(model.params["10Y"] + 0.08) < 0.02
    assert same_sign(model, "10Y", -0.08)
