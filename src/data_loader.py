from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
UNIVERSE_DIR = DATA_DIR / "universe"


def load_universe(filename: str = "equity_universe.csv") -> pd.Series:
    """Load a ticker-to-sector mapping from data/universe/<filename>."""
    path = UNIVERSE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing universe file: {path}. Add a CSV with columns 'ticker' and 'sector'."
        )

    universe = pd.read_csv(path)
    required = {"ticker", "sector"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"Universe file is missing required columns: {sorted(missing)}")

    universe = universe.dropna(subset=["ticker", "sector"]).copy()
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["sector"] = universe["sector"].astype(str).str.strip()
    universe = universe.drop_duplicates("ticker")
    return universe.set_index("ticker")["sector"]


def load_fred(series: str, refresh: bool = False) -> pd.Series:
    """Download a FRED series and cache it locally as CSV."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{series}.csv"

    if cache_path.exists() and not refresh:
        table = pd.read_csv(cache_path, index_col=0, parse_dates=True, na_values=".")
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        table = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
        table.to_csv(cache_path)

    values = table.iloc[:, 0].rename(series)
    values.index = pd.to_datetime(values.index).tz_localize(None).normalize()
    return values.sort_index()


def load_adjusted_close(
    tickers: Iterable[str],
    start: str,
    end: str,
    market: str = "SPY",
    refresh: bool = False,
    cache_name: str = "adjusted_close.csv",
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance and cache a wide CSV."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name

    requested = list(dict.fromkeys([*tickers, market]))

    if cache_path.exists() and not refresh:
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if set(requested).issubset(prices.columns):
            prices = prices.loc[(prices.index >= pd.Timestamp(start)) & (prices.index < pd.Timestamp(end)), requested]
            return prices.sort_index().dropna(axis=1, how="all")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required to download equity prices. Install requirements.txt first.") from exc

    downloaded = yf.download(
        requested,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no price data.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        prices = downloaded["Close"].copy()
    else:
        # yfinance can return a single-level frame for one ticker.
        prices = downloaded[["Close"]].rename(columns={"Close": requested[0]})

    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices.sort_index().dropna(axis=1, how="all")
    prices.to_csv(cache_path)
    return prices
