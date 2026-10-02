# Equity universe input

`equity_universe.csv` is the explicit stock universe used by the research pipeline.

The current file contains **218 unique U.S. equity tickers across all 11 sectors**. Ten sectors contain 20 names each and Communication Services contains 18. The required columns are:

```csv
ticker,sector
DHI,Consumer Discretionary
...
```

The supplied file also includes `company`, `sub_industry`, and `cap_rank` as descriptive metadata. The analysis currently uses only `ticker` and `sector`.

Keeping the universe explicit matters because silently rebuilding a historical sample from today's index membership can introduce survivorship and selection bias. The repository therefore treats the universe as a research input rather than claiming it is a point-in-time historical S&P 500 membership file.
