"""
prepare_data.py
----------------
Generates historical_reliance.csv — the file the Streamlit app's Market
Overview tab reads for candlesticks, moving averages, RSI, volatility, etc.

Data source: Yahoo Finance (via the `yfinance` package), ticker RELIANCE.NS.
This gives real historical Open/High/Low/Close/Volume for Reliance
Industries on the NSE.

IMPORTANT — about the `oi` (Open Interest) column:
Open Interest is a derivatives/F&O concept (futures & options), not an
equity price-history concept. Free sources like Yahoo Finance do not
provide it for cash-market equity data. This script SYNTHESIZES a
plausible `oi` series (loosely correlated with volume, with its own
random walk) purely so the app's charts and existing model pipeline have
a value to work with. Treat any `oi`-based chart or prediction as
illustrative, not real market open interest. If you have an actual NSE
F&O open-interest feed, replace this column with real data before relying
on it for anything serious.

Usage:
    pip install yfinance pandas numpy
    python prepare_data.py [--years 3] [--out historical_reliance.csv]

Produces (next to wherever you run it, or --out path):
    historical_reliance.csv   with columns:
        date, symbol, open, high, low, close, volume, oi
"""

import argparse
import sys

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print(
        "The `yfinance` package is required. Install it with:\n"
        "    pip install yfinance\n"
        "then re-run this script.",
        file=sys.stderr,
    )
    sys.exit(1)

TICKER = "RELIANCE.NS"
SYMBOL_LABEL = "RELIANCE"


def fetch_history(years: int) -> pd.DataFrame:
    print(f"Fetching {years} year(s) of daily history for {TICKER} from Yahoo Finance...")
    tk = yf.Ticker(TICKER)
    df = tk.history(period=f"{years}y", interval="1d", auto_adjust=False)

    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned for {TICKER}. Check your internet connection, "
            "or that the ticker is still valid on Yahoo Finance."
        )

    df = df.reset_index()
    # yfinance returns a "Date" column (sometimes "Datetime"); normalise it.
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={
        date_col: "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["symbol"] = SYMBOL_LABEL
    df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return df


def synthesize_open_interest(df: pd.DataFrame, seed: int = 7) -> pd.Series:
    """Create a plausible-looking OI series: starts at a round number,
    drifts with a mild random walk, and nudges up/down with volume so it
    isn't pure noise. This is NOT real open interest data — see module
    docstring."""
    rng = np.random.default_rng(seed)
    n = len(df)

    vol_z = (df["volume"] - df["volume"].mean()) / (df["volume"].std() + 1e-9)
    base = 40_000_000
    walk = np.cumsum(rng.normal(0, 350_000, n))
    volume_effect = vol_z.values * 800_000

    oi = base + walk + volume_effect
    oi = np.clip(oi, 5_000_000, None)  # keep it sane/positive
    return pd.Series(oi.round().astype(int), index=df.index)


def main():
    parser = argparse.ArgumentParser(description="Generate historical_reliance.csv")
    parser.add_argument("--years", type=int, default=3, help="Years of history to fetch (default: 3)")
    parser.add_argument("--out", type=str, default="historical_reliance.csv", help="Output CSV path")
    args = parser.parse_args()

    df = fetch_history(args.years)
    df["oi"] = synthesize_open_interest(df)

    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows to {args.out}")
    print(df.tail(5).to_string(index=False))
    print(
        "\nNote: the 'oi' column is synthetic (see the script's docstring) — "
        "everything else (open/high/low/close/volume) is real historical "
        "NSE data for RELIANCE via Yahoo Finance."
    )


if __name__ == "__main__":
    main()
