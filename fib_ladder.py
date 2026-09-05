#!/usr/bin/env python3
"""
fib_ladder.py — reproduce the "Guru Prasad Academy"-style stock report for any ticker.

    python fib_ladder.py QCOM
    python fib_ladder.py QCOM CAKE NEM          # several at once
    python fib_ladder.py QCOM --high 195 --low 142   # override the swing anchors
    python fib_ladder.py --watchlist tickers.txt --csv out.csv

How it works
  1. Downloads prices (yfinance) on two timeframes:
        long  = 6 years of WEEKLY bars, 25% ZigZag  -> multi-year base (CAKE-style)
        short = 2 years of DAILY bars,   8% ZigZag  -> recent leg       (QCOM-style)
     The broker's reports use either, chosen by eye, so both ladders are shown.
  2. In each, takes the most recent completed DOWN leg: pivot high -> pivot low.
     That leg is the Fibonacci anchor: low = 0, high = 1.0, D = high - low.
  3. (Long-term only) if the swing high is below the current price, the anchor is
     the ceiling price broke out of - exactly the CAKE case (BOA 82 < CMP 109).
  4. Builds the ladder:  SL2 = 0D, SL1 = 0.333D, BOA = 1.0D,
                         T1 = 2.0D, T2 = 3.0D, T3 = 4.618D
  5. Prints the report. Prices rounded to whole dollars like the originals.

The ladder is a reverse-engineered *pattern* fitted to published reports. It is
not the academy's stated method and has no demonstrated predictive value.
This is data analysis, not financial advice.

Install once:  pip install yfinance pandas
"""

import argparse
import csv
import math
import sys
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("pip install pandas yfinance")

# ---------------------------------------------------------------------------
# Model parameters (edit here if you want different multipliers)
# ---------------------------------------------------------------------------
MULTIPLIERS = {
    "SL2 (deep stop / 0 line)": 0.0,
    "SL1 (first stop)":         1 / 3,
    "BOA (entry / 1.0 line)":   1.0,
    "Target 1":                 2.0,
    "Target 2":                 3.0,
    "Target 3":                 4.618,
}
ZIGZAG_PCT = None      # None = use the timeframe default in TIMEFRAMES
LOOKBACK_DAYS = 320    # roughly 15 months of trading days


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
TIMEFRAMES = {
    # name: (yfinance period, interval, zigzag reversal %)
    "long":  ("6y", "1wk", 0.25),   # multi-year base, weekly bars  (CAKE-style)
    "short": ("2y", "1d",  0.08),   # recent daily swing            (QCOM-style)
}


def fetch_prices(ticker: str, timeframe: str = "long") -> pd.DataFrame:
    import yfinance as yf
    period, interval, _ = TIMEFRAMES[timeframe]
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                     progress=False)
    if df.empty:
        raise ValueError(f"no price data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):        # newer yfinance layout
        df.columns = df.columns.get_level_values(0)
    df = df[["High", "Low", "Close"]].dropna()
    if interval == "1d":
        df = df.tail(LOOKBACK_DAYS)
    return df


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------
def zigzag_pivots(df: pd.DataFrame, pct: float = ZIGZAG_PCT):
    """Return list of (date, price, 'H'|'L') pivots using a % reversal filter."""
    highs, lows, idx = df["High"].values, df["Low"].values, df.index
    pivots = []
    trend = None                    # 'up' or 'down'
    ext_i = 0                       # index of current extreme
    for i in range(1, len(df)):
        if trend in (None, "up"):
            if highs[i] >= highs[ext_i]:
                ext_i = i
            elif lows[i] <= highs[ext_i] * (1 - pct):
                pivots.append((idx[ext_i], float(highs[ext_i]), "H"))
                trend, ext_i = "down", i
        if trend == "down":
            if lows[i] <= lows[ext_i]:
                ext_i = i
            elif highs[i] >= lows[ext_i] * (1 + pct):
                pivots.append((idx[ext_i], float(lows[ext_i]), "L"))
                trend, ext_i = "up", i
        if trend is None:
            # decide initial direction from first meaningful move
            if lows[i] <= highs[0] * (1 - pct):
                trend, ext_i = "down", i
                pivots.append((idx[0], float(highs[0]), "H"))
            elif highs[i] >= lows[0] * (1 + pct):
                trend, ext_i = "up", i
                pivots.append((idx[0], float(lows[0]), "L"))
    # the current, unconfirmed extreme
    pivots.append((idx[ext_i], float(highs[ext_i] if trend == "up" else lows[ext_i]),
                   "H" if trend == "up" else "L"))
    return pivots


def breakout_base(df: pd.DataFrame, run_pct: float = 0.05, ceiling_years: int = 2,
                  base_years: int = 2):
    """
    Long-term anchor for a stock that has broken out of a base
    (the CAKE case: BOA 82 = ceiling before the July surge, SL2 38 = base low).

    Run start = the last ZigZag pivot low (run_pct, weekly) before the recent peak.
    Ceiling   = highest high in the `ceiling_years` up to that pivot low, i.e. the
                level price had to clear to break out.
    Base low  = lowest low in the `base_years` before the ceiling bar.
    Only used when today's close is above the ceiling.
    Returns (high_date, ceiling, low_date, base_low) or None.
    """
    if len(df) < 60:
        return None
    cmp_ = float(df["Close"].iloc[-1])
    peak_i = df["High"].idxmax()
    piv = [p for p in zigzag_pivots(df, run_pct) if p[2] == "L" and p[0] < peak_i]
    if not piv:
        return None
    run_start = piv[-1][0]
    win = df.loc[(df.index >= run_start - pd.Timedelta(days=365 * ceiling_years)) &
                 (df.index <= run_start)]
    if win.empty:
        return None
    ceiling_i = win["High"].idxmax()
    ceiling = float(win.loc[ceiling_i, "High"])
    if cmp_ <= ceiling:
        return None                                    # still inside the base
    base = df.loc[(df.index >= ceiling_i - pd.Timedelta(days=365 * base_years)) &
                  (df.index <= ceiling_i)]
    low_i = base["Low"].idxmin()
    return ceiling_i, ceiling, low_i, float(base.loc[low_i, "Low"])


def last_down_leg(pivots):
    """Most recent H -> L pair. Returns (high_date, high, low_date, low)."""
    for k in range(len(pivots) - 1, 0, -1):
        if pivots[k][2] == "L" and pivots[k - 1][2] == "H":
            return pivots[k - 1][0], pivots[k - 1][1], pivots[k][0], pivots[k][1]
    raise ValueError("could not find a completed high->low swing; "
                     "use --high/--low to set anchors manually")


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------
def build_ladder(high: float, low: float) -> dict:
    d = high - low
    return {name: low + m * d for name, m in MULTIPLIERS.items()}


def holding_period(d_pct: float) -> str:
    if d_pct < 0.20:
        return "8-12 Months"
    if d_pct < 0.45:
        return "12-18 Months"
    return "12-24 Months"


def analyse(ticker: str, high=None, low=None, timeframe: str = "long") -> dict:
    df = fetch_prices(ticker, timeframe)
    cmp_ = float(df["Close"].iloc[-1])
    if high is None or low is None:
        bb = breakout_base(df) if timeframe == "long" else None
        if bb:
            hd, h, ld, l = bb
            anchor_note = (f"breakout base: ceiling {h:.2f} ({hd.date()}), "
                           f"base low {l:.2f} ({ld.date()}); price now above the ceiling")
        else:
            pct = TIMEFRAMES[timeframe][2] if ZIGZAG_PCT is None else ZIGZAG_PCT
            hd, h, ld, l = last_down_leg(zigzag_pivots(df, pct))
            anchor_note = f"swing {hd.date()} high {h:.2f} -> {ld.date()} low {l:.2f}"
        high = high or h
        low = low or l
    else:
        anchor_note = "anchors supplied manually"
    ladder = build_ladder(high, low)
    d = high - low
    return {
        "ticker": ticker.upper(),
        "timeframe": timeframe,
        "asof": df.index[-1].date(),
        "cmp": cmp_,
        "high": high, "low": low, "D": d, "D_pct": d / high,
        "anchor_note": anchor_note,
        "ladder": ladder,
        "period": holding_period(d / high),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def r0(x):
    return int(round(x))


def print_report(res: dict):
    L = res["ladder"]
    boa = L["BOA (entry / 1.0 line)"]
    sl1, sl2 = L["SL1 (first stop)"], L["SL2 (deep stop / 0 line)"]
    t1, t2, t3 = L["Target 1"], L["Target 2"], L["Target 3"]
    cmp_ = res["cmp"]

    tf = {"long": "LONG-TERM swing (weekly, multi-year base)",
          "short": "SHORT-TERM swing (daily, recent leg)"}[res["timeframe"]]
    print(f"\n{res['ticker']} – Stock Report   [{tf}]   (data as of {res['asof']})")
    print(f"CMP: ${cmp_:.2f}")
    print(f"BOA: {r0(boa)}")
    print(f"Targets: {r0(t1)}, {r0(t2)}, {r0(t3)}")
    print(f"SL: {r0(sl1)} or {r0(sl2)}")
    print(f"Period: {res['period']}")
    print(f"Anchor: {res['anchor_note']}  |  D = {res['D']:.2f} ({res['D_pct']:.0%} of BOA)")

    # QCOM-style accumulation wording when price is below the 1.0 line
    if cmp_ < boa:
        print(f"Status: price is below BOA ({r0(boa)}) — 'accumulate on dips' format applies.")
        print(f"        Buying zone near the swing low ~{r0(math.ceil(sl2 / 5) * 5)}; "
              f"deeper buys {r0(sl2 - (1/3)*res['D'])} / {r0(sl2 - res['D'])} "
              f"(-0.333D / -1.0D).")
        print(f"        A close above {r0(boa)} for 3 sessions would be the 'breakout confirmation'.")
    else:
        print(f"Status: price is above BOA — original list format applies (buy on approach to {r0(boa)}).")

    print("\nFull ladder (SL2 + multiplier x D):")
    for name, m in MULTIPLIERS.items():
        flag = "  <- CMP is here" if abs(L[name] - cmp_) / cmp_ < 0.03 else ""
        print(f"  {m:5.3f}D  {name:<26} {L[name]:9.2f}{flag}")
    print("\nReverse-engineered template, not advice. Levels have no demonstrated predictive value.")


def write_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date_run", "ticker", "timeframe", "asof", "cmp", "swing_high", "swing_low", "D",
                    "SL2", "SL1", "BOA", "T1", "T2", "T3", "period", "anchor"])
        for r in results:
            L = r["ladder"]
            w.writerow([datetime.now().date(), r["ticker"], r["timeframe"], r["asof"], round(r["cmp"], 2),
                        round(r["high"], 2), round(r["low"], 2), round(r["D"], 2),
                        *[r0(L[k]) for k in MULTIPLIERS], r["period"], r["anchor_note"]])
    print(f"\nSaved {len(results)} rows to {path}")


# ---------------------------------------------------------------------------
def main():
    global ZIGZAG_PCT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*", help="ticker symbols")
    ap.add_argument("--watchlist", help="text file, one ticker per line")
    ap.add_argument("--high", type=float, help="manual swing high (1.0 anchor)")
    ap.add_argument("--low", type=float, help="manual swing low (0 anchor)")
    ap.add_argument("--timeframe", choices=["long", "short", "both"], default="both",
                    help="long = weekly multi-year base (CAKE-style); "
                         "short = recent daily swing (QCOM-style); default both")
    ap.add_argument("--zigzag", type=float, default=None,
                    help="override reversal %% for swing detection (e.g. 0.05)")
    ap.add_argument("--csv", help="append results to this CSV")
    a = ap.parse_args()
    ZIGZAG_PCT = a.zigzag

    tickers = list(a.tickers)
    if a.watchlist:
        tickers += [t.strip() for t in open(a.watchlist) if t.strip() and not t.startswith("#")]
    if not tickers:
        ap.error("give at least one ticker or --watchlist")
    if (a.high or a.low) and len(tickers) > 1:
        ap.error("--high/--low only make sense with a single ticker")

    results = []
    frames = ["long", "short"] if a.timeframe == "both" else [a.timeframe]
    if a.high and a.low:
        frames = frames[:1]                          # manual anchors: one report
    for t in tickers:
        for tf in frames:
            try:
                res = analyse(t, a.high, a.low, tf)
                print_report(res)
                results.append(res)
            except Exception as e:                   # keep going on bad tickers
                print(f"\n{t} ({tf}): ERROR — {e}")
    if a.csv and results:
        write_csv(results, a.csv)


if __name__ == "__main__":
    main()
