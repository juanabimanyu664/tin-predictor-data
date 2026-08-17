"""
Tin Predictor Pipeline (v3 — fixed, GitHub Actions friendly)
==============================================================
Fetches daily closing prices for the semiconductor basket (+ TINS) via
Yahoo Finance and appends/updates tin_predictor.db. Recomputes semi_index,
per-stock log returns and tins_ret from the full price history each run.

LME tin price is intentionally left blank (tin_price / tin_ret / tin_rebased /
tin_reb_ret are never touched by the automated fetch) — there is no reliable
free daily API for it. Use --override-tin if you key in a number manually
(e.g. from a licensed data feed) later.

Fixes vs the previous version:
  - daily_metrics INSERT now matches the REAL 12-column schema
    (previous version only supplied 10 values -> "table daily_metrics has
    12 columns but 10 values were supplied").
  - No longer fetches a "TIN" ticker (SN=F isn't a real free daily source for
    LME tin) and, critically, no longer overwrites tin_rebased with a flat
    100-chain — the tin_* columns are excluded from the UPDATE clause
    entirely, so existing Bloomberg-sourced values are preserved untouched.
  - Adds TINS.JK (PT Timah) to the automated fetch, since that's a plain
    stock price like the others, and recomputes tins_ret alongside it.
  - GitHub push is no longer done from Python — see the accompanying
    GitHub Actions workflow, which handles checkout/commit/push instead.

Tickers:
  INTC        Intel Corporation (NASDAQ)
  MU          Micron Technology (NASDAQ)
  TSM         TSMC ADR (NYSE)               -> stored as "TSMC"
  000660.KS   SK Hynix (KRX, KRW)           -> converted to USD via USDKRW=X
  005930.KS   Samsung Electronics (KRX, KRW) -> converted to USD via USDKRW=X
  TINS.JK     PT Timah Tbk (IDX, native IDR) -> stored as-is (matches how the
                                                 existing DB already has it)

Usage:
  python update_tin_predictor.py                  # daily update (last ~10 days)
  python update_tin_predictor.py --backfill        # refetch 5y of history
  python update_tin_predictor.py --db path/to.db   # custom db path
  python update_tin_predictor.py --no-fetch        # recompute metrics only
  python update_tin_predictor.py --override-tin 33500 --override-date 2026-08-17
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import sqlite3

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DB_PATH = Path(__file__).parent / "tin_predictor.db"

# name -> yfinance ticker
STOCK_TICKERS = {
    "INTC":    "INTC",
    "MU":      "MU",
    "TSMC":    "TSM",
    "SKHYNIX": "000660.KS",
    "SAMSUNG": "005930.KS",
    "TINS":    "TINS.JK",
}
KRW_TICKERS = {"SKHYNIX", "SAMSUNG"}   # need FX conversion, KRW -> USD
FX_TICKER   = "USDKRW=X"

SEMI_COLS = ["INTC", "MU", "TSMC", "SKHYNIX", "SAMSUNG"]   # the 5-stock basket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "pipeline.log"),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────
def init_db(conn: sqlite3.Connection):
    """Create tables if they don't exist yet. Matches the real schema
    (12 columns in daily_metrics) so this is safe to run against the
    existing tin_predictor.db without altering anything."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            date      TEXT NOT NULL,
            ticker    TEXT NOT NULL,
            price_usd REAL,
            source    TEXT DEFAULT 'yfinance',
            PRIMARY KEY (date, ticker)
        );

        CREATE TABLE IF NOT EXISTS daily_metrics (
            date        TEXT PRIMARY KEY,
            semi_index  REAL,
            tin_price   REAL,
            semi_ret    REAL,
            tin_ret     REAL,
            intc_ret    REAL,
            mu_ret      REAL,
            tsmc_ret    REAL,
            skhynix_ret REAL,
            samsung_ret REAL,
            tin_rebased REAL,
            tin_reb_ret REAL,
            tins_ret    REAL
        );

        CREATE INDEX IF NOT EXISTS idx_prices_date  ON prices(date);
        CREATE INDEX IF NOT EXISTS idx_metrics_date ON daily_metrics(date);
    """)
    conn.commit()
    log.info("DB ready: %s", DB_PATH)


# ─────────────────────────────────────────
# FETCH FROM YAHOO FINANCE
# ─────────────────────────────────────────
def fetch_ticker(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Fetch closing prices for a single ticker. Returns DataFrame[date, close]."""
    for attempt in range(retries):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                log.warning("  %-12s  no data returned", ticker)
                return pd.DataFrame(columns=["date", "close"])

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)

            closes = raw["Close"].dropna()
            df = pd.DataFrame({
                "date":  closes.index.strftime("%Y-%m-%d"),
                "close": closes.values.astype(float),
            })
            log.info("  %-12s  %d rows  (%s -> %s)",
                     ticker, len(df),
                     df["date"].iloc[0], df["date"].iloc[-1])
            return df

        except Exception as e:
            log.warning("  %-12s  attempt %d failed: %s", ticker, attempt + 1, e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    log.error("  %-12s  all retries failed", ticker)
    return pd.DataFrame(columns=["date", "close"])


def fetch_all(start: str, end: str) -> dict:
    """Fetch every stock ticker (+ FX for KRW conversion).
    Returns {name: DataFrame(date, price_usd)}."""
    log.info("Fetching Yahoo Finance: %s -> %s", start, end)
    results = {}

    fx_df = fetch_ticker(FX_TICKER, start, end)
    fx_map = dict(zip(fx_df["date"], fx_df["close"])) if not fx_df.empty else {}

    for name, yticker in STOCK_TICKERS.items():
        df = fetch_ticker(yticker, start, end)
        if df.empty:
            continue

        if name in KRW_TICKERS:
            df["price_usd"] = df.apply(
                lambda r: r["close"] / fx_map.get(r["date"], np.nan)
                if r["date"] in fx_map else np.nan,
                axis=1,
            )
            df = df.dropna(subset=["price_usd"])
        else:
            df["price_usd"] = df["close"]

        results[name] = df[["date", "price_usd"]]

    return results


# ─────────────────────────────────────────
# SAVE TO DB
# ─────────────────────────────────────────
def upsert_prices(conn: sqlite3.Connection, results: dict, source: str = "yfinance"):
    rows = []
    for ticker, df in results.items():
        for _, row in df.iterrows():
            if pd.notna(row["price_usd"]) and row["price_usd"] > 0:
                rows.append((row["date"], ticker, float(row["price_usd"]), source))

    conn.executemany("""
        INSERT INTO prices (date, ticker, price_usd, source)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(date, ticker) DO UPDATE SET
            price_usd = CASE
                WHEN prices.source = 'manual' AND excluded.source != 'manual' THEN prices.price_usd
                ELSE excluded.price_usd
            END,
            source = CASE
                WHEN prices.source = 'manual' AND excluded.source != 'manual' THEN prices.source
                ELSE excluded.source
            END
    """, rows)
    conn.commit()
    log.info("Upserted %d price rows", len(rows))


def override_tin_price(conn: sqlite3.Connection, price: float, dt: str):
    """Manually key in a tin_price for one date (e.g. from a licensed feed).
    Only touches tin_price -- tin_ret / tin_rebased / tin_reb_ret are left
    for you (or a future script) to backfill deliberately, since a single
    manual point can't extend a continuous chained series on its own."""
    cur = conn.execute(
        "UPDATE daily_metrics SET tin_price = ? WHERE date = ?", (price, dt)
    )
    if cur.rowcount == 0:
        conn.execute(
            "INSERT INTO daily_metrics (date, tin_price) VALUES (?, ?)", (dt, price)
        )
    conn.commit()
    log.info("Manual tin_price override: %s = %.2f USD/t", dt, price)


# ─────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────
def compute_metrics(conn: sqlite3.Connection):
    """
    Rebuild semi_index / semi_ret / per-stock ret / tins_ret from the full
    prices table each run (cheap at this data size, and avoids drift).

    Does NOT touch tin_price, tin_ret, tin_rebased, tin_reb_ret -- those stay
    exactly as they already are in the DB (manually entered / left NULL).
    """
    df = pd.read_sql("SELECT date, ticker, price_usd FROM prices ORDER BY date", conn)
    if df.empty:
        log.warning("No price data found in DB")
        return 0

    wide = df.pivot(index="date", columns="ticker", values="price_usd")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()

    # Forward-fill so a KRX/IDX holiday on a NYSE trading day carries the
    # last known Korean/Indonesian price forward (matches how the existing
    # history in this DB was built -- confirmed against the actual rows).
    wide = wide.ffill()

    missing = [c for c in SEMI_COLS if c not in wide.columns]
    if missing:
        log.warning("Missing columns in prices: %s", missing)

    wide = wide.dropna(subset=[c for c in SEMI_COLS if c in wide.columns])
    if wide.empty:
        log.warning("No complete rows after dropping nulls")
        return 0

    log_rets = {}
    for col in SEMI_COLS:
        if col in wide.columns:
            log_rets[col] = np.log(wide[col] / wide[col].shift(1))

    ret_df   = pd.DataFrame(log_rets)
    semi_ret = ret_df.mean(axis=1)   # equal-weighted average of log-returns

    semi_index = 100.0 * (1 + semi_ret.fillna(0)).cumprod()
    semi_index.iloc[0] = 100.0

    tins_ret = (
        np.log(wide["TINS"] / wide["TINS"].shift(1))
        if "TINS" in wide.columns
        else pd.Series(np.nan, index=wide.index)
    )

    metrics = pd.DataFrame({
        "date":        wide.index.strftime("%Y-%m-%d"),
        "semi_index":  semi_index.values,
        "semi_ret":    semi_ret.values,
        "intc_ret":    log_rets.get("INTC",    pd.Series(np.nan, index=wide.index)).values,
        "mu_ret":      log_rets.get("MU",      pd.Series(np.nan, index=wide.index)).values,
        "tsmc_ret":    log_rets.get("TSMC",    pd.Series(np.nan, index=wide.index)).values,
        "skhynix_ret": log_rets.get("SKHYNIX", pd.Series(np.nan, index=wide.index)).values,
        "samsung_ret": log_rets.get("SAMSUNG", pd.Series(np.nan, index=wide.index)).values,
        "tins_ret":    tins_ret.values,
    })

    conn.executemany("""
        INSERT INTO daily_metrics
            (date, semi_index, semi_ret, intc_ret, mu_ret, tsmc_ret, skhynix_ret, samsung_ret, tins_ret)
        VALUES
            (:date, :semi_index, :semi_ret, :intc_ret, :mu_ret, :tsmc_ret, :skhynix_ret, :samsung_ret, :tins_ret)
        ON CONFLICT(date) DO UPDATE SET
            semi_index  = excluded.semi_index,
            semi_ret    = excluded.semi_ret,
            intc_ret    = excluded.intc_ret,
            mu_ret      = excluded.mu_ret,
            tsmc_ret    = excluded.tsmc_ret,
            skhynix_ret = excluded.skhynix_ret,
            samsung_ret = excluded.samsung_ret,
            tins_ret    = excluded.tins_ret
    """, metrics.to_dict("records"))
    conn.commit()

    log.info("Computed metrics for %d trading days", len(metrics))
    return len(metrics)


# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
def print_summary(conn: sqlite3.Connection):
    n_prices   = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    n_metrics  = conn.execute("SELECT COUNT(*) FROM daily_metrics").fetchone()[0]
    date_range = conn.execute("SELECT MIN(date), MAX(date) FROM daily_metrics WHERE date != ''").fetchone()
    last_tin   = conn.execute(
        "SELECT date, tin_price FROM daily_metrics WHERE tin_price IS NOT NULL ORDER BY date DESC LIMIT 1"
    ).fetchone()

    log.info("-" * 50)
    log.info("SUMMARY")
    log.info("  prices rows    : %d", n_prices)
    log.info("  metrics rows   : %d", n_metrics)
    log.info("  date range     : %s -> %s", date_range[0], date_range[1])
    if last_tin:
        log.info("  last tin_price : %s = %.2f USD/t (manual/legacy)", last_tin[0], last_tin[1])
    else:
        log.info("  last tin_price : none on file")
    log.info("  DB size        : %.2f MB", DB_PATH.stat().st_size / 1024 / 1024)
    log.info("-" * 50)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Tin Predictor Daily Pipeline v3")
    parser.add_argument("--backfill", action="store_true",
                         help="Fetch 5 years of historical data from Yahoo Finance")
    parser.add_argument("--override-tin", type=float, metavar="PRICE",
                         help="Manually set tin_price (USD/t) for one date")
    parser.add_argument("--override-date", type=str, default=date.today().isoformat(),
                         help="Date for manual override (default: today, YYYY-MM-DD)")
    parser.add_argument("--no-fetch", action="store_true",
                         help="Skip Yahoo Finance fetch, only recompute metrics")
    parser.add_argument("--db", type=str, default=None,
                         help="Path to SQLite database file (default: tin_predictor.db in same folder)")
    args = parser.parse_args()

    global DB_PATH
    if args.db:
        DB_PATH = Path(args.db)

    log.info("=" * 50)
    log.info("TIN PREDICTOR PIPELINE v3 — %s", date.today())
    log.info("DB: %s", DB_PATH)
    log.info("=" * 50)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    if not args.no_fetch:
        if args.backfill:
            start = "2019-01-01"
            log.info("Backfill mode: fetching from %s", start)
        else:
            start = (date.today() - timedelta(days=10)).isoformat()
            log.info("Daily update mode: fetching from %s", start)

        end = (date.today() + timedelta(days=1)).isoformat()
        results = fetch_all(start, end)

        if results:
            upsert_prices(conn, results, source="yfinance")
        else:
            log.warning("No data fetched from Yahoo Finance")

    if args.override_tin:
        override_tin_price(conn, args.override_tin, args.override_date)

    n = compute_metrics(conn)
    if n == 0:
        log.error("No metrics computed — check your price data")

    print_summary(conn)
    conn.close()


if __name__ == "__main__":
    main()
