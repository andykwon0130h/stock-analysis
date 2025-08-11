#!/usr/bin/env python3
import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone

def rsi_wilder(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def stochastic_oscillator(df, length=14, smooth_k=3, smooth_d=3):
    low_min  = df['Low'].rolling(window=length, min_periods=length).min()
    high_max = df['High'].rolling(window=length, min_periods=length).max()
    percent_k_raw = (df['Close'] - low_min) / (high_max - low_min) * 100
    percent_k = percent_k_raw.rolling(window=smooth_k, min_periods=smooth_k).mean()
    percent_d = percent_k.rolling(window=smooth_d, min_periods=smooth_d).mean()
    return percent_k, percent_d

def sma(series, window=200):
    return series.rolling(window=window, min_periods=window).mean()

def last_on_or_before(df, when):
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(timezone.utc)
    df2 = df.copy()
    df2.index = idx
    valid = df2[df2.index <= when]
    if valid.empty:
        return None
    return valid.index[-1]

def download_history(ticker, start_dt, end_dt):
    return yf.Ticker(ticker).history(start=start_dt.date(), end=(end_dt + timedelta(days=1)).date(), auto_adjust=False)

def nasdaq_drawdown_at(date_ts):
    try:
        start = date_ts - timedelta(days=3650)
        ixic = yf.Ticker("^IXIC").history(start=start.date(), end=(date_ts + timedelta(days=1)).date())
        ts = last_on_or_before(ixic, date_ts)
        if ts is None: return None
        ixic = ixic.loc[:ts]
        peak_until = ixic["Close"].cummax().iloc[-1]
        last_close = ixic["Close"].iloc[-1]
        if peak_until == 0: return None
        return (last_close / peak_until - 1) * 100
    except Exception:
        return None

def compute_for(ticker, date_str, rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3, sma_window=200):
    try:
        when = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return {"Ticker": ticker, "InputDate": date_str, "Error": "YYYY-MM-DD 형식 아님"}
    start = when - timedelta(days=500)
    price = download_history(ticker, start, when)
    if price is None or price.empty:
        return {"Ticker": ticker, "InputDate": date_str, "Error": "데이터 없음"}
    ts = last_on_or_before(price, when)
    if ts is None:
        return {"Ticker": ticker, "InputDate": date_str, "Error": "직전 거래일 없음"}

    rsi = rsi_wilder(price["Close"], rsi_len)
    k, d = stochastic_oscillator(price, stoch_len, smooth_k, smooth_d)
    ma200 = sma(price["Close"], sma_window)
    dd = nasdaq_drawdown_at(ts)

    gap_pct = None
    if pd.notna(ma200.loc[ts]) and ma200.loc[ts] != 0:
        gap_pct = (price.loc[ts, "Close"] - ma200.loc[ts]) / ma200.loc[ts] * 100

    row = {
        "Ticker": ticker,
        "ReportDate": ts.date().isoformat(),
        "Close": round(float(price.loc[ts, "Close"]), 4),
        f"RSI_{rsi_len}": round(float(rsi.loc[ts]), 2) if pd.notna(rsi.loc[ts]) else None,
        f"Stoch%K({stoch_len},{smooth_k})": round(float(k.loc[ts]), 2) if pd.notna(k.loc[ts]) else None,
        f"Stoch%D({smooth_d})": round(float(d.loc[ts]), 2) if pd.notna(d.loc[ts]) else None,
        f"SMA_{sma_window}": round(float(ma200.loc[ts]), 4) if pd.notna(ma200.loc[ts]) else None,
        "Gap_from_SMA200_%": round(float(gap_pct), 2) if gap_pct is not None else None,
        "Price>200SMA": bool(price.loc[ts, "Close"] > ma200.loc[ts]) if pd.notna(ma200.loc[ts]) else None,
        "NASDAQ_Drawdown_%": round(float(dd), 2) if dd is not None else None,
        "Error": ""
    }
    return row

def main():
    # 입력은 환경변수로 받음: 콤마로 구분
    tickers = os.getenv("TICKERS", "AAPL,MSFT,005930.KS").split(",")
    dates   = os.getenv("DATES",   "2024-06-03").split(",")
    rsi_len = int(os.getenv("RSI_LEN", "14"))
    st_len  = int(os.getenv("STOCH_LEN", "14"))
    sm_k    = int(os.getenv("SMOOTH_K", "3"))
    sm_d    = int(os.getenv("SMOOTH_D", "3"))
    sma_w   = int(os.getenv("SMA_WINDOW", "200"))

    inputs = [{"ticker": t.strip(), "date": d.strip()} for t in tickers for d in dates]
    rows = [compute_for(i["ticker"], i["date"], rsi_len, st_len, sm_k, sm_d, sma_w) for i in inputs]
    df = pd.DataFrame(rows, columns=[
        "Ticker","ReportDate","Close",
        f"RSI_{rsi_len}",
        f"Stoch%K({st_len},{sm_k})",
        f"Stoch%D({sm_d})",
        f"SMA_{sma_w}",
        "Gap_from_SMA200_%","Price>200SMA",
        "NASDAQ_Drawdown_%","Error"
    ])
    out = "indicators_result.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    sys.exit(main())
