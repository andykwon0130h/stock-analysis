#!/usr/bin/env python3
import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

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

def to_kst_index(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 시계열 인덱스를 KST로 변환"""
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    # yfinance가 tz-naive면 UTC로 간주 후 KST로 변환
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df2 = df.copy()
    df2.index = idx.tz_convert(KST)
    return df2

def last_on_or_before_kst(df: pd.DataFrame, when_kst: datetime):
    df_kst = to_kst_index(df)
    valid = df_kst[df_kst.index <= when_kst]
    if valid.empty:
        return None
    return valid.index[-1]

def first_on_or_after_kst(df: pd.DataFrame, when_kst: datetime):
    df_kst = to_kst_index(df)
    valid = df_kst[df_kst.index >= when_kst]
    if valid.empty:
        return None
    return valid.index[0]

def download_history(ticker: str, start_dt_kst: datetime, end_dt_kst: datetime) -> pd.DataFrame:
    """KST 기준 요청을 yfinance에 넘길 때는 날짜만 쓰므로 하루 여유를 둔다."""
    # yfinance의 start/end는 보통 UTC로 해석되니, 넉넉히 버퍼 줌
    start = (start_dt_kst - timedelta(days=2)).date()
    end   = (end_dt_kst + timedelta(days=2)).date()
    return yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)

def nasdaq_drawdown_at_kst(date_kst: datetime):
    try:
        start_kst = date_kst - timedelta(days=3650)
        ixic = download_history("^IXIC", start_kst, date_kst)
        ts_kst = last_on_or_before_kst(ixic, date_kst)
        if ts_kst is None:
            return None
        ixic_kst = to_kst_index(ixic).loc[:ts_kst]
        peak_until = ixic_kst["Close"].cummax().iloc[-1]
        last_close = ixic_kst["Close"].iloc[-1]
        if peak_until == 0:
            return None
        return (last_close / peak_until - 1) * 100
    except Exception:
        return None

def compute_for(
    ticker: str,
    date_str: str,
    rsi_len: int = 14,
    stoch_len: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
    sma_window: int = 200,
    align: str = "prev"  # "prev" | "next" | "exact"
):
    # 입력 날짜를 KST 기준으로 해석
    try:
        # 하루 전체를 포괄하도록 23:59로 설정해도 되지만, 비교는 <= 이라서 자정이면 충분
        when_kst = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=KST)
    except ValueError:
        return {"Ticker": ticker, "RequestedDate": date_str, "Error": "YYYY-MM-DD 형식 아님"}

    # 200일선 계산 여유분
    start_kst = when_kst - timedelta(days=500)
    price = download_history(ticker, start_kst, when_kst)
    if price is None or price.empty:
        return {"Ticker": ticker, "RequestedDate": date_str, "Error": "가격 데이터 없음"}

    # 기준 거래일 결정(KST)
    price_kst = to_kst_index(price)
    exact_exists = when_kst in price_kst.index

    if align == "prev":
        ts = last_on_or_before_kst(price, when_kst)
    elif align == "next":
        ts = first_on_or_after_kst(price, when_kst)
    elif align == "exact":
        ts = when_kst if exact_exists else None
    else:
        ts = last_on_or_before_kst(price, when_kst)

    if ts is None:
        return {"Ticker": ticker, "RequestedDate": date_str, "Error": "요청 날짜에 해당/인접 거래일 없음"}

    # 지표 계산
    rsi = rsi_wilder(price_kst["Close"], rsi_len)
    k, d = stochastic_oscillator(price_kst, stoch_len, smooth_k, smooth_d)
    ma200 = sma(price_kst["Close"], sma_window)
    dd = nasdaq_drawdown_at_kst(ts)

    # 200일선 이격률
    gap_pct = None
    if pd.notna(ma200.loc[ts]) and ma200.loc[ts] != 0:
        gap_pct = (price_kst.loc[ts, "Close"] - ma200.loc[ts]) / ma200.loc[ts] * 100

    row = {
        "Ticker": ticker,
        "RequestedDate": date_str,                 # 사용자가 요청한 날짜(KST)
        "ReportDate": ts.date().isoformat(),       # 실제 사용된 거래일(KST)
        "Close": round(float(price_kst.loc[ts, "Close"]), 4),
        f"RSI_{rsi_len}": round(float(rsi.loc[ts]), 2) if pd.notna(rsi.loc[ts]) else None,
        f"Stoch%K({stoch_len},{smooth_k})": round(float(k.loc[ts]), 2) if pd.notna(k.loc[ts]) else None,
        f"Stoch%D({smooth_d})": round(float(d.loc[ts]), 2) if pd.notna(d.loc[ts]) else None,
        f"SMA_{sma_window}": round(float(ma200.loc[ts]), 4) if pd.notna(ma200.loc[ts]) else None,
        "Gap_from_SMA200_%": round(float(gap_pct), 2) if gap_pct is not None else None,
        "Price>200SMA": bool(price_kst.loc[ts, "Close"] > ma200.loc[ts]) if pd.notna(ma200.loc[ts]) else None,
        "NASDAQ_Drawdown_%": round(float(dd), 2) if dd is not None else None,
        "Error": ""
    }
    return row

def main():
    # 환경변수 + config.json 병행 지원
    import json
    tickers = os.getenv("TICKERS")
    dates   = os.getenv("DATES")
    rsi_len = int(os.getenv("RSI_LEN", "14"))
    st_len  = int(os.getenv("STOCH_LEN", "14"))
    sm_k    = int(os.getenv("SMOOTH_K", "3"))
    sm_d    = int(os.getenv("SMOOTH_D", "3"))
    sma_w   = int(os.getenv("SMA_WINDOW", "200"))
    align   = os.getenv("ALIGN", "prev")  # prev/next/exact

    if not tickers or not dates:
        # config.json 지원
        if os.path.exists("config.json"):
            cfg = json.load(open("config.json", encoding="utf-8"))
            tickers = ",".join(cfg.get("tickers", [])) or "AAPL"
            dates   = ",".join(cfg.get("dates", [])) or "2024-06-03"
            p = cfg.get("params", {})
            rsi_len = int(p.get("rsi_len", rsi_len))
            st_len  = int(p.get("stoch_len", st_len))
            sm_k    = int(p.get("smooth_k", sm_k))
            sm_d    = int(p.get("smooth_d", sm_d))
            sma_w   = int(p.get("sma_window", sma_w))
            align   = p.get("align", align)

    tickers_list = [t.strip() for t in tickers.split(",")]
    dates_list   = [d.strip() for d in dates.split(",")]

    rows = []
    for t in tickers_list:
        for d in dates_list:
            rows.append(
                compute_for(
                    t, d,
                    rsi_len=rsi_len, stoch_len=st_len,
                    smooth_k=sm_k, smooth_d=sm_d,
                    sma_window=sma_w, align=align
                )
            )
    df = pd.DataFrame(rows, columns=[
        "Ticker","RequestedDate","ReportDate","Close",
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
