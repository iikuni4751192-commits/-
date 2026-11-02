"""陽線連続中の陰線を、翌日の陽線が包み前日高値を更新した場合の「1週間後上昇確率」。
python analysis/engulfing_after_bull_run.py

パターン（シグナル日 = t）
  ・t-3, t-2 が陽線（終値 > 始値）＝ 陰線の直前に陽線が2本以上連続
  ・t-1 が陰線（終値 < 始値）
  ・t が陽線で、t-1 の実体を包む（始値[t] <= 終値[t-1] かつ 終値[t] >= 始値[t-1]）
  ・高値[t] > 高値[t-1]（前日高値を更新）
  ・陽線の長さ条件はなし
判定: 終値[t+5]（5営業日後）> 始値[t+1]（翌日始値）なら「上昇」
比較基準: 全営業日 t について同じ「翌日始値→5営業日後終値」が上昇した割合

データ: Yahoo Finance（yfinance、非公式API）。値は一次情報で要確認。
  ・1321.T: 出来高0の行（休日の補完行）を除外。始値が0だった 2010/2/3 以前は除外。
  ・^N225: 1988/4/7 以前は四本値が同値（始値なし）のため除外。
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest225 import data as D  # noqa: E402

H = 5  # 何営業日後の終値で判定するか


def load(ticker: str) -> pd.DataFrame:
    df, _ = D.load_prices("yahoo", ticker)
    if ticker == "1321.T":
        df = df[(df["volume"] > 0) & (df.index > "2010-02-03")]
    elif ticker == "^N225":
        df = df[df.index > "1988-04-07"]
    return df


def signals(df: pd.DataFrame, min_run: int = 2, close_above_high: bool = False) -> pd.Series:
    o, h, c = df["open"], df["high"], df["close"]
    bull, bear = c > o, c < o
    run = pd.Series(True, index=df.index)
    for k in range(2, 2 + min_run):
        run &= bull.shift(k, fill_value=False)
    engulf = bull & (o <= c.shift(1)) & (c >= o.shift(1))
    new_high = (c if close_above_high else h) > h.shift(1)
    return run & bear.shift(1, fill_value=False) & engulf & new_high


def outcome(df: pd.DataFrame) -> pd.Series:
    """翌日始値→H営業日後終値の騰落率。末尾の未確定分は NaN。"""
    return df["close"].shift(-H) / df["open"].shift(-1) - 1


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return m - w, m + w


def binom_p_greater(k: int, n: int, p0: float) -> float:
    """片側 P(X >= k | n, p0) の正確二項検定。"""
    return sum(math.comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(k, n + 1))


def summarize(name: str, df: pd.DataFrame, sig: pd.Series) -> dict:
    r = outcome(df)
    valid = r.notna()
    rs = r[sig & valid]
    n, k = len(rs), int((rs > 0).sum())
    base = r[valid]
    p0 = float((base > 0).mean())
    lo, hi = wilson(k, n)
    return {
        "対象": name,
        "期間": f"{df.index[0].date()}〜{df.index[-1].date()}",
        "件数": n,
        "上昇": k,
        "上昇確率%": round(100 * k / n, 1) if n else math.nan,
        "95%CI": f"{100 * lo:.1f}〜{100 * hi:.1f}" if n else "-",
        "平均騰落%": round(100 * rs.mean(), 2) if n else math.nan,
        "中央値%": round(100 * rs.median(), 2) if n else math.nan,
        "基準確率%": round(100 * p0, 1),
        "基準平均%": round(100 * base.mean(), 2),
        "片側p値": round(binom_p_greater(k, n, p0), 3) if n else math.nan,
    }


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    data = {t: load(t) for t in ["1321.T", "^N225"]}

    print("■ 本条件（陽線2本以上→陰線→包み足陽線・高値更新、判定=翌日始値→5営業日後終値）")
    rows = [summarize(t, df, signals(df)) for t, df in data.items()]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n■ 参考: 期間別（日経平均）")
    n = data["^N225"]
    sig = signals(n)
    rows = []
    for a, b in [("1988", "1999"), ("2000", "2012"), ("2013", "2026")]:
        part = n.loc[a:b]
        rows.append(summarize(f"^N225 {a}-{b}", part, sig.loc[a:b]))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n■ 参考: 条件を変えた場合（感度確認）")
    rows = []
    for t, df in data.items():
        rows.append(summarize(f"{t} 陽線3本以上", df, signals(df, min_run=3)))
        rows.append(summarize(f"{t} 終値で前日高値超え", df, signals(df, close_above_high=True)))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n■ 直近のシグナル（日経平均）")
    r = outcome(n)
    recent = pd.DataFrame({"翌日始値→5日後終値%": (100 * r[sig]).round(2)}).tail(10)
    print(recent.to_string())


if __name__ == "__main__":
    main()
