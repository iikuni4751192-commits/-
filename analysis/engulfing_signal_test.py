"""包み足（陰陽・陽陰）がシグナルとして使えるかを、その後の騰落で確認する。
python analysis/engulfing_signal_test.py

定義（シグナル日 = t）
  陰陽の包み足（買いシグナル候補）:
    t-1 が陰線（終値 < 始値）、t が陽線で 始値[t] <= 終値[t-1] かつ 終値[t] >= 始値[t-1]
  陽陰の包み足（売りシグナル候補）:
    t-1 が陽線、t が陰線で 始値[t] >= 終値[t-1] かつ 終値[t] <= 始値[t-1]
地合い（包み足の直前、t-1 終値までの5営業日騰落で分ける）:
  上昇中 = 終値[t-1] / 終値[t-6] > 1、下落中 = < 1、条件なし = すべて
結果: 翌日始値 → h 営業日後終値の騰落率（h = 1, 5, 10, 20）
基準: 全営業日について同じ騰落率を計算したもの
判定:
  上昇確率 … 正確二項検定（両側）で基準の確率と比較
  平均騰落 … 基準平均との差を t 検定（正規近似、両側）。h>1 では結果期間が重なるため p 値は甘めに出る
  検定を多数行うため、Bonferroni 補正後の有意水準（0.05 / 検定数）も併記する

データ: Yahoo Finance（yfinance、非公式API）。値は一次情報で要確認。
  1321.T は出来高0の行（休日の補完行）と始値0の 2010/2/3 以前、^N225 は四本値が同値の 1988/4/7 以前を除外。
"""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engulfing_after_bull_run import load  # noqa: E402

HORIZONS = [1, 5, 10, 20]


def patterns(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, c = df["open"], df["close"]
    bull, bear = c > o, c < o
    po, pc = o.shift(1), c.shift(1)
    bull_eng = bear.shift(1, fill_value=False) & bull & (o <= pc) & (c >= po)
    bear_eng = bull.shift(1, fill_value=False) & bear & (o >= pc) & (c <= po)
    trend = c.shift(1) / c.shift(6) - 1
    out = {}
    for name, sig in [("陰陽(買い)", bull_eng), ("陽陰(売り)", bear_eng)]:
        out[f"{name} 条件なし"] = sig
        out[f"{name} 上昇中"] = sig & (trend > 0)
        out[f"{name} 下落中"] = sig & (trend < 0)
    return out


def binom_two_sided(k: int, n: int, p0: float) -> float:
    pmf = [math.comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[k] * (1 + 1e-9)))


def norm_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def analyze(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        r = df["close"].shift(-h) / df["open"].shift(-1) - 1
        base = r.dropna()
        p0, m0 = float((base > 0).mean()), float(base.mean())
        for name, sig in patterns(df).items():
            rs = r[sig].dropna()
            n, k = len(rs), int((rs > 0).sum())
            se = rs.std(ddof=1) / math.sqrt(n) if n > 1 else math.nan
            rows.append({
                "対象": ticker, "h": h, "パターン": name, "件数": n,
                "上昇確率%": round(100 * k / n, 1), "基準%": round(100 * p0, 1),
                "確率p値": round(binom_two_sided(k, n, p0), 4),
                "平均%": round(100 * rs.mean(), 3), "基準平均%": round(100 * m0, 3),
                "平均p値": round(norm_two_sided((rs.mean() - m0) / se), 4),
            })
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    res = pd.concat([analyze(t, load(t)) for t in ["1321.T", "^N225"]], ignore_index=True)
    n_tests = 2 * len(res)
    alpha = 0.05 / n_tests
    print(f"検定数 {n_tests}（確率・平均 × {len(res)}行）、Bonferroni 補正後の有意水準 = {alpha:.5f}\n")
    for t, g in res.groupby("対象", sort=False):
        print(f"■ {t}")
        print(g.drop(columns="対象").to_string(index=False))
        print()
    hit = res[(res["確率p値"] < 0.05) | (res["平均p値"] < 0.05)]
    print("■ 補正なしで p<0.05 の行")
    print(hit.to_string(index=False) if len(hit) else "なし")
    strict = res[(res["確率p値"] < alpha) | (res["平均p値"] < alpha)]
    print("\n■ Bonferroni 補正後も有意な行")
    print(strict.to_string(index=False) if len(strict) else "なし")


if __name__ == "__main__":
    main()
