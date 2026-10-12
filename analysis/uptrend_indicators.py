"""上昇環境で出たシグナル（A の押し目ゾーン / B のブレイク条件を満たした日）について、
追加指標の値と「その後20日の騰落」の関係を調べる。 python analysis/uptrend_indicators.py

・対象: 2_環境認識 E1「上昇」の日のうち、Setup A のゾーン条件か Setup B のブレイク条件を満たした日（見送り・採用を問わない）
・結果: 翌日始値→20営業日後終値の騰落率（%）
・各指標を3分位（低・中・高）に分け、開発期間（2010/1〜2024/8）と W（2024/9〜2026/10）で平均騰落を比べる
・海外指標（S&P500・VIX・ドル円）はシグナル日より前の日付の終値を使う（日本の大引け時点で確定している値）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest225 import data as D, indicators as ind  # noqa: E402
from backtest225.rulebook import RulebookParams, compute_features, load_breadth  # noqa: E402
from backtest225.rulebook_cli import _nt_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    a = lambda s: s.ewm(alpha=1 / n, adjust=False).mean()
    pdi, mdi = 100 * a(pdm) / a(tr), 100 * a(mdm) / a(tr)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return a(dx)


def prior(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """index の各日より前の日付の最新値（海外市場の終値を日本の判定日に合わせる）。"""
    s = series.copy()
    s.index = s.index + pd.Timedelta(days=1)  # 米国日付 d の終値は日本日付 d+1 以降に利用可能
    return s.reindex(index, method="ffill")


def main():
    base = yaml.safe_load(open(ROOT / "strategies/rulebook_1321.yaml", encoding="utf-8"))["params"]
    p = RulebookParams(**base)
    df, _ = D.load_prices()
    br = load_breadth(ROOT / "data/cache/breadth_prime25.csv")
    f = compute_features(df, p, _nt_ratio(), br)
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    atr = f["atr"]
    zoneA = ((c - f.ma25).abs() <= p.near_atr * atr) | ((c - f.kijun).abs() <= p.near_atr * atr) | ((c >= f.fib618) & (c <= f.fib382))
    condB = (c > f.cloud_top) & (c > f.hh_prev)
    sig = (f.env == "上昇") & (zoneA | condB) & (df.index >= "2010-01-04")

    spx, vix, jpy = (D.fetch_yahoo(t)["close"] for t in ("^GSPC", "^VIX", "JPY=X"))
    X = pd.DataFrame(index=df.index)
    X["ADX(14)"] = adx(h, l, c)
    X["MA25の傾き(5日,%)"] = (f.ma25 / f.ma25.shift(5) - 1) * 100
    X["MA75の傾き(20日,%)"] = (f.ma75 / f.ma75.shift(20) - 1) * 100
    X["MA200乖離(%)"] = (c / f.ma200 - 1) * 100
    X["ROC20(%)"] = (c / c.shift(20) - 1) * 100
    X["出来高比(20日平均比)"] = v / v.rolling(20).mean()
    X["ATR/株価(%)"] = atr / c * 100
    X["RSI(14)"] = ind.rsi(c, 14)
    X["NT倍率のMA20乖離(%)"] = (f.nt / f.nt_ma - 1) * 100
    X["S&P500のMA200乖離(%)"] = prior((spx / spx.rolling(200).mean() - 1) * 100, df.index)
    X["VIX"] = prior(vix, df.index)
    X["ドル円20日変化(%)"] = prior((jpy / jpy.shift(20) - 1) * 100, df.index)
    X["騰落レシオ"] = f.breadth
    fwd = (c.shift(-20) / df["open"].shift(-1) - 1) * 100

    d = X[sig].assign(fwd=fwd[sig], setup=np.where(condB[sig], "B", "A")).dropna(subset=["fwd"])
    dev = d[d.index <= "2024-08-31"]
    w = d[d.index >= "2024-09-17"]
    print(f"上昇環境のシグナル日: 開発 {len(dev)}日 / W {len(w)}日（連続日を含む延べ数）")
    print(f"20日後騰落の平均: 開発 {dev.fwd.mean():.2f}% / W {w.fwd.mean():.2f}%\n")
    rows = []
    for col in X.columns:
        dd = dev[[col, "fwd"]].dropna()
        if len(dd) < 30:
            rows.append({"指標": col, "備考": f"開発期間のデータ不足（{len(dd)}日）"})
            continue
        q = dd[col].quantile([1 / 3, 2 / 3]).to_numpy()
        lab = lambda s: pd.cut(s, [-np.inf, q[0], q[1], np.inf], labels=["低", "中", "高"])
        g = dd.groupby(lab(dd[col]), observed=False).fwd.mean()
        ww = w[[col, "fwd"]].dropna()
        gw = ww.groupby(lab(ww[col]), observed=False).fwd.mean() if len(ww) else pd.Series(dtype=float)
        rows.append({"指標": col, "区切り(低|中|高)": f"{q[0]:.2f} | {q[1]:.2f}",
                     "開発 低": g["低"], "開発 中": g["中"], "開発 高": g["高"], "開発 高−低": g["高"] - g["低"],
                     "W 低": gw.get("低", np.nan), "W 中": gw.get("中", np.nan), "W 高": gw.get("高", np.nan),
                     "W 高−低": gw.get("高", np.nan) - gw.get("低", np.nan),
                     "順位相関(開発)": dd[col].rank().corr(dd.fwd.rank())})
    out = pd.DataFrame(rows).round(2)
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.unicode.east_asian_width", True)
    print(out.to_string(index=False))
    out.to_csv(ROOT / "reports/uptrend_indicators.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
