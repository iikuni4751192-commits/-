"""海外指標・日経先物で、上昇環境のトレンド見逃しを減らせるかの検証。 python analysis/overseas_indicators.py

検証1（トレンドの判定）: シグナル日（日本の大引け）時点で確定している海外指数のトレンドと、その後20日の騰落の関係
検証2（翌朝の判断）: シグナル日の夜（米国市場・CME日経先物）の値動きが、翌日の窓開け・指値不成立・その後の騰落を示すか
  - CME日経平均先物（円建て, NIY=F）の米国日付 d の終値は日本時間 d+1 の早朝に確定 → 日本の d+1 の寄り前に利用可能
  - 米国指数（S&P500, NASDAQ, SOX）の米国日付 d の終値も同様
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest225 import data as D  # noqa: E402
from backtest225.rulebook import RulebookParams, compute_features, load_breadth  # noqa: E402
from backtest225.rulebook_cli import _nt_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
US = {"S&P500": "^GSPC", "NASDAQ": "^IXIC", "SOX": "^SOX"}


def avail_before(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """日本の日付 T の時点（寄り前・大引けとも）で確定している最新値＝米国日付 T-1 以前の終値。"""
    s = series.copy()
    s.index = s.index + pd.Timedelta(days=1)
    return s.reindex(index, method="ffill")


def tercile_table(d: pd.DataFrame, cols, target="fwd"):
    dev, w = d[d.index <= "2024-08-31"], d[d.index >= "2024-09-17"]
    rows = []
    for col in cols:
        dd = dev[[col, target]].dropna()
        q = dd[col].quantile([1 / 3, 2 / 3]).to_numpy()
        lab = lambda s: pd.cut(s, [-np.inf, q[0], q[1], np.inf], labels=["低", "中", "高"])
        g = dd.groupby(lab(dd[col]), observed=False)[target].mean()
        ww = w[[col, target]].dropna()
        gw = ww.groupby(lab(ww[col]), observed=False)[target].mean()
        rows.append({"指標": col, "区切り(低|中|高)": f"{q[0]:.2f} | {q[1]:.2f}", "開発 低": g["低"], "開発 中": g["中"],
                     "開発 高": g["高"], "開発 高−低": g["高"] - g["低"], "W 低": gw.get("低"), "W 中": gw.get("中"),
                     "W 高": gw.get("高"), "W 高−低": gw.get("高", np.nan) - gw.get("低", np.nan)})
    return pd.DataFrame(rows).round(2)


def main():
    base = yaml.safe_load(open(ROOT / "strategies/rulebook_1321.yaml", encoding="utf-8"))["params"]
    p = RulebookParams(**base)
    df, _ = D.load_prices()
    f = compute_features(df, p, _nt_ratio(), load_breadth(ROOT / "data/cache/breadth_prime25.csv"))
    idx, c, o = df.index, df["close"], df["open"]
    atr = f["atr"]
    zoneA = ((c - f.ma25).abs() <= p.near_atr * atr) | ((c - f.kijun).abs() <= p.near_atr * atr) | ((c >= f.fib618) & (c <= f.fib382))
    condB = (c > f.cloud_top) & (c > f.hh_prev)
    up = (f.env == "上昇") & (idx >= "2010-01-04")
    fwd = (c.shift(-20) / o.shift(-1) - 1) * 100
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.unicode.east_asian_width", True)

    # ---- 検証1 ----
    X = pd.DataFrame(index=idx)
    for name, t in US.items():
        s = D.fetch_yahoo(t)["close"]
        X[f"{name} MA200乖離%"] = avail_before((s / s.rolling(200).mean() - 1) * 100, idx)
        X[f"{name} 20日騰落%"] = avail_before((s / s.shift(20) - 1) * 100, idx)
        X[f"{name} MA50の傾き(10日)%"] = avail_before((s.rolling(50).mean() / s.rolling(50).mean().shift(10) - 1) * 100, idx)
    sig = up & (zoneA | condB)
    d1 = X[sig].assign(fwd=fwd[sig]).dropna(subset=["fwd"])
    print("■検証1 上昇環境のシグナル日（A ゾーン or B ブレイク条件）: 開発 %d日 / W %d日、20日後騰落 平均 開発 %.2f%% / W %.2f%%"
          % ((d1.index <= "2024-08-31").sum(), (d1.index >= "2024-09-17").sum(),
             d1[d1.index <= "2024-08-31"].fwd.mean(), d1[d1.index >= "2024-09-17"].fwd.mean()))
    t1 = tercile_table(d1, X.columns)
    print(t1.to_string(index=False))
    # 単純な二値フィルタ：米国指数が200日線の上か
    for name in US:
        col = f"{name} MA200乖離%"
        for per, sel in (("開発", d1.index <= "2024-08-31"), ("W", d1.index >= "2024-09-17")):
            dd = d1[sel]
            a, b = dd[dd[col] > 0].fwd, dd[dd[col] <= 0].fwd
            print(f"  {name}が200日線の上/下: {per} 上 {a.mean():5.2f}%({len(a)}日) / 下 {b.mean():5.2f}%({len(b)}日)")

    # ---- 検証2 ----
    niy = D.fetch_yahoo("NIY=F")["close"]
    n225 = D.fetch_yahoo("^N225")["close"].reindex(idx)
    nxt = pd.Series(idx[1:].append(pd.DatetimeIndex([idx[-1] + pd.Timedelta(days=1)])), index=idx)  # 翌営業日
    niy_pre_next = avail_before(niy, pd.DatetimeIndex(nxt.values))  # 翌営業日の寄り前に分かる先物終値
    Y = pd.DataFrame(index=idx)
    Y["夜間 日経先物(CME円建て)の変化%"] = (niy_pre_next.to_numpy() / n225.to_numpy() - 1) * 100
    for name, t in US.items():
        s = D.fetch_yahoo(t)["close"]
        Y[f"夜間 {name}の騰落%"] = (avail_before(s, pd.DatetimeIndex(nxt.values)).to_numpy() / avail_before(s, idx).to_numpy() - 1) * 100
    gap = (o.shift(-1) / c - 1) * 100
    Y["翌日の窓(寄り/前日終値)%"] = gap
    sigB = up & condB
    level = f.hh_prev
    d2 = Y[sigB].assign(fwd=fwd[sigB], unfilled=(df["low"].shift(-1) > level)[sigB]).dropna(subset=["fwd"])
    print("\n■検証2 上昇環境の Setup B シグナル日: 開発 %d日 / W %d日" % ((d2.index <= "2024-08-31").sum(), (d2.index >= "2024-09-17").sum()))
    dev = d2[d2.index <= "2024-08-31"]
    print("  夜間指標と翌日の窓の相関（開発期間）:", {k: round(dev[k].corr(dev["翌日の窓(寄り/前日終値)%"]), 2) for k in Y.columns[:-1]})
    for k in ["夜間 日経先物(CME円建て)の変化%", "夜間 SOXの騰落%", "夜間 NASDAQの騰落%"]:
        for per, sel in (("開発", d2.index <= "2024-08-31"), ("W", d2.index >= "2024-09-17")):
            dd = d2[sel]
            hi, lo = dd[dd[k] > 0.5], dd[dd[k] <= 0.5]
            print(f"  {k} >+0.5%: {per} {len(hi):3d}日 指値不成立 {hi.unfilled.mean()*100:4.0f}% 20日後 {hi.fwd.mean():5.2f}% | "
                  f"それ以外 {len(lo):3d}日 指値不成立 {lo.unfilled.mean()*100:4.0f}% 20日後 {lo.fwd.mean():5.2f}%")
    print(tercile_table(d2, list(Y.columns[:-1])).to_string(index=False))


if __name__ == "__main__":
    main()
