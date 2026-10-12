"""買い持ちを利益額で上回る3つの設計の比較（楽天証券 買方金利 年2.80%）。

① 買い持ち＋ルール表のシグナル時だけ信用で上乗せ（原因1〜6修正後のルール表）
② 買い持ち＋売られすぎ（終値≦ボリンジャーバンド−2σ）で資産の100%分を信用で上乗せ。
   終値がBB中心線（20日平均）以上に戻るか、10営業日で上乗せ分を手仕舞い
③ トレンドで建玉比率を変える（終値>200日平均なら資産の2倍、以下なら ③a 1倍 / ③b 0倍）
比較用: 各設計と同じ平均建玉比率を毎日一定で持つ場合（判断なしのレバレッジ）

python analysis/compare_leverage_designs.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest225 import data as D, indicators as ind  # noqa: E402
from backtest225.core_satellite import run_core_satellite, run_levered_target  # noqa: E402
from backtest225.metrics import _curve_stats  # noqa: E402
from backtest225.rulebook import RulebookParams, load_breadth, load_events  # noqa: E402
from backtest225.rulebook_cli import _nt_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PERIODS = [("開発 2010-01〜2024-08", "2010-01-04", "2024-08-31"), ("W 2024-09〜2026-10", "2024-09-17", None),
           ("全期間 2010-01〜2026-10", "2010-01-04", None)]
RATE, CAP = 0.028, 2_400_000


def dip_weights(df, add=1.0, max_days=10):
    c = df["close"]
    lo, mid = ind.bb_lower(c, 20, 2), ind.sma(c, 20)
    w = np.ones(len(df))
    held = -1
    for i in range(len(df)):
        if held >= 0:
            held += 1
            if c.iloc[i] >= mid.iloc[i] or held >= max_days:
                held = -1
        elif c.iloc[i] <= lo.iloc[i]:
            held = 0
        w[i] = 1 + add if held >= 0 else 1.0
    return pd.Series(w, index=df.index)


def trend_weights(df, up=2.0, down=1.0):
    c = df["close"]
    ma = ind.sma(c, 200)
    return pd.Series(np.where(c > ma, up, down), index=df.index).where(ma.notna(), 1.0)


def stats(eq):
    s = _curve_stats(eq, CAP)
    return s["final_equity"], s["cagr"], s["max_drawdown"]


def main():
    base = yaml.safe_load(open(ROOT / "strategies/rulebook_1321.yaml", encoding="utf-8"))["params"]
    df, _ = D.load_prices()
    nt, ev = _nt_ratio(), load_events()
    div = D.to_ex_dates(D.load_dividends_csv(D.BUILTIN_DIVIDENDS["1321.T"]), df.index)
    lots = D.lot_size_series(df.index, "1321.T", "auto")
    br = load_breadth(ROOT / "data/cache/breadth_prime25.csv")
    designs = {"② 押し目で上乗せ": dip_weights(df), "③a トレンド 2倍/1倍": trend_weights(df, 2, 1),
               "③b トレンド 2倍/0倍": trend_weights(df, 2, 0)}
    for label, start, end in PERIODS:
        d = df if end is None else df[df.index <= end]
        print(f"■ {label}")
        bh, _ = run_levered_target(d, pd.Series(1.0, index=d.index), CAP, lots, div, start, margin_rate=RATE)
        f, g, m = stats(bh["equity"])
        print(f"  買い持ち              最終資産 {f:>12,.0f}円  年率 {g*100:6.2f}%  最大DD {m*100:6.1f}%")
        results = {}
        res1, c1 = run_core_satellite(d, RulebookParams(**{**base, "margin_rate": RATE}), lots=lots, dividends=div,
                                      start=start, nt=nt, breadth=br, events=ev)
        dd = d[d.index >= start]
        lev1 = 1 + res1.equity["units"] * dd["close"] / res1.equity["equity"]
        results["① ルール表で上乗せ"] = (res1.equity["equity"], lev1, c1)
        for name, w in designs.items():
            eq, c = run_levered_target(d, w, CAP, lots, div, start, margin_rate=RATE)
            results[name] = (eq["equity"], eq["exposure"], c)
        for name, (eq, lev, c) in results.items():
            f, g, m = stats(eq)
            avg = float(lev.mean())
            const, _ = run_levered_target(d, pd.Series(avg, index=d.index), CAP, lots, div, start, margin_rate=RATE)
            fc, gc, mc = stats(const["equity"])
            print(f"  {name:<16} 最終資産 {f:>12,.0f}円  年率 {g*100:6.2f}%  最大DD {m*100:6.1f}%  | 平均建玉 {avg:.2f}倍 最大 {lev.max():.2f}倍 "
                  f"金利 {c['total_interest']:>9,.0f}円 維持率最低 {c['margin_min_ratio']*100:4.0f}% 追証日 {c['margin_call_days']}"
                  f"  || 同じ{avg:.2f}倍を常時: 年率 {gc*100:6.2f}% 最大DD {mc*100:6.1f}%")


if __name__ == "__main__":
    main()
