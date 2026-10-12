"""ルール表の修正案を「開発期間」と「検証期間W」で比較する。

python analysis/compare_variants.py  （VARIANTS を編集して使う）
基準は strategies/rulebook_1321.yaml の params。
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest225 import data as D  # noqa: E402
from backtest225.metrics import compute  # noqa: E402
from backtest225.rulebook import RulebookParams, load_breadth, load_events, run_rulebook, summarize  # noqa: E402
from backtest225.rulebook_cli import _nt_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PERIODS = [("開発 2010-01〜2024-08", "2010-01-04", "2024-08-31"), ("W 2024-09〜2026-10", "2024-09-17", None)]


def compare(variants: dict, breadth=ROOT / "data/cache/breadth_prime25.csv"):
    base = (yaml.safe_load(open(ROOT / "strategies/rulebook_1321.yaml", encoding="utf-8")).get("params") or {})
    df, _ = D.load_prices()
    nt, ev = _nt_ratio(), load_events()
    div = D.to_ex_dates(D.load_dividends_csv(D.BUILTIN_DIVIDENDS["1321.T"]), df.index)
    lots = D.lot_size_series(df.index, "1321.T", "auto")
    br = load_breadth(breadth) if Path(breadth).exists() else None
    for name, kw in variants.items():
        print(name)
        for label, start, end in PERIODS:
            d = df if end is None else df[df.index <= end]
            res, c = run_rulebook(d, RulebookParams(**{**base, **kw}), nt=nt, breadth=br, events=ev,
                                  dividends=div, lots=lots, start=start)
            m, sm = compute(res)["strategy"], summarize(res)
            t = sm.get("trades")
            stops = t[t.reason.isin(["初期損切り", "緊急損切り"])] if t is not None and len(t) else None
            worst = t["r"].min() if t is not None and len(t) else np.nan
            print(f"  {label}: 年率 {m['cagr']*100:5.2f}% 損益 {m['profit']:>9,.0f}円 最大DD {m['max_drawdown']*100:5.1f}% "
                  f"取引 {m['n_trades']:3d} 勝率 {m['win_rate']*100:4.0f}% PF {m['profit_factor']:.2f} 平均R {sm.get('avg_r', np.nan):5.2f} "
                  f"| 損切り {0 if stops is None else len(stops)}件 平均 {np.nan if stops is None or not len(stops) else stops.r.mean():.2f}R 最悪 {worst:.2f}R")
            if t is not None and len(t):
                for st, g in t.groupby("setup"):
                    print(f"      Setup {st}: {len(g)}件 勝率 {(g.pnl > 0).mean()*100:3.0f}% 平均R {g.r.mean():5.2f} 損益 {g.pnl.sum():>9,.0f}円 "
                          f"平均保有 {g.days.mean():4.1f}日 決済理由 " + ", ".join(f"{k}{v}" for k, v in g.reason.value_counts().items()))


CASES = {
    # 原因2（基準は当時の ①+② ＝ stop_mode: intraday）
    "2": {
        "原因2 基準（①+②、ザラ場逆指値）": {"stop_mode": "intraday", "a_exit_mode": "ma25"},
        "原因2 案B 終値で損切り判定→翌寄り決済": {"stop_mode": "close", "a_exit_mode": "ma25"},
        "原因2 参考 案B＋緊急逆指値（損切り価格−1ATR）": {"stop_mode": "close", "disaster_atr": 1.0, "a_exit_mode": "ma25"},
    },
    # 原因3（基準は原因2修正後 ＝ stop_mode: close）
    "3": {
        "原因3 基準（Setup A も MA25割れで決済）": {"a_exit_mode": "ma25"},
        "原因3 案1 Setup A は MA25−1ATR 割れで決済": {"a_exit_mode": "ma25_atr", "a_exit_atr": 1.0},
        "原因3 案2 Setup A は MA25割れを使わずスイング安値割れで決済": {"a_exit_mode": "swing_low"},
    },
}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cause", default="3", choices=sorted(CASES))
    compare(CASES[ap.parse_args().cause])
