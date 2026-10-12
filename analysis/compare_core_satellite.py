"""買い持ち vs 買い持ち＋ルール表の信用上乗せ（楽天証券の金利）を2期間で比較。

python analysis/compare_core_satellite.py
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest225 import data as D  # noqa: E402
from backtest225.core_satellite import run_core_satellite  # noqa: E402
from backtest225.metrics import compute  # noqa: E402
from backtest225.rulebook import RulebookParams, load_breadth, load_events, summarize  # noqa: E402
from backtest225.rulebook_cli import _nt_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PERIODS = [("開発 2010-01〜2024-08", "2010-01-04", "2024-08-31"), ("W 2024-09〜2026-10", "2024-09-17", None),
           ("全期間 2010-01〜2026-10", "2010-01-04", None)]
RAKUTEN_RATE = 0.028  # 楽天証券 買方金利（制度・一般無期限）年2.80%


def main(variants):
    base = yaml.safe_load(open(ROOT / "strategies/rulebook_1321.yaml", encoding="utf-8"))["params"]
    df, _ = D.load_prices()
    nt, ev = _nt_ratio(), load_events()
    div = D.to_ex_dates(D.load_dividends_csv(D.BUILTIN_DIVIDENDS["1321.T"]), df.index)
    lots = D.lot_size_series(df.index, "1321.T", "auto")
    br = load_breadth(ROOT / "data/cache/breadth_prime25.csv")
    for name, kw in variants.items():
        print(name)
        for label, start, end in PERIODS:
            d = df if end is None else df[df.index <= end]
            p = RulebookParams(**{**base, "margin_rate": RAKUTEN_RATE, **kw})
            res, c = run_core_satellite(d, p, lots=lots, dividends=div, start=start, nt=nt, breadth=br, events=ev)
            m = compute(res)
            s, b = m["strategy"], m["benchmark"]
            sm = summarize(res.satellite)
            lev = res.equity["exposure"]
            print(f"  {label}: 最終資産 {s['final_equity']:>12,.0f}円 (買い持ち {b['final_equity']:>12,.0f}円 差 {s['final_equity']-b['final_equity']:+,.0f}) "
                  f"年率 {s['cagr']*100:5.2f}% (買い持ち {b['cagr']*100:5.2f}%) 最大DD {s['max_drawdown']*100:5.1f}% (買い持ち {b['max_drawdown']*100:5.1f}%)")
            print(f"      信用: 取引 {s['n_trades']} PF {s['profit_factor']:.2f} 平均R {sm.get('avg_r', np.nan):.2f} 金利計 {c['total_interest']:,.0f}円 "
                  f"| レバレッジ 平均 {lev.mean():.2f}倍 最大 {lev.max():.2f}倍 | 保証金維持率 最低 {c['margin_min_ratio']*100:.0f}% 追証日数 {c['margin_call_days']} 余力不足で減らした回数 {c['margin_cap_limited']}")


if __name__ == "__main__":
    main({
        "買い持ち＋ルール表（現行設定：許容リスク2%・建玉上限100%）を信用で上乗せ": {},
    })
