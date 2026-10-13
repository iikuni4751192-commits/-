"""2026-10-13 改定ルール（コア約100万円を信用買いで口数固定・シグナル時に建玉合計225万円まで上乗せ）のバックテスト。
python analysis/backtest_margin_core.py

口座資金（委託保証金の現金）1,057,500円、コア 100万円分、上乗せはリスク基準（口座資産×2%）＋上限。
A 文字どおり: 建玉合計の上限を 225万円（円で固定）
B 比率: 建玉合計の上限を 口座資産×2.128倍（225万÷105.75万）
比較: 買い持ち（現物、口座資金全額）、コア信用のみ（上乗せなし）、① コア現物＋上乗せ（従来の検証方式）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest225 import data as D  # noqa: E402
from backtest225.core_satellite import run_core_satellite, run_margin_core_satellite  # noqa: E402
from backtest225.metrics import _curve_stats  # noqa: E402
from backtest225.rulebook import RulebookParams, load_breadth, load_events  # noqa: E402
from backtest225.rulebook_cli import _nt_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CAPITAL, CORE, CAP = 1_057_500, 1_000_000, 2_250_000
PERIODS = [("開発 2010-01〜2024-08", "2010-01-04", "2024-08-31"), ("W 2024-09〜2026-10", "2024-09-17", "2026-10-09"),
           ("全期間 2010-01〜2026-10", "2010-01-04", "2026-10-09")]


def line(name, eq, extra=""):
    s = _curve_stats(eq, CAPITAL)
    return f"  {name:<34} 最終資産 {s['final_equity']:>12,.0f}円 年率 {s['cagr']*100:6.2f}% 最大DD {s['max_drawdown']*100:6.1f}% {extra}"


def main():
    base = yaml.safe_load(open(ROOT / "strategies/rulebook_1321.yaml", encoding="utf-8"))["params"]
    df, _ = D.load_prices()
    nt, ev, br = _nt_ratio(), load_events(), load_breadth(ROOT / "data/cache/breadth_prime25.csv")
    div = D.to_ex_dates(D.load_dividends_csv(D.BUILTIN_DIVIDENDS["1321.T"]), df.index)
    lots = D.lot_size_series(df.index, "1321.T", "auto")
    p = RulebookParams(**{**base, "capital": CAPITAL, "position_cap": 10.0})  # 建玉金額上限（資金比）は廃止→合計上限で管理
    kw = dict(nt=nt, breadth=br, events=ev)
    for label, start, end in PERIODS:
        d = df[df.index <= end]
        print(f"■ {label}")
        res_a, ca = run_margin_core_satellite(d, p, lots, div, start, CORE, "fixed", CAP, **kw)
        print(line("買い持ち（現物・口座資金全額）", res_a.benchmark))
        core_only = res_a.equity["equity"] - (res_a.satellite.equity["equity"] - CAPITAL)
        print(line("コア信用のみ（上乗せなし）", core_only, f"コア {ca['core_units']:,}口 金利計 {ca['core_interest']:,.0f}円"))
        for name, res, c in (("A 改定ルール（上限225万円固定）", res_a, ca),
                             ("B 改定ルール（上限＝口座資産×2.13倍）",) + run_margin_core_satellite(d, p, lots, div, start, CORE, "ratio", CAP, **kw)):
            t = [x for x in res.satellite.trades if not x.exit_reason.startswith("期間終了")]
            pnl = sum(x.pnl for x in t)
            print(line(name, res.equity["equity"],
                       f"上乗せ {len(t)}回 損益 {pnl:>10,.0f}円 上限で削った {c['margin_cap_limited']}回 "
                       f"保証金率 最低 {c['min_guarantee_ratio']*100:4.0f}% 追証日 {c['call_days']} "
                       f"建玉 最大 {res.equity['exposure'].max():.2f}倍 期末の上限 {c['cap_total_last']:,.0f}円 期末コア {c['core_value_last']:,.0f}円"))
        r1, c1 = run_core_satellite(d, RulebookParams(**{**base, "capital": CAPITAL, "margin_rate": 0.028}), lots=lots,
                                    dividends=div, start=start, **kw)
        print(line("参考 ① コア現物＋上乗せ（従来方式）", r1.equity["equity"], f"上乗せ {len(r1.satellite.trades)}回"))


if __name__ == "__main__":
    main()
