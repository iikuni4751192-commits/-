"""買い持ち（コア）＋ルール表の売買を信用取引で上乗せ（サテライト）の検証。

口座 = 初期資金で1321を買えるだけ買って持ち続ける（現物・分配金込み）
     + ルール表のシグナルが出たときだけ 1321 を信用買い（買方金利を日割りで負担）
口座全体の資産 = 買い持ちの評価額 + 信用取引の累積損益（金利・滑り込み）
信用の建玉は「委託保証金（買い持ち分の代用評価額×掛目＋信用の実現損益）÷ 委託保証金率」まで。
"""
from __future__ import annotations

import pandas as pd

from .engine import ExecutionConfig, Result, buy_and_hold
from .rulebook import RulebookParams, run_rulebook

COLLATERAL_HAIRCUT = 0.80  # 楽天証券 代用掛目（株式等＝上場ETF含む）80%（信用取引ルール 2026年9月）


def run_core_satellite(df, p: RulebookParams, lots: pd.Series, dividends: pd.Series | None = None,
                       start: str | None = None, haircut: float = COLLATERAL_HAIRCUT, **kw):
    d = df[df.index >= pd.Timestamp(start)] if start else df
    lot = lots.reindex(d.index).ffill().bfill()
    div = (dividends.groupby(dividends.index).sum().reindex(d.index).fillna(0.0)
           if dividends is not None and len(dividends) else pd.Series(0.0, index=d.index))
    core = buy_and_hold(d, p.capital, lot.to_numpy().astype(int), ExecutionConfig(slippage_rate=p.slippage_rate), div)
    # 代用評価額: 買い持ちの株式部分（評価額−現金残）× 掛目。現金残は小さいので評価額×掛目で近似せず厳密に分ける
    first_open = d["open"].iloc[0] * (1 + p.slippage_rate)
    units = int(p.capital // first_open // lot.iloc[0] * lot.iloc[0])
    collateral = units * d["close"] * haircut + (core - units * d["close"])  # 現金残は掛目100%
    extra = core - p.capital
    sat, counts = run_rulebook(df, p, dividends=dividends, lots=lots, start=start,
                               extra_equity=extra, collateral=collateral, **kw)
    total = core + (sat.equity["equity"] - p.capital)
    eq = sat.equity.copy()
    eq["equity"] = total
    eq["exposure"] = (units * d["close"] + eq["units"] * d["close"]) / total  # 口座全体の建玉÷資産（レバレッジ）
    res = Result(equity=eq, trades=sat.trades, benchmark=core, initial_capital=p.capital,
                 total_cost=sat.total_cost + counts["total_interest"], total_dividends=sat.total_dividends)
    res.price = d
    res.satellite = sat
    return res, counts
