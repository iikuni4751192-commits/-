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


def run_levered_target(df, weight: pd.Series, capital: float, lots: pd.Series, dividends: pd.Series | None = None,
                       start: str | None = None, margin_rate: float = 0.028, slippage: float = 0.0005,
                       haircut: float = COLLATERAL_HAIRCUT, max_weight: float = 3.3):
    """口座全体の目標建玉比率（資産に対する1321の建玉額、1.0=買い持ち、2.0=信用で2倍）に沿って売買する。

    weight はその日の終値で決まり、翌営業日の始値で調整（値が変わった日だけ売買）。
    資産を超える建玉は信用買い（借入残高に年率 margin_rate を暦日で日割り）。手数料0円（楽天証券のETF信用取引）。
    保証金維持率は「資産×掛目 ÷ 借入残高」で近似して記録する。
    """
    d = df[df.index >= pd.Timestamp(start)] if start else df
    w = weight.reindex(d.index).ffill().fillna(1.0).clip(0, max_weight).to_numpy()
    lot = lots.reindex(d.index).ffill().bfill().astype(int).to_numpy()
    div = (dividends.groupby(dividends.index).sum().reindex(d.index).fillna(0.0).to_numpy()
           if dividends is not None and len(dividends) else [0.0] * len(d))
    O, C = d["open"].to_numpy(float), d["close"].to_numpy(float)
    cash, units, cur_w = capital, 0, None
    interest_total, min_ratio, calls, trades = 0.0, float("inf"), 0, 0
    rows = []
    for i, day in enumerate(d.index):
        if i > 0:
            cash += units * div[i]
            loan = max(-cash, 0.0)
            intr = loan * margin_rate * (day - d.index[i - 1]).days / 365
            cash -= intr
            interest_total += intr
        target = w[i - 1] if i > 0 else w[0]  # 初日は始値で初期比率に合わせる
        if target != cur_w:
            eq = cash + units * O[i]
            new_units = int(target * eq / O[i] // lot[i] * lot[i])
            delta = new_units - units
            if delta:
                px = O[i] * (1 + slippage * (1 if delta > 0 else -1))
                cash -= delta * px
                units = new_units
                trades += 1
            cur_w = target
        eq = cash + units * C[i]
        loan = max(-cash, 0.0)
        if loan > 0:
            ratio = eq * haircut / loan
            min_ratio = min(min_ratio, ratio)
            calls += ratio < 0.20
        rows.append((day, eq, cash, units, C[i], units * C[i] / eq if eq else 0.0))
    eqdf = pd.DataFrame(rows, columns=["date", "equity", "cash", "units", "close", "exposure"]).set_index("date")
    return eqdf, {"total_interest": interest_total, "margin_min_ratio": min_ratio if min_ratio != float("inf") else float("nan"),
                  "margin_call_days": calls, "rebalances": trades}
