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


def run_margin_core_satellite(df, p: RulebookParams, lots: pd.Series, dividends: pd.Series | None = None,
                              start: str | None = None, core_yen: float = 1_000_000, cap_mode: str = "ratio",
                              cap_total_yen: float = 2_250_000, cap_ratio: float | None = None, **kw):
    """コアも信用買い（口数固定）＋シグナル時の信用上乗せ。建玉合計に上限を設ける（2026-10-13 改定ルール）。

    口座資金（p.capital）は委託保証金として差し入れる現金。コアは初日の始値で core_yen 分を信用買いし、口数を固定。
    cap_mode="fixed": 建玉合計の上限は cap_total_yen（円で固定）
    cap_mode="ratio": 建玉合計の上限は 口座資産 × cap_ratio（既定は cap_total_yen ÷ 口座資金）
    サテライトの新規は、上限−コア評価額 と 信用建余力（口座資産÷委託保証金率−コア建玉）の小さい方まで。
    """
    d = df[df.index >= pd.Timestamp(start)] if start else df
    lot = lots.reindex(d.index).ffill().bfill().astype(int)
    div = (dividends.groupby(dividends.index).sum().reindex(d.index).fillna(0.0)
           if dividends is not None and len(dividends) else pd.Series(0.0, index=d.index))
    entry = d["open"].iloc[0] * (1 + p.slippage_rate)
    units = int(core_yen // entry // lot.iloc[0] * lot.iloc[0])
    days = d.index.to_series().diff().dt.days.fillna(1).to_numpy()  # 初日は両端入れの1日分
    interest = pd.Series(units * entry * p.margin_rate * days / 365, index=d.index).cumsum()
    div_cum = (div * units).cumsum() - div.iloc[0] * units  # 初日の始値で建てるため初日の権利落ちは対象外
    core_val = units * d["close"]
    core_pnl = units * (d["close"] - entry) + div_cum - interest
    acct = p.capital + core_pnl  # サテライトを除いた口座資産
    ratio = cap_ratio if cap_ratio is not None else cap_total_yen / p.capital
    cap_total = pd.Series(cap_total_yen, index=d.index) if cap_mode == "fixed" else acct * ratio
    # 委託保証金（楽天証券: 評価益は加算されず、評価損は差し引かれる）＝現金＋分配金調整金−金利＋min(コア評価損益, 0)
    core_unreal = units * (d["close"] - entry)
    core_guarantee = p.capital + div_cum - interest + core_unreal.clip(upper=0)
    # run_rulebook 内の余力判定は「coll＋サテライト実現損益」÷保証金率。コアの建玉に必要な保証金分を差し引いて渡す
    coll = (core_guarantee - p.margin_requirement * units * entry).shift(1).bfill()
    ncap = (cap_total - core_val).shift(1).bfill()
    sat, counts = run_rulebook(df, p, dividends=dividends, lots=lots, start=start,
                               extra_equity=core_pnl, collateral=coll, notional_cap=ncap, **kw)
    total = acct + (sat.equity["equity"] - p.capital)
    eq = sat.equity.copy()
    eq["equity"] = total
    eq["exposure"] = (core_val + eq["units"] * d["close"]) / total
    # 保証金率（全建玉）：サテライトの実現損益は決済日に計上、評価損益は損のみ差し引く
    realized = pd.Series(0.0, index=d.index)
    entry_notional = pd.Series(0.0, index=d.index)
    for t in sat.trades:
        if not t.exit_reason.startswith("期間終了"):
            realized.loc[t.exit_date] += t.pnl
        held = (d.index >= t.entry_date) & (d.index < t.exit_date)
        entry_notional[held] += t.initial_units * t.entry_price
    realized = realized.cumsum()
    sat_unreal = (sat.equity["equity"] - p.capital) - realized
    guarantee = core_guarantee + realized + sat_unreal.clip(upper=0)
    guarantee_ratio = guarantee / (units * entry + entry_notional)
    bench = buy_and_hold(d, p.capital, lot.to_numpy(), ExecutionConfig(slippage_rate=p.slippage_rate), div)
    res = Result(equity=eq, trades=sat.trades, benchmark=bench, initial_capital=p.capital,
                 total_cost=sat.total_cost + counts["total_interest"] + float(interest.iloc[-1]),
                 total_dividends=sat.total_dividends + float(div_cum.iloc[-1]))
    res.price = d
    res.satellite = sat
    counts.update({"core_units": units, "core_interest": float(interest.iloc[-1]),
                   "min_guarantee_ratio": float(guarantee_ratio.min()), "call_days": int((guarantee_ratio < p.maintenance_ratio).sum()),
                   "cap_total_last": float(cap_total.iloc[-1]), "core_value_last": float(core_val.iloc[-1])})
    return res, counts
