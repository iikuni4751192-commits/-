"""日足バックテストエンジン。

約定の前提（未来参照を避けるため保守的に設計）:
  - シグナルは当日終値確定後に判定し、翌営業日の始値で約定（timing: next_open, 既定）
    timing: close を選ぶと当日終値で約定（終値を見てから同じ終値で売買するので楽観的）
  - 損切り/利確/トレーリングは日中の高値・安値で判定。窓を開けて逆指値を飛び越えた場合は始値で約定。
    同じ日に損切りと利確の両方に届いた場合は損切りが先に起きたとみなす（保守的）
  - 売買単位（口数）の整数倍でしか売買しない。手数料・スリッページを反映
  - 分配金は権利落ち日に保有口数×分配金を現金に加算（支払時期のずれは無視）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ExecutionConfig:
    timing: str = "next_open"          # next_open | close
    commission_rate: float = 0.0       # 約定代金に対する手数料率（例 0.001 = 0.1%）
    commission_min: float = 0.0        # 1回あたり最低手数料（円）
    commission_fixed: float = 0.0      # 1回あたり固定手数料（円）
    slippage_rate: float = 0.0         # 不利方向への滑り（例 0.0005 = 0.05%）
    short_fee_rate: float = 0.0        # 空売りの貸株料等（年率）
    tax_rate: float = 0.0              # 年間の実現益に課税（例 0.20315）。損失の繰越は考慮しない


@dataclass
class RiskConfig:
    stop_loss: float | None = None     # 建値から何%逆行で損切り（0.05 = 5%）
    take_profit: float | None = None   # 建値から何%で利確
    trailing_stop: float | None = None # 保有中の最高値（空売りは最安値）から何%戻したら決済
    max_hold_days: int | None = None   # 最大保有営業日数（超えたら翌始値で決済）


@dataclass
class PositionConfig:
    mode: str = "percent"              # percent: 資産×target / fixed_amount: amount×target
    amount: float | None = None


@dataclass
class Trade:
    direction: str
    entry_date: pd.Timestamp
    entry_price: float
    units: int
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    cost: float = 0.0
    exit_reason: str = ""
    max_units: int = 0
    invested: float = 0.0              # 投入元本（建玉金額の累計）

    @property
    def return_pct(self) -> float:
        return self.pnl / self.invested * 100 if self.invested else 0.0


@dataclass
class Result:
    equity: pd.DataFrame
    trades: list[Trade]
    benchmark: pd.Series
    initial_capital: float
    notes: list[str] = field(default_factory=list)
    total_cost: float = 0.0
    total_tax: float = 0.0
    total_dividends: float = 0.0


class _Book:
    """現金・建玉・取引記録の管理"""

    def __init__(self, cash: float, ex: ExecutionConfig):
        self.cash = cash
        self.units = 0
        self.avg = 0.0
        self.ex = ex
        self.trade: Trade | None = None
        self.trades: list[Trade] = []
        self.total_cost = 0.0
        self.realized_by_year: dict[int, float] = {}
        self.entry_i = -1
        self.extreme = 0.0  # 保有中の最高値（ロング）/最安値（ショート）

    def commission(self, notional: float) -> float:
        if notional <= 0:
            return 0.0
        return max(notional * self.ex.commission_rate, self.ex.commission_min) + self.ex.commission_fixed

    def fill_price(self, price: float, side: int) -> float:
        return price * (1 + self.ex.slippage_rate * side)

    def _realize(self, date: pd.Timestamp, amount: float):
        self.realized_by_year[date.year] = self.realized_by_year.get(date.year, 0.0) + amount

    def trade_to(self, target_units: int, price: float, date: pd.Timestamp, i: int, reason: str):
        """建玉を target_units にする（符号付き：正=買い持ち、負=売り持ち）。"""
        if target_units == self.units:
            return
        # 方向転換は「全決済→新規」に分ける
        if self.units != 0 and target_units != 0 and np.sign(target_units) != np.sign(self.units):
            self.trade_to(0, price, date, i, reason)
        delta = target_units - self.units
        side = 1 if delta > 0 else -1
        px = self.fill_price(price, side)
        qty = abs(delta)
        cost = self.commission(qty * px)
        self.total_cost += cost + qty * abs(px - price)
        self.cash -= delta * px + cost

        opening = self.units == 0 or abs(target_units) > abs(self.units)
        if opening:
            if self.units == 0:
                self.trade = Trade(
                    direction="long" if delta > 0 else "short",
                    entry_date=date, entry_price=px, units=qty,
                )
                self.avg = px
                self.entry_i = i
                self.extreme = price
            else:
                self.avg = (self.avg * abs(self.units) + px * qty) / (abs(self.units) + qty)
                self.trade.units += qty
                self.trade.entry_price = self.avg
            self.trade.cost += cost
            self.trade.pnl -= cost
            self.trade.invested += qty * px
            self._realize(date, -cost)
        else:
            sign = 1 if self.units > 0 else -1
            gross = (px - self.avg) * qty * sign
            self.trade.pnl += gross - cost
            self.trade.cost += cost
            self._realize(date, gross - cost)
            # 決済平均単価の記録
            closed_before = self.trade.max_units - abs(self.units) if self.trade.max_units else 0
            prev_exit = self.trade.exit_price or 0.0
            self.trade.exit_price = (prev_exit * closed_before + px * qty) / (closed_before + qty)
        self.units = target_units
        if self.trade:
            self.trade.max_units = max(self.trade.max_units, abs(self.units))
        if self.units == 0 and self.trade is not None:
            self.trade.exit_date = date
            self.trade.exit_reason = reason
            self.trades.append(self.trade)
            self.trade = None
            self.avg = 0.0


def _units_for(weight: float, equity: float, price: float, lot: int, book: _Book, pos: PositionConfig) -> int:
    base = equity if pos.mode == "percent" else float(pos.amount or 0)
    notional = abs(weight) * base
    px = book.fill_price(price, 1 if weight > 0 else -1)
    # 手数料込みで買える最大口数（概算後に調整）
    units = int(notional // (px * (1 + book.ex.commission_rate)) // lot * lot)
    if weight > 0:
        # 現金の範囲に収める（レバレッジなし）。保有分を売る場合の受取額も考慮
        budget = book.cash + max(book.units, 0) * price
        while units > 0 and units * px + book.commission(units * px) > budget + 1e-9:
            units -= lot
    return int(math.copysign(max(units, 0), weight))


def run_backtest(
    df: pd.DataFrame,
    target: pd.Series,
    capital: float = 1_000_000,
    lots: pd.Series | None = None,
    execution: ExecutionConfig | None = None,
    risk: RiskConfig | None = None,
    position: PositionConfig | None = None,
    dividends: pd.Series | None = None,
) -> Result:
    ex = execution or ExecutionConfig()
    rk = risk or RiskConfig()
    pos = position or PositionConfig()
    if ex.timing not in ("next_open", "close"):
        raise ValueError("execution.timing は next_open か close")
    lots = lots if lots is not None else pd.Series(1, index=df.index)
    div = (dividends.groupby(dividends.index.normalize()).sum().reindex(df.index).fillna(0.0)
           if dividends is not None and len(dividends) else pd.Series(0.0, index=df.index))

    div_arr = div.to_numpy(float)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    tgt = target.reindex(df.index).to_numpy(float)
    lot_arr = lots.reindex(df.index).ffill().bfill().to_numpy(int)
    dates = df.index
    n = len(df)

    book = _Book(capital, ex)
    cur_target = 0.0
    total_div = 0.0
    total_tax = 0.0
    rows = []

    def apply_target(w: float, price: float, i: int, reason: str):
        nonlocal cur_target
        equity = book.cash + book.units * price
        units = 0 if w == 0 else _units_for(w, equity, price, lot_arr[i], book, pos)
        book.trade_to(units, price, dates[i], i, reason)
        cur_target = w

    for i in range(n):
        date = dates[i]
        # 1) 分配金（前日終値時点の保有に対して）
        if div_arr[i] and book.units:
            amt = book.units * div_arr[i]
            book.cash += amt
            total_div += amt
            book._realize(date, amt)
            if book.trade:
                book.trade.pnl += amt

        # 2) 始値での約定（前日シグナル / 最大保有日数）
        if ex.timing == "next_open" and i > 0:
            w = tgt[i - 1]
            if not np.isnan(w) and w != cur_target:
                apply_target(w, o[i], i, "シグナル")
        if book.units and rk.max_hold_days and i - book.entry_i >= rk.max_hold_days:
            apply_target(0.0, o[i], i, "保有期間上限")

        # 3) 日中の損切り・利確・トレーリング
        if book.units:
            long = book.units > 0
            levels = []  # (優先度, 価格, 理由)
            if rk.stop_loss:
                levels.append((0, book.avg * (1 - rk.stop_loss) if long else book.avg * (1 + rk.stop_loss), "損切り"))
            if rk.trailing_stop:
                levels.append((1, book.extreme * (1 - rk.trailing_stop) if long else book.extreme * (1 + rk.trailing_stop), "トレーリング"))
            if rk.take_profit:
                levels.append((2, book.avg * (1 + rk.take_profit) if long else book.avg * (1 - rk.take_profit), "利確"))
            hit = None
            for pri, lvl, why in sorted(levels):
                adverse = pri < 2
                if long:
                    touched = (l[i] <= lvl) if adverse else (h[i] >= lvl)
                    gap = (o[i] <= lvl) if adverse else (o[i] >= lvl)
                else:
                    touched = (h[i] >= lvl) if adverse else (l[i] <= lvl)
                    gap = (o[i] >= lvl) if adverse else (o[i] <= lvl)
                if touched:
                    hit = (o[i] if gap else lvl, why)
                    break
            if hit:
                book.trade_to(0, hit[0], date, i, hit[1])
                cur_target = 0.0
            else:
                book.extreme = max(book.extreme, h[i]) if long else min(book.extreme, l[i])

        # 4) 終値での約定（timing: close）
        if ex.timing == "close":
            w = tgt[i]
            if not np.isnan(w) and w != cur_target:
                apply_target(w, c[i], i, "シグナル")

        # 5) 空売りコスト・年末課税・評価
        if book.units < 0 and ex.short_fee_rate:
            fee = -book.units * c[i] * ex.short_fee_rate / 365
            book.cash -= fee
            book.total_cost += fee
            book._realize(date, -fee)
            if book.trade:
                book.trade.pnl -= fee
        year_end = i == n - 1 or dates[i + 1].year != date.year
        if ex.tax_rate and year_end:
            gain = book.realized_by_year.get(date.year, 0.0)
            if gain > 0:
                tax = gain * ex.tax_rate
                book.cash -= tax
                total_tax += tax

        equity = book.cash + book.units * c[i]
        rows.append((date, equity, book.cash, book.units, c[i], book.units * c[i] / equity if equity else 0.0))

    # 最終日に建玉が残っていれば、記録上は未決済として終値で評価
    if book.trade is not None:
        t = book.trade
        sign = 1 if book.units > 0 else -1
        t.pnl += (c[-1] - book.avg) * abs(book.units) * sign
        t.exit_date = dates[-1]
        t.exit_price = c[-1]
        t.exit_reason = "期間終了（未決済・評価）"
        book.trades.append(t)

    eq = pd.DataFrame(rows, columns=["date", "equity", "cash", "units", "close", "exposure"]).set_index("date")
    bench = buy_and_hold(df, capital, lot_arr, ex, div)
    return Result(
        equity=eq, trades=book.trades, benchmark=bench, initial_capital=capital,
        total_cost=book.total_cost, total_tax=total_tax, total_dividends=total_div,
    )


def buy_and_hold(df: pd.DataFrame, capital: float, lot_arr: np.ndarray, ex: ExecutionConfig, div: pd.Series) -> pd.Series:
    """比較用：初日の始値で買えるだけ買い、最後まで保有（同じ手数料・分配金条件、課税なし）。"""
    book = _Book(capital, ExecutionConfig(**{**ex.__dict__, "tax_rate": 0.0}))
    price = df["open"].iloc[0]
    px = book.fill_price(price, 1)
    units = int(capital // (px * (1 + ex.commission_rate)) // lot_arr[0] * lot_arr[0])
    while units > 0 and units * px + book.commission(units * px) > capital:
        units -= lot_arr[0]
    book.trade_to(units, price, df.index[0], 0, "")
    d = div.to_numpy().copy()
    d[0] = 0.0  # 初日の始値で買うため、初日が権利落ち日でも分配金は受け取れない
    cash = book.cash + (d * units).cumsum()
    return pd.Series(cash + units * df["close"].to_numpy(), index=df.index, name="buy_and_hold")
