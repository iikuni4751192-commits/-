"""成績指標の計算"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Result

TRADING_DAYS = 245  # 東証の年間営業日数の目安（年により変動）


def _curve_stats(eq: pd.Series, initial: float) -> dict:
    ret = eq.pct_change().dropna()
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    final = float(eq.iloc[-1])
    total = final / initial - 1
    cagr = (final / initial) ** (1 / years) - 1 if years and final > 0 else np.nan
    vol = ret.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = ret.mean() / ret.std(ddof=0) * np.sqrt(TRADING_DAYS) if ret.std(ddof=0) > 0 else np.nan
    downside = ret[ret < 0].std(ddof=0)
    sortino = ret.mean() / downside * np.sqrt(TRADING_DAYS) if downside and downside > 0 else np.nan
    peak = eq.cummax()
    dd = eq / peak - 1
    mdd = float(dd.min())
    # 最長のドローダウン期間（高値更新までの暦日数）
    underwater = dd < 0
    longest, start = 0, None
    for d, u in underwater.items():
        if u and start is None:
            start = d
        elif not u and start is not None:
            longest = max(longest, (d - start).days)
            start = None
    if start is not None:
        longest = max(longest, (eq.index[-1] - start).days)
    return {
        "final_equity": final,
        "profit": final - initial,
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "max_dd_days": longest,
        "calmar": cagr / abs(mdd) if mdd < 0 and not np.isnan(cagr) else np.nan,
        "years": years,
    }


def compute(result: Result) -> dict:
    eq = result.equity["equity"]
    s = _curve_stats(eq, result.initial_capital)
    b = _curve_stats(result.benchmark, result.initial_capital)

    trades = result.trades
    closed = [t for t in trades if not t.exit_reason.startswith("期間終了")]
    pnls = np.array([t.pnl for t in closed]) if closed else np.array([])
    wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
    hold_days = [np.busday_count(t.entry_date.date(), t.exit_date.date()) for t in closed]
    s.update({
        "n_trades": len(closed),
        "open_trade": len(trades) - len(closed),
        "win_rate": len(wins) / len(pnls) if len(pnls) else np.nan,
        "avg_win": wins.mean() if len(wins) else np.nan,
        "avg_loss": losses.mean() if len(losses) else np.nan,
        "profit_factor": wins.sum() / -losses.sum() if len(losses) and losses.sum() < 0 else np.nan,
        "expectancy": pnls.mean() if len(pnls) else np.nan,
        "avg_hold_days": float(np.mean(hold_days)) if hold_days else np.nan,
        "exposure": float((result.equity["units"] != 0).mean()),
        "total_cost": result.total_cost,
        "total_tax": result.total_tax,
        "total_dividends": result.total_dividends,
        "max_consecutive_losses": _max_streak(pnls <= 0),
    })
    return {"strategy": s, "benchmark": b}


def _max_streak(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def annual_returns(eq: pd.Series, initial: float) -> pd.Series:
    year_end = eq.groupby(eq.index.year).last()
    prev = year_end.shift(1)
    prev.iloc[0] = initial
    return year_end / prev - 1


def monthly_returns(eq: pd.Series, initial: float) -> pd.DataFrame:
    m = eq.groupby([eq.index.year, eq.index.month]).last()
    prev = m.shift(1)
    prev.iloc[0] = initial
    r = (m / prev - 1).rename("ret")
    table = r.unstack(level=1)
    table.index.name, table.columns.name = "year", "month"
    return table


def trades_frame(result: Result) -> pd.DataFrame:
    rows = [{
        "方向": "買い" if t.direction == "long" else "売り",
        "建日": t.entry_date.date(),
        "建値": round(t.entry_price, 2),
        "決済日": t.exit_date.date() if t.exit_date is not None else None,
        "決済値": round(t.exit_price, 2) if t.exit_price is not None else None,
        "口数": t.max_units,
        "損益(円)": round(t.pnl),
        "損益率(%)": round(t.return_pct, 2),
        "コスト(円)": round(t.cost),
        "決済理由": t.exit_reason,
    } for t in result.trades]
    return pd.DataFrame(rows)


LABELS = {
    "final_equity": ("最終資産", "yen"),
    "profit": ("損益", "yen"),
    "total_return": ("トータルリターン", "pct"),
    "cagr": ("年率リターン(CAGR)", "pct"),
    "volatility": ("年率ボラティリティ", "pct"),
    "sharpe": ("シャープレシオ(無リスク金利0)", "num"),
    "sortino": ("ソルティノレシオ", "num"),
    "max_drawdown": ("最大ドローダウン", "pct"),
    "max_dd_days": ("最長ドローダウン期間(日)", "int"),
    "calmar": ("カルマーレシオ", "num"),
    "n_trades": ("取引回数(決済済)", "int"),
    "win_rate": ("勝率", "pct"),
    "avg_win": ("平均利益(円)", "yen"),
    "avg_loss": ("平均損失(円)", "yen"),
    "profit_factor": ("プロフィットファクター", "num"),
    "expectancy": ("1取引あたり期待値(円)", "yen"),
    "avg_hold_days": ("平均保有営業日数", "num"),
    "max_consecutive_losses": ("最大連敗数", "int"),
    "exposure": ("市場に居た日の割合", "pct"),
    "total_cost": ("手数料+スリッページ+貸株料", "yen"),
    "total_dividends": ("受取分配金(差引前)", "yen"),
    "total_tax": ("税金", "yen"),
}


def fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if kind == "pct":
        return f"{value * 100:,.2f}%"
    if kind == "yen":
        return f"{value:,.0f}円"
    if kind == "int":
        return f"{int(value):,}"
    return f"{value:,.2f}"
