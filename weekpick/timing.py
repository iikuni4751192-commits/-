"""特定銘柄の週末の買いタイミング判定（RSI(2)<10 と、同時に出ている条件）。

判定の型は weekpick の検証と同じ: 週の最終営業日の終値で判定 → 翌週初日の寄りで買い → その週の最終営業日の引けで売り。
条件の選び方は 2008〜2026年の検証（ANA・あおぞら銀行・TOPIX500相当の全銘柄で向きがそろったもの）に基づく。
いずれも統計的に確かとは言えないため、判定は「参考」として扱う。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import weeks, week_returns
from .data import Panel
from .factors import eligible

TARGETS = {"9202.T": "ANAホールディングス", "8304.T": "あおぞら銀行"}
RSI2_THRESHOLD = 10.0


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    side: str  # "+": 上昇確率が上がる側 / "-": 下がる側 / "?": 1銘柄だけで目立った（偶然の可能性が高い）
    note: str


CONDITIONS = [
    Condition("dev25", "25日線乖離率 ≤ -10%", "+", "2銘柄・全銘柄とも上昇確率が高い。全銘柄では前半・後半とも+3.7ポイント"),
    Condition("wk10", "週間騰落率 ≤ -10%", "+", "2銘柄・全銘柄とも高い。全銘柄では効果が主に2018年以降"),
    Condition("rsi14", "RSI(14) < 30", "+", "2銘柄で約80%。全銘柄では+1.2ポイントと小さい"),
    Condition("mkt_drop", "市場全体（全銘柄平均）が週間-3%以下", "+", "全銘柄の効果は2018年以降に限られる"),
    Condition("ma75_up", "75日線が上向き（20日前比）", "-", "2銘柄・全銘柄とも上昇確率が低い（上昇トレンド中の軽い押し目）"),
    Condition("above200", "終値 > 200日線", "-", "あおぞらで大きく低下。ANA・全銘柄では差が小さい"),
    Condition("mkt_up", "市場全体は週間で上昇（この銘柄だけ下落）", "-", "ANA・全銘柄で低下。あおぞらでは逆"),
    Condition("sector_drop", "業種平均より週間で5%以上下落", "?", "あおぞらで17回中17回上昇。1銘柄だけの結果で偶然の可能性"),
    Condition("down4", "4日以上連続で下落", "?", "あおぞらで83.8%。ANA・全銘柄では差なし"),
]


def _rsi(x: pd.DataFrame, n: int) -> pd.DataFrame:
    d = x.diff()
    w = lambda s: s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + w(d.clip(lower=0)) / w((-d).clip(lower=0)))


def indicators(p: Panel, sectors: pd.Series) -> dict[str, pd.DataFrame]:
    """日次の指標（終値基準）と、CONDITIONS の各条件の真偽。sectors: ティッカー → 業種。"""
    C = p.close
    sma = lambda n: C.rolling(n, min_periods=n).mean()
    m25, m75, m200 = sma(25), sma(75), sma(200)
    wret = C / C.shift(5) - 1
    mkt = wret.mean(axis=1)  # 全銘柄の週間騰落率の単純平均
    sec_mean = wret.T.groupby(sectors.reindex(C.columns)).transform("mean").T
    bcast = lambda s: pd.DataFrame(np.repeat(s.to_numpy()[:, None], C.shape[1], axis=1), index=C.index, columns=C.columns)
    down = C.diff() < 0
    out = {
        "rsi2": _rsi(C, 2),
        "rsi14": _rsi(C, 14),
        "dev25_pct": (C / m25 - 1) * 100,
        "wret_pct": wret * 100,
        "mkt_wret_pct": bcast(mkt * 100),
        "sector_rel_pct": (wret - sec_mean) * 100,
    }
    flags = {
        "dev25": C / m25 - 1 <= -0.10,
        "wk10": wret <= -0.10,
        "rsi14": out["rsi14"] < 30,
        "mkt_drop": bcast(mkt <= -0.03),
        "ma75_up": m75 > m75.shift(20),
        "above200": C > m200,
        "mkt_up": bcast(mkt > 0),
        "sector_drop": wret - sec_mean <= -0.05,
        "down4": down & down.shift(1, fill_value=False) & down.shift(2, fill_value=False) & down.shift(3, fill_value=False),
    }
    out.update({f"flag_{k}": v for k, v in flags.items()})
    return out


def history(p: Panel, ind: dict[str, pd.DataFrame], tickers: list[str], start: str = "2008-01-01") -> pd.DataFrame:
    """過去のシグナル週（判定日に RSI(2)<10）の一覧と、翌週の騰落率（配当込み・コスト控除前）。"""
    mask = eligible(p)
    ws = [w for w in weeks(p.close.index) if w.entry >= pd.Timestamp(start)]
    ret, _ = week_returns(p, ws, cost=0.0)
    rows = []
    for w in ws:
        for t in tickers:
            r2 = ind["rsi2"].at[w.signal, t]
            r = ret.at[w.entry, t]
            if not (r2 < RSI2_THRESHOLD) or pd.isna(r) or not mask.at[w.signal, t]:
                continue
            row = {"ticker": t, "判定日": w.signal, "買い": w.entry, "売り": w.exit, "RSI2": r2, "翌週騰落率": r}
            row.update({c.key: bool(ind[f"flag_{c.key}"].at[w.signal, t]) for c in CONDITIONS})
            row["参考点"] = score(row)
            rows.append(row)
    return pd.DataFrame(rows)


def score(flags: dict) -> int:
    """「+」の条件数 - 「-」の条件数（「?」は数えない）。"""
    return sum(bool(flags[c.key]) for c in CONDITIONS if c.side == "+") - sum(
        bool(flags[c.key]) for c in CONDITIONS if c.side == "-")


def conditional_table(hist: pd.DataFrame) -> pd.DataFrame:
    """銘柄・条件ごとに、条件が出ていた/出ていなかったシグナル週の上昇確率。"""
    rows = []
    for t, g in hist.groupby("ticker"):
        up = g["翌週騰落率"] > 0
        for c in CONDITIONS:
            on = g[c.key]
            rows.append({
                "ticker": t, "条件": c.label, "向き": c.side,
                "あり_回数": int(on.sum()), "あり_上昇確率": up[on].mean() if on.any() else np.nan,
                "なし_回数": int((~on).sum()), "なし_上昇確率": up[~on].mean() if (~on).any() else np.nan,
            })
    return pd.DataFrame(rows)


def latest(ind: dict[str, pd.DataFrame], tickers: list[str], day: pd.Timestamp | None = None) -> pd.DataFrame:
    """判定日（既定は最新日）の状態。参考点 = 「+」の条件数 - 「-」の条件数（シグナルが出ていない銘柄は空欄）。"""
    day = day or ind["rsi2"].index[-1]
    rows = []
    for t in tickers:
        sig = bool(ind["rsi2"].at[day, t] < RSI2_THRESHOLD)
        row = {"ticker": t, "判定日": day.date(), "RSI2": ind["rsi2"].at[day, t], "シグナル": sig,
               "RSI14": ind["rsi14"].at[day, t], "25日線乖離%": ind["dev25_pct"].at[day, t],
               "週間騰落%": ind["wret_pct"].at[day, t], "市場の週間騰落%": ind["mkt_wret_pct"].at[day, t],
               "業種平均との差%": ind["sector_rel_pct"].at[day, t]}
        flags = {c.key: bool(ind[f"flag_{c.key}"].at[day, t]) for c in CONDITIONS}
        row.update({c.label: flags[c.key] for c in CONDITIONS})
        row["+条件の数"] = sum(flags[c.key] for c in CONDITIONS if c.side == "+")
        row["-条件の数"] = sum(flags[c.key] for c in CONDITIONS if c.side == "-")
        row["参考点"] = score(flags) if sig else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")


def trade_levels(p: Panel, ticker: str, entry: float | None = None, lookback: int = 10, rr: float = 2.0) -> dict:
    """損切り・利確の目安（損失:利益 = 1:rr）。
    A: 損切り＝直近 lookback 日の安値（買値の方が低ければ買値）から ATR(14)×0.1 下 / B: 損切り＝買値 − ATR(14)×2。
    entry を省略すると最新の終値を仮の買値にする（実際は翌週初日の寄りで決まる）。"""
    H, L, C = p.high[ticker], p.low[ticker], p.close[ticker]
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().iloc[-1]
    entry = float(entry if entry is not None else C.iloc[-1])
    low = min(float(L.iloc[-lookback:].min()), entry)  # 寄りで直近安値を割って始まった場合は買値を基準にする
    stop = low - 0.1 * atr
    risk = entry - stop
    risk_b = 2 * atr
    return {"買値": entry, "直近安値": low, "ATR14": atr,
            "A_損切り": stop, "A_損切り%": -risk / entry * 100, "A_利確": entry + rr * risk, "A_利確%": rr * risk / entry * 100,
            "B_損切り": entry - risk_b, "B_損切り%": -risk_b / entry * 100,
            "B_利確": entry + rr * risk_b, "B_利確%": rr * risk_b / entry * 100}
