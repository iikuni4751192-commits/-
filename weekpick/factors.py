"""評点に使う指標（すべて「その日の終値までの情報」だけで計算）と、合計スコア。

各指標は「値が大きいほど翌週に上がりやすい」と想定した向きにそろえてある。
本当にその向きで効くかは `python -m weekpick factors` で確かめること。
"""
from __future__ import annotations

import pandas as pd

from .data import Panel

# 名前 → 説明（レポート表示用）
FACTORS: dict[str, str] = {
    "reversal": "短期リバーサル: 直近5営業日の騰落率×(-1)（下げた銘柄ほど高い）",
    "momentum": "中期モメンタム: 12か月前→1か月前の騰落率（直近1か月は除く）",
    "high52": "52週高値への近さ: 終値 ÷ 過去250営業日の高値",
    "volume_surge": "出来高急増: 5日平均出来高 ÷ 60日平均出来高",
    "low_vol": "低ボラティリティ: 60日の日次騰落率の標準偏差×(-1)",
}


def compute_factors(p: Panel) -> dict[str, pd.DataFrame]:
    ac = p.close * p.adjf  # 配当込みの終値
    ah = p.high * p.adjf
    vol = p.volume.where(p.volume > 0)
    return {
        "reversal": -(ac / ac.shift(5) - 1),
        "momentum": ac.shift(21) / ac.shift(252) - 1,
        "high52": ac / ah.rolling(250, min_periods=200).max(),
        "volume_surge": vol.rolling(5, min_periods=3).mean() / vol.rolling(60, min_periods=40).mean(),
        "low_vol": -ac.pct_change(fill_method=None).rolling(60, min_periods=40).std(),
    }


def eligible(p: Panel, min_turnover: float = 5e8, min_price: float = 0.0, min_history: int = 260) -> pd.DataFrame:
    """足切り: 上場からの営業日数・直近20日の平均売買代金（円）・株価。"""
    turnover = (p.close * p.volume).rolling(20, min_periods=15).mean()
    history = p.close.notna().cumsum()
    return p.close.notna() & (history >= min_history) & (turnover >= min_turnover) & (p.close >= min_price)


def score(factors: dict[str, pd.DataFrame], weights: dict[str, float], mask: pd.DataFrame) -> pd.DataFrame:
    """日ごとに、足切りを通った銘柄の中での順位（0〜1）を重み付きで平均した値。

    重みが0でない指標がひとつでも欠けている銘柄は対象外（NaN）。
    """
    used = {k: w for k, w in weights.items() if w}
    unknown = set(used) - set(factors)
    if unknown:
        raise ValueError(f"未知の指標: {sorted(unknown)}（使えるもの: {', '.join(FACTORS)}）")
    if not used:
        raise ValueError("weights がすべて0です")
    total = None
    ok = mask.copy()
    for k, w in used.items():
        f = factors[k].where(mask)
        ok &= f.notna()
        r = f.rank(axis=1, pct=True) * w
        total = r if total is None else total + r
    return (total / sum(abs(w) for w in used.values())).where(ok)
