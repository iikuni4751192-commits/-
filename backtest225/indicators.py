"""テクニカル指標。すべて「その日の終値までの情報」だけで計算する（未来参照なし）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: pd.Series, n: int) -> pd.Series:
    """単純移動平均"""
    return x.rolling(int(n), min_periods=int(n)).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    """指数移動平均"""
    return x.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def std(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(int(n), min_periods=int(n)).std(ddof=0)


def rsi(x: pd.Series, n: int = 14) -> pd.Series:
    """RSI（Wilder の平滑化）"""
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(down != 0, 100.0).where(up.notna())


def macd(x: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return ema(x, fast) - ema(x, slow)


def macd_signal(x: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    return ema(macd(x, fast, slow), signal)


def macd_hist(x: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    return macd(x, fast, slow) - macd_signal(x, fast, slow, signal)


def bb_upper(x: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """ボリンジャーバンド上限"""
    return sma(x, n) + k * std(x, n)


def bb_lower(x: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """ボリンジャーバンド下限"""
    return sma(x, n) - k * std(x, n)


def highest(x: pd.Series, n: int) -> pd.Series:
    """直近n日（当日含む）の最高値"""
    return x.rolling(int(n), min_periods=int(n)).max()


def lowest(x: pd.Series, n: int) -> pd.Series:
    """直近n日（当日含む）の最安値"""
    return x.rolling(int(n), min_periods=int(n)).min()


def roc(x: pd.Series, n: int) -> pd.Series:
    """n日変化率（%）"""
    return (x / x.shift(int(n)) - 1) * 100


def deviation(x: pd.Series, n: int) -> pd.Series:
    """移動平均乖離率（%）"""
    m = sma(x, n)
    return (x / m - 1) * 100


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def ref(x: pd.Series, n: int = 1) -> pd.Series:
    """n日前の値"""
    return x.shift(int(n))


def cross_up(a, b) -> pd.Series:
    """a が b を下から上に抜けた日に True"""
    a, b = _align(a, b)
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_down(a, b) -> pd.Series:
    """a が b を上から下に抜けた日に True"""
    a, b = _align(a, b)
    return (a < b) & (a.shift(1) >= b.shift(1))


def rising(x: pd.Series, n: int = 1) -> pd.Series:
    """n日連続で上昇"""
    return (x.diff() > 0).astype(int).rolling(int(n), min_periods=int(n)).sum() == n


def falling(x: pd.Series, n: int = 1) -> pd.Series:
    """n日連続で下落"""
    return (x.diff() < 0).astype(int).rolling(int(n), min_periods=int(n)).sum() == n


def _align(a, b):
    if isinstance(a, pd.Series) and not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    elif isinstance(b, pd.Series) and not isinstance(a, pd.Series):
        a = pd.Series(a, index=b.index)
    return a, b
