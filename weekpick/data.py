"""複数銘柄の日足を yfinance でまとめて取得し、銘柄×日付の表（パネル）にする。

yfinance（非公式API）の値の扱い:
  - open/high/low/close/volume は株式分割を調整済み、配当は未調整
  - adj_close は配当も調整した終値。adjf = adj_close / close を「配当込み係数」として使い、
    週の損益に配当（権利落ちによる値下がり分）を含める
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest225.data import clean

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "jp"
FIELDS = ["open", "high", "low", "close", "volume", "adjf"]


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('.', '_')}.csv"


def fetch(tickers: list[str], refresh: bool = False, chunk: int = 50, start: str = "2005-01-01") -> list[str]:
    """未取得（refresh時は全部）の銘柄を取得してキャッシュに保存。取得できなかった銘柄を返す。"""
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if refresh or not _cache_path(t).exists()]
    failed: list[str] = []
    for i in range(0, len(todo), chunk):
        part = todo[i : i + chunk]
        print(f"  取得中 {i + 1}〜{i + len(part)} / {len(todo)}", flush=True)
        raw = yf.download(part, start=start, auto_adjust=False, progress=False, actions=False,
                          group_by="ticker", threads=True)
        for t in part:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.rename(columns=str.lower).dropna(subset=["close"])
            except KeyError:
                d = pd.DataFrame()
            if d.empty:
                failed.append(t)
                continue
            out = d[["open", "high", "low", "close", "volume"]].copy()
            out["adj_close"] = d["adj close"]
            out.index = pd.to_datetime(out.index).tz_localize(None)
            out.index.name = "date"
            out.to_csv(_cache_path(t))
        time.sleep(1)  # 連続アクセスを控えめに
    return failed


@dataclass
class Panel:
    """各フィールドが「行=日付, 列=ティッカー」の DataFrame。"""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    adjf: pd.DataFrame
    notes: list[str]

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)


SPLIT_RATIOS = [2, 3, 4, 5, 10, 20, 25, 50, 100]


def adjust_unadjusted_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """前後5営業日の終値の中央値が、ほぼ整数倍（2,3,4,5,10…）で段差になっている日を未調整の分割
    （または併合）とみなし、前後5日が平穏（日々の値動き10%未満）な場合に限ってそれより前の価格を割る（出来高は掛ける）。Yahoo が分割を反映し損ねる場合の対策。
    """
    df = df.copy()
    found: list[str] = []
    after_date = df.index[0]
    while True:
        c = df["close"]
        before = c.rolling(5, min_periods=3).median().shift(1)
        after = c[::-1].rolling(5, min_periods=3).median()[::-1]
        prev = c.shift(1)
        move = (c / prev - 1).abs()
        calm = (move.shift(1).rolling(5, min_periods=3).max() < 0.1) & (move[::-1].shift(1).rolling(5, min_periods=3).max()[::-1] < 0.1)
        hit = None
        for d, x in (before / after).dropna().items():
            if d <= after_date:
                continue
            inv = x >= 1
            k = x if inv else 1 / x
            best = min(SPLIT_RATIOS, key=lambda m: abs(k / m - 1))
            day = prev[d] / c[d]  # 当日の段差も同じ大きさか
            if calm[d] and k >= 1.8 and abs(k / best - 1) <= 0.06 and abs((day if inv else 1 / day) / best - 1) < 0.15:
                hit = (d, best, inv)
                break
        if hit is None:
            break
        d, best, inv = hit
        f = best if inv else 1 / best
        pre = df.index < d
        df.loc[pre, ["open", "high", "low", "close"]] /= f
        df.loc[pre, "volume"] *= f
        found.append(f"{d.date()} {'1:' if inv else ''}{best:g}{'' if inv else ':1'}")
        after_date = d
    return df, found


def drop_outliers(df: pd.DataFrame, ratio: float = 3.0, min_ref: float = 100.0) -> tuple[pd.DataFrame, int]:
    """直前の正常な終値（min_ref 円以上）から ratio 倍超・1/ratio 未満に飛んだ日を欠損にする。

    東証の値幅制限では、100円以上の株価が1日で3倍・3分の1になることはないため、配信データの誤り
    （上場廃止前後の異常値など）とみなす。100円未満の銘柄の暴落（例: 経営破綻時）は本物の値動きとして残す。
    30日以上データが途切れた後（再上場など）は基準を取り直す。
    """
    close = df["close"].to_numpy()
    idx = df.index
    bad = np.zeros(len(df), dtype=bool)
    ref, ref_date = np.nan, None
    for i, x in enumerate(close):
        if np.isnan(x):
            continue
        if ref_date is not None and (idx[i] - ref_date).days > 30:
            ref = np.nan
        if ref >= min_ref and (x / ref > ratio or x / ref < 1 / ratio):
            bad[i] = True
            continue
        ref, ref_date = x, idx[i]
    if bad.any():
        df = df.copy()
        df.loc[bad] = np.nan
    return df, int(bad.sum())


def dividend_factor(adj: pd.Series, close: pd.Series, max_yield: float = 0.2) -> tuple[pd.Series, int]:
    """配当込み係数（最新日=1）。adj_close / close の日々の変化のうち、配当として妥当なもの
    （権利落ち日に 0〜max_yield 相当だけ増える）だけを残す。Yahoo の adj_close は銘柄により
    桁外れ・負の値になることがあるため。2つ目の戻り値は異常を無視したかどうか（0/1）。
    """
    f = (adj / close).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    g = (f / f.shift(1)).fillna(1.0)
    ok = (g >= 1 - 1e-6) & (g <= 1 / (1 - max_yield))
    bad = int((~ok & ((g - 1).abs() > 1e-6)).any())
    g = g.where(ok, 1.0).clip(lower=1.0)
    cum = g.cumprod()
    return cum / cum.iloc[-1], bad


def build_panel(frames: dict[str, pd.DataFrame], min_coverage: float = 0.3) -> Panel:
    """銘柄ごとの日足（open..volume, adj_close）からパネルを作る。

    営業日は「全銘柄の半数近くが値を持つ日」とし、祝日に紛れ込んだ単独の行を除く。
    """
    notes: list[str] = []
    fixed = broken = 0
    dropped: list[str] = []
    splits: list[str] = []
    cols: dict[str, dict[str, pd.Series]] = {f: {} for f in FIELDS}
    for t, df in frames.items():
        adj = df["adj_close"] if "adj_close" in df.columns else df["close"]
        c, n = clean(df[["open", "high", "low", "close", "volume"]])
        fixed += sum(1 for x in n if "補正" in x or "代用" in x)
        c = c.where(c["volume"] > 0)  # 出来高0の日は約定がなく、配信値は埋め草のことがある（合併前後など）
        c, sp = adjust_unadjusted_splits(c)
        if sp:
            splits.append(f"{t}({', '.join(sp)})")
        c, k = drop_outliers(c)
        if k:
            dropped.append(f"{t}({k}日)")
        f, bad = dividend_factor(adj.reindex(c.index), df["close"].reindex(c.index))
        broken += bad
        for k in ["open", "high", "low", "close", "volume"]:
            cols[k][t] = c[k]
        cols["adjf"][t] = f
    wide = {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}
    count = wide["close"].notna().sum(axis=1)
    days = count.index[count >= count.rolling(20, min_periods=1).max() * min_coverage]
    wide = {k: v.reindex(days) for k, v in wide.items()}
    if splits:
        notes.append("未調整の株式分割とみられる段差を検出し、それ以前の価格を調整: " + ", ".join(splits))
    if dropped:
        notes.append("直前の正常な終値から1日で3倍超/3分の1未満に飛んだ日を除外: " + ", ".join(dropped))
    if broken:
        notes.append(f"配当込み終値に異常な変化があった {broken} 銘柄では、その日の配当調整を無視しました")
    if fixed:
        notes.append(f"{fixed} 銘柄でデータの欠損/桁ずれを補正しました（backtest225.data.clean と同じ処理）")
    return Panel(**wide, notes=notes)


def load_panel(tickers: list[str]) -> Panel:
    frames, missing = {}, []
    for t in tickers:
        p = _cache_path(t)
        if not p.exists():
            missing.append(t)
            continue
        frames[t] = pd.read_csv(p, index_col=0, parse_dates=True)
    if not frames:
        raise FileNotFoundError("価格データがありません。先に `python -m weekpick fetch <設定>` を実行してください")
    panel = build_panel(frames)
    if missing:
        panel.notes.append(f"価格データがない {len(missing)} 銘柄を除外: {', '.join(missing[:10])}"
                           + (" ほか" if len(missing) > 10 else ""))
    return panel
