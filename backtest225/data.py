"""価格データの取得・読み込み・クリーニング。

対応ソース:
  - Yahoo Finance（yfinance 経由の非公式API。精度・継続性は保証されない）
  - 手持ちの日足CSV（証券会社等からダウンロードしたもの）

戻り値はすべて DatetimeIndex を持ち、列 open/high/low/close/volume の DataFrame。
価格は「分割調整済み（最新の口数基準）」にそろえる。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKER = "1321.T"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
BUILTIN_DIVIDENDS = {"1321.T": DATA_DIR / "dividends_1321.csv"}


@dataclass(frozen=True)
class Split:
    """受益権（株式）分割。date はその比率で価格が切り替わる最初の営業日。"""

    date: pd.Timestamp
    ratio: float  # 1口 → ratio 口


# 1321 の既知の分割。
# 出典: 野村アセットマネジメント「上場投資信託(ETF)の受益権分割および売買単位変更に関するお知らせ」
#       2026年8月25日 https://nextfunds.jp/data/2026/td_260825a.pdf
#   - 1口を100口に分割（分割基準日 2026/10/6、効力発生日 2026/10/7）
#   - 東証の売買単位を 1口 → 10口 に変更（変更実施日 2026/10/5）
# Yahoo Finance 上は 2026/10/5 に分割が記録され、同日から新口数基準の価格になっている。
KNOWN_SPLITS: dict[str, list[Split]] = {
    "1321.T": [Split(pd.Timestamp("2026-10-05"), 100.0)],
}

# 売買単位の推移（分割調整後＝現在の口数基準で表した最小売買口数）。
# 2026/10/5 より前は 旧1口 = 新100口 が最小単位、以降は 10口。
KNOWN_LOT_SCHEDULE: dict[str, list[tuple[pd.Timestamp, int]]] = {
    "1321.T": [(pd.Timestamp("1900-01-01"), 100), (pd.Timestamp("2026-10-05"), 10)],
}

# CSVの列名ゆれを吸収する対応表（小文字化・空白除去後に照合）
_COLUMN_ALIASES = {
    "date": ["date", "日付", "年月日", "datetime", "time"],
    "open": ["open", "始値"],
    "high": ["high", "高値"],
    "low": ["low", "安値"],
    "close": ["close", "終値", "close price"],
    "volume": ["volume", "出来高", "vol"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        for std, aliases in _COLUMN_ALIASES.items():
            if key in aliases and std not in rename.values():
                rename[col] = std
                break
    return df.rename(columns=rename)


def load_csv(path: str | Path, encoding: str | None = None) -> pd.DataFrame:
    """日足CSVを読み込む。UTF-8 / Shift_JIS(cp932) を自動判定。"""
    path = Path(path)
    encodings = [encoding] if encoding else ["utf-8-sig", "cp932"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            raw = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError as e:  # 次のエンコーディングを試す
            last_err = e
    else:
        raise ValueError(f"CSVの文字コードを判定できません: {path}") from last_err

    df = _normalize_columns(raw)
    missing = {"date", "close"} - set(df.columns)
    if missing:
        raise ValueError(
            f"CSVに必要な列がありません: {sorted(missing)}。"
            f"検出した列: {list(raw.columns)}（日付・終値は必須。始値/高値/安値/出来高は推奨）"
        )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )
    return df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]


def fetch_yahoo(ticker: str = DEFAULT_TICKER, use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    """Yahoo Finance から日足を取得（非調整価格、分割のみ調整済み）。キャッシュあり。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{ticker.replace('.', '_')}.csv"
    if use_cache and cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df

    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("yfinance が未インストールです: pip install yfinance") from e

    raw = yf.download(ticker, period="max", auto_adjust=False, progress=False, actions=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo Finance からデータを取得できませんでした: {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(1, axis=1)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    if use_cache:
        df.to_csv(cache)
    return df


def fetch_yahoo_dividends(ticker: str, prices: pd.DataFrame) -> pd.Series:
    """Yahoo Finance の分配金（1口あたり、現在の口数基準へ換算）。

    Yahoo の分配金データは分割調整が不統一なことがある（1321では2025年以前が旧口数基準のまま）。
    そこで「分配金 / その日の終値 > 50%」のものは旧口数基準とみなし、既知の分割比率で割る。
    非公式データのため、正確な検証には運用会社公表値のCSVを使うこと。
    """
    import yfinance as yf

    div = yf.Ticker(ticker).dividends
    if div is None or div.empty:
        return pd.Series(dtype=float)
    div.index = pd.to_datetime(div.index).tz_localize(None).normalize()
    return normalize_dividends(div, prices, KNOWN_SPLITS.get(ticker, []))


def normalize_dividends(div: pd.Series, prices: pd.DataFrame, splits: list[Split]) -> pd.Series:
    div = div.astype(float).copy()
    closes = prices["close"].reindex(div.index, method="ffill")
    for sp in splits:
        mask = (div.index < sp.date) & (div / closes > 0.5)
        div[mask] = div[mask] / sp.ratio
    return div[div > 0]


def load_dividends_csv(path: str | Path) -> pd.Series:
    """分配金CSV（列: date, amount）。date は決算日(基準日)、amount は価格と同じ口数基準の1口あたり金額。"""
    df = pd.read_csv(path, comment="#")
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    df = df.rename(columns={"日付": "date", "権利落ち日": "date", "分配金": "amount"})
    s = pd.Series(pd.to_numeric(df["amount"], errors="coerce").values, index=pd.to_datetime(df["date"]))
    return s.dropna().sort_index()


T_PLUS_2_START = pd.Timestamp("2019-07-16")  # 受渡しがT+2になった約定日（JPX）


def to_ex_dates(div: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """基準日(決算日)の分配金を、権利落ち日（価格が分配金分下がる営業日）に付け替える。

    権利付最終日 = 受渡日が基準日以前になる最後の営業日（T+3、2019/7/16約定分からT+2）。
    権利落ち日 = その翌営業日。営業日は価格データの日付で近似する。
    データ期間外の分配金は除外する。
    """
    out = {}
    pos = np.arange(len(index))
    for rec, amt in div.items():
        rec = pd.Timestamp(rec).normalize()
        ok = []
        for j in pos:
            s = 2 if index[j] >= T_PLUS_2_START else 3
            if j + s < len(index) and index[j + s] <= rec:
                ok.append(j)
            elif j + s < len(index):
                break
        if not ok or ok[-1] + 1 >= len(index):
            continue
        ex = index[ok[-1] + 1]
        out[ex] = out.get(ex, 0.0) + float(amt)
    return pd.Series(out, dtype=float).sort_index()


def adjust_splits(df: pd.DataFrame, splits: list[Split]) -> tuple[pd.DataFrame, list[str]]:
    """未調整の価格（分割前後で水準が飛んでいる）を検出した場合のみ、分割前の価格を調整する。

    すでに調整済みのデータ（Yahoo等）には何もしない。
    """
    notes: list[str] = []
    df = df.copy()
    for sp in splits:
        before = df.loc[df.index < sp.date, "close"]
        after = df.loc[df.index >= sp.date, "close"]
        if before.empty or after.empty:
            continue
        jump = before.iloc[-1] / after.iloc[0]
        if jump > sp.ratio / 2:  # 分割前後で 1/ratio 近い急落 → 未調整と判断
            cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
            df.loc[df.index < sp.date, cols] /= sp.ratio
            if "volume" in df.columns:
                df.loc[df.index < sp.date, "volume"] *= sp.ratio
            notes.append(f"{sp.date.date()} の 1:{sp.ratio:g} 分割を検出し、それ以前の価格を調整しました")
    return df, notes


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """欠損・異常値の補正。補正内容は notes として返し、レポートに明記する。"""
    notes: list[str] = []
    df = df.copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["close"].notna() & (df["close"] > 0)]

    for c in ["open", "high", "low"]:
        if c not in df.columns:
            df[c] = np.nan
    if "volume" not in df.columns:
        df["volume"] = np.nan

    bad_open = df["open"].isna() | (df["open"] <= 0)
    if bad_open.any():
        n = int(bad_open.sum())
        last = df.index[bad_open][-1].date()
        df.loc[bad_open, "open"] = df["close"].shift(1)[bad_open]
        df["open"] = df["open"].fillna(df["close"])
        notes.append(
            f"始値が欠損/0の日が {n} 日（最終 {last}）あり、前日終値で代用しました（約定価格は近似）"
        )

    for c in ["high", "low"]:
        bad = df[c].isna() | (df[c] <= 0)
        if bad.any():
            notes.append(f"{c} が欠損/0の日が {int(bad.sum())} 日あり、始値・終値から補完しました")
            df.loc[bad, c] = np.nan
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    ret = df["close"].pct_change().abs()
    jumps = ret[ret > 0.2]
    if not jumps.empty:
        notes.append(
            "前日比±20%超の値動きがあります（データ異常/未処理の分割の可能性）: "
            + ", ".join(str(d.date()) for d in jumps.index[:5])
        )
    return df[["open", "high", "low", "close", "volume"]], notes


def load_prices(
    source: str = "yahoo",
    ticker: str = DEFAULT_TICKER,
    csv_path: str | None = None,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """データソースに応じて価格を読み込み、分割調整・クリーニングして返す。"""
    notes: list[str] = []
    if source == "csv":
        if not csv_path:
            raise ValueError("source=csv の場合は csv パスを指定してください")
        df = load_csv(csv_path)
        notes.append(f"データ: CSV {csv_path}")
    elif source == "yahoo":
        df = fetch_yahoo(ticker, refresh=refresh)
        notes.append(f"データ: Yahoo Finance {ticker}（非公式API。値は一次情報で要確認）")
    else:
        raise ValueError(f"未知のデータソース: {source}")

    df, n1 = adjust_splits(df, KNOWN_SPLITS.get(ticker, []))
    df, n2 = clean(df)
    notes += n1 + n2
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    if len(df) < 2:
        raise ValueError("指定期間のデータが2日未満です")
    return df, notes


def lot_size_series(index: pd.DatetimeIndex, ticker: str, lot_size: int | str | None) -> pd.Series:
    """各営業日の最小売買口数。lot_size='auto' なら既知のスケジュール、数値なら固定。"""
    if lot_size in (None, "auto"):
        sched = KNOWN_LOT_SCHEDULE.get(ticker)
        if sched is None:
            warnings.warn(f"{ticker} の売買単位が未登録のため 1 口単位として扱います")
            return pd.Series(1, index=index)
        s = pd.Series(np.nan, index=index)
        for d, lot in sched:
            s[index >= d] = lot
        return s.astype(int)
    return pd.Series(int(lot_size), index=index)
