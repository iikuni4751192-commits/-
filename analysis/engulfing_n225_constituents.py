"""包み足（陰陽・陽陰）のシグナル検証を、日経225の構成銘柄すべてで行う。
python analysis/engulfing_n225_constituents.py [--refresh]

・構成銘柄: 日経の公式ファイル（nikkei_stock_average_weight_jp.csv）を実行時に取得する。
  日経の著作物のためリポジトリには保存しない（data/cache/ に一時保存、.gitignore 済み）。
  「現在の」構成銘柄で過去を検証するため生存者バイアスがある。
・株価: Yahoo Finance（yfinance、非公式API）の日足（分割調整あり・配当調整なし）。値は一次情報で要確認。
  出来高0の行と四本値が同値の行を除外し、backtest225.data.clean で桁ずれ等を補正。
・パターン定義・地合い・保有期間は analysis/engulfing_signal_test.py と同じ。
  結果 = 翌日始値 → h 営業日後終値の騰落率
・比較:
  対基準 … 各銘柄の「全営業日の同じ騰落率」との差（上昇確率・平均）。銘柄ごとの差を全件で平均
  対市場 … 同じ期間の日経平均の騰落率との差（相場全体の動きを除いた超過騰落）から、
           その銘柄の全営業日の同じ超過騰落の平均を引いたもの（現構成銘柄が指数を上回りやすい
           生存者バイアスを打ち消す）
・p値: 同じ日に多数の銘柄で出たシグナルは相場全体の動きを共有するため、シグナル日でまとめた
  クラスター頑健標準誤差で計算（正規近似、両側）。h>1 では期間の重なりで p 値は甘めに出る。
"""
import argparse
import io
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest225 import data as D  # noqa: E402
from engulfing_after_bull_run import load  # noqa: E402
from engulfing_signal_test import HORIZONS, patterns  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
LIST_URL = "https://indexes.nikkei.co.jp/nkave/archives/file/nikkei_stock_average_weight_jp.csv"


def constituents(refresh: bool) -> pd.DataFrame:
    path = CACHE / "nikkei225_constituents.csv"
    if refresh or not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(LIST_URL, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=60).read()
        path.write_text(raw.decode("cp932"), encoding="utf-8")
    df = pd.read_csv(path, dtype=str).dropna(subset=["コード"])
    return df[df["コード"].str.fullmatch(r"[0-9A-Z]{4}")]


def prices(codes: list[str], refresh: bool) -> dict[str, pd.DataFrame]:
    path = CACHE / "nikkei225_prices.pkl"
    if refresh or not path.exists():
        import yfinance as yf

        raw = yf.download([c + ".T" for c in codes], period="max", auto_adjust=False,
                          progress=False, actions=False, group_by="ticker", threads=True)
        raw.to_pickle(path)
    raw = pd.read_pickle(path)
    out = {}
    for c in codes:
        t = c + ".T"
        if t not in raw.columns.get_level_values(0):
            continue
        x = raw[t].rename(columns=str.lower).dropna(subset=["close", "open", "high", "low"])
        x = x[(x["volume"] > 0) & ~((x["open"] == x["close"]) & (x["high"] == x["low"]))]
        if len(x) < 300:
            continue
        x.index = pd.to_datetime(x.index).tz_localize(None)
        x.index.name = "date"
        x, _ = D.clean(x)
        out[c] = x
    return out


def events(code: str, df: pd.DataFrame, mkt_ret: dict[int, pd.Series]) -> pd.DataFrame:
    rows = []
    pats = patterns(df)
    for h in HORIZONS:
        r = df["close"].shift(-h) / df["open"].shift(-1) - 1
        # データ異常（未処理の分割等）を含む期間を除外
        bad = (df["close"].pct_change().abs() > 0.5).astype(int)
        bad_win = bad[::-1].rolling(h + 1, min_periods=1).max()[::-1].shift(-1).fillna(1).astype(bool)
        r = r.where(~bad_win)
        base = r.dropna()
        p0, m0 = float((base > 0).mean()), float(base.mean())
        mk = mkt_ret[h].reindex(df.index)
        mx_all = (r - mk).dropna()
        mx0 = float(mx_all.mean())  # 銘柄の全営業日の対市場超過（生存者バイアス等）を差し引く
        for name, sig in pats.items():
            s = sig & r.notna()
            if not s.any():
                continue
            rows.append(pd.DataFrame({
                "code": code, "h": h, "pattern": name, "date": df.index[s],
                "ret": r[s].values, "up_ex": (r[s] > 0).astype(float).values - p0,
                "ret_ex": r[s].values - m0, "mkt_ex": (r[s] - mk[s]).values - mx0,
            }))
    return pd.concat(rows) if rows else pd.DataFrame()


def cluster_mean(x: pd.Series, groups: pd.Series) -> tuple[float, float]:
    """平均とクラスター頑健 p 値（両側、正規近似）。"""
    x = x.astype(float)
    n = len(x)
    m = x.mean()
    g = (x - m).groupby(groups.values).sum()
    se = math.sqrt((g**2).sum()) / n * math.sqrt(len(g) / max(len(g) - 1, 1))
    p = math.erfc(abs(m / se) / math.sqrt(2)) if se > 0 else math.nan
    return m, p


def per_stock_hits(ev: pd.DataFrame) -> pd.DataFrame:
    """銘柄ごとに対基準平均の t 検定を行い、期待方向に p<0.05 となった銘柄の割合。"""
    rows = []
    for (h, pat), g in ev.groupby(["h", "pattern"]):
        good = bad = tested = 0
        for _, s in g.groupby("code"):
            if len(s) < 10:
                continue
            tested += 1
            t = s["ret_ex"].mean() / (s["ret_ex"].std(ddof=1) / math.sqrt(len(s)))
            if abs(t) > 1.96:
                expect_up = pat.startswith("陰陽")
                good += (t > 0) == expect_up
                bad += (t > 0) != expect_up
        rows.append({"h": h, "パターン": pat, "銘柄数": tested,
                     "期待方向で有意%": round(100 * good / tested, 1),
                     "逆方向で有意%": round(100 * bad / tested, 1)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)

    lst = constituents(args.refresh)
    data = prices(lst["コード"].tolist(), args.refresh)
    n225 = load("^N225")
    mkt = {h: n225["close"].shift(-h) / n225["open"].shift(-1) - 1 for h in HORIZONS}
    print(f"構成銘柄リスト: {lst['日付'].iloc[0]} 時点 {len(lst)} 銘柄、価格を取得・検証できた銘柄 {len(data)}")
    starts = pd.Series({c: d.index[0] for c, d in data.items()})
    print(f"データ期間: 開始 {starts.min().date()}〜{starts.max().date()}（中央値 {starts.median().date()}）、"
          f"終了 {max(d.index[-1] for d in data.values()).date()}\n")

    ev = pd.concat([events(c, d, mkt) for c, d in data.items()], ignore_index=True)

    rows = []
    for (h, pat), g in ev.groupby(["h", "pattern"], sort=False):
        up, p_up = cluster_mean(g["up_ex"], g["date"])
        ex, p_ex = cluster_mean(g["ret_ex"], g["date"])
        mx, p_mx = cluster_mean(g["mkt_ex"].dropna(), g.loc[g["mkt_ex"].notna(), "date"])
        rows.append({
            "h": h, "パターン": pat, "件数": len(g), "銘柄": g["code"].nunique(),
            "上昇確率%": round(100 * (g["ret"] > 0).mean(), 1),
            "対基準 確率差pt": round(100 * up, 2), "p(確率)": round(p_up, 4),
            "対基準 平均差%": round(100 * ex, 3), "p(平均)": round(p_ex, 4),
            "対市場 超過%": round(100 * mx, 3), "p(対市場)": round(p_mx, 4),
        })
    res = pd.DataFrame(rows)
    n_tests = 3 * len(res)
    alpha = 0.05 / n_tests
    print(f"■ 全銘柄プール（検定数 {n_tests}、Bonferroni 補正後の有意水準 {alpha:.5f}）")
    print(res.to_string(index=False))

    pcols = ["p(確率)", "p(平均)", "p(対市場)"]
    sig = res[(res[pcols] < alpha).any(axis=1)]
    print("\n■ Bonferroni 補正後も有意な行")
    print(sig.to_string(index=False) if len(sig) else "なし")

    print("\n■ 銘柄ごとの検定（対基準平均、|t|>1.96）。偶然だけなら各方向 約2.5%")
    print(per_stock_hits(ev).to_string(index=False))

    print("\n■ 期間別（対基準 平均差%・対市場 超過%、h=5）")
    e5 = ev[ev["h"] == 5].copy()
    e5["期間"] = pd.cut(e5["date"], pd.to_datetime(["1990", "2008", "2017", "2027"]),
                       labels=["〜2007", "2008〜2016", "2017〜"], right=False)
    tab = e5.groupby(["pattern", "期間"], observed=True).agg(
        件数=("ret", "size"), 平均差=("ret_ex", "mean"), 対市場=("mkt_ex", "mean"))
    tab[["平均差", "対市場"]] = (100 * tab[["平均差", "対市場"]]).round(3)
    print(tab.to_string())


if __name__ == "__main__":
    main()
