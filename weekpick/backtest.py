"""週次の検証: 第k週の最終営業日の終値で採点 → 第k+1週の最初の営業日の始値で買い → 同じ週の最終営業日の終値で売る。

週は月曜〜日曜で区切る（祝日で月曜が休みなら火曜の始値、金曜が休みなら木曜の終値）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Panel
from .factors import compute_factors, eligible, score

LOT = 100  # 国内株式の売買単位（2018/10以降すべて100株に統一。それ以前の銘柄も100株として近似）


@dataclass
class Week:
    signal: pd.Timestamp  # 採点日（前週の最終営業日）
    entry: pd.Timestamp   # 買う日（その週の最初の営業日、始値）
    exit: pd.Timestamp    # 売る日（その週の最終営業日、終値）


def weeks(index: pd.DatetimeIndex) -> list[Week]:
    per = index.to_period("W-SUN")
    g = pd.Series(index, index=index).groupby(per)
    first, last = g.min(), g.max()
    out = []
    for k in range(1, len(first)):
        out.append(Week(last.iloc[k - 1], first.iloc[k], last.iloc[k]))
    return out


def week_returns(p: Panel, ws: list[Week], cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """各週・各銘柄の「月曜始値→金曜終値」の騰落率（配当込み・往復コスト控除後）と、買値（未調整の始値）。

    買う日に出来高0・始値なしの銘柄は買えないので NaN。売る日に値がなければ週内の最後の終値で売ったとみなす。
    """
    close_ff = (p.close * p.adjf).ffill(limit=4)
    rows, opens = {}, {}
    for w in ws:
        o = p.open.loc[w.entry].where(p.volume.loc[w.entry] > 0)
        buy = o * p.adjf.loc[w.entry]
        sell = close_ff.loc[w.exit]
        gross = sell / buy - 1
        rows[w.entry] = (1 + gross) * (1 - cost) / (1 + cost) - 1
        opens[w.entry] = o
    return pd.DataFrame(rows).T, pd.DataFrame(opens).T


@dataclass
class Setup:
    weights: dict[str, float]
    min_turnover: float = 5e8
    min_price: float = 0.0
    min_history: int = 260
    cost: float = 0.001  # 片道（手数料率+滑り）
    capital: float | None = None  # 指定時: 100株単位・資金内で買える銘柄だけ、端数は現金
    start: str | None = None
    end: str | None = None


@dataclass
class Prepared:
    weeks: list[Week]
    scores: pd.DataFrame     # 行=買う日（entry）, 列=銘柄
    returns: pd.DataFrame    # 行=買う日, 列=銘柄（コスト控除後）
    entry_open: pd.DataFrame
    signal_close: pd.DataFrame
    factors: dict[str, pd.DataFrame] = field(default_factory=dict)  # 行=買う日
    mask: pd.DataFrame | None = None


def prepare(p: Panel, s: Setup) -> Prepared:
    facs = compute_factors(p)
    mask = eligible(p, s.min_turnover, s.min_price, s.min_history)
    sc = score(facs, s.weights, mask)
    ws = weeks(p.close.index)
    if s.start:
        ws = [w for w in ws if w.entry >= pd.Timestamp(s.start)]
    if s.end:
        ws = [w for w in ws if w.exit <= pd.Timestamp(s.end)]
    if not ws:
        raise ValueError("検証期間に週がありません")
    ret, op = week_returns(p, ws, s.cost)
    sig = [w.signal for w in ws]
    ent = pd.DatetimeIndex([w.entry for w in ws])

    def by_entry(df):
        out = df.loc[sig]
        out.index = ent
        return out

    return Prepared(ws, by_entry(sc), ret, op, by_entry(p.close),
                    {k: by_entry(v) for k, v in facs.items()}, by_entry(mask))


def run(prep: Prepared, n: int, capital: float | None = None) -> pd.DataFrame:
    """上位 n 銘柄を均等に買った場合の週次成績。

    列: ret（ポートフォリオ）, universe（足切り通過・買える全銘柄の均等平均＝比較対象）, held（保有数）,
        invested（投資比率）, picks（銘柄）
    """
    rows = []
    for d in prep.scores.index:
        r = prep.returns.loc[d]
        tradable = r.notna()
        sc = prep.scores.loc[d].where(tradable).dropna()
        univ = r[sc.index].mean() if len(sc) else np.nan
        budget = capital / n if capital else None
        if budget:
            sc = sc[prep.signal_close.loc[d, sc.index] * LOT <= budget]
        picks = sc.nlargest(n).index
        if budget:
            op = prep.entry_open.loc[d, picks]
            shares = np.floor(budget / (op * LOT)) * LOT
            cost_in = shares * op
            pnl = float((cost_in * r[picks]).sum())
            ret, invested = pnl / capital, float(cost_in.sum()) / capital
        else:
            ret = float(r[picks].sum()) / n if len(picks) else 0.0
            invested = len(picks) / n
        rows.append({"date": d, "ret": ret, "universe": univ, "held": len(picks),
                     "invested": invested, "picks": " ".join(picks)})
    return pd.DataFrame(rows).set_index("date")


def stats(w: pd.DataFrame) -> dict:
    """週次成績の要約。年率は 52週で換算。"""
    r, u = w["ret"], w["universe"]
    ex = (r - u).dropna()
    eq = (1 + r).cumprod()
    years = len(r) / 52
    return {
        "weeks": len(r),
        "mean": r.mean(),
        "median": r.median(),
        "win_rate": (r > 0).mean(),
        "cagr": eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan,
        "vol": r.std() * np.sqrt(52),
        "sharpe": r.mean() / r.std() * np.sqrt(52) if r.std() > 0 else np.nan,
        "max_dd": float((eq / eq.cummax() - 1).min()),
        "worst_week": r.min(),
        "excess_mean": ex.mean(),
        "excess_t": ex.mean() / (ex.std() / np.sqrt(len(ex))) if len(ex) > 1 and ex.std() > 0 else np.nan,
        "beat_rate": (ex > 0).mean(),
        "avg_held": w["held"].mean(),
        "invested": w["invested"].mean(),
    }


def universe_stats(w: pd.DataFrame) -> dict:
    """比較対象（足切り通過銘柄の均等平均）を毎週持った場合。"""
    return stats(pd.DataFrame({"ret": w["universe"], "universe": w["universe"], "held": np.nan, "invested": 1.0}))


def factor_report(prep: Prepared, quantiles: int = 5) -> pd.DataFrame:
    """指標ごとに、毎週の対象銘柄を指標の大きさで5分位に分け、翌週の平均騰落率（週平均との差）を比べる。

    IC は指標と翌週騰落率の順位相関（週ごと）の平均。t値は週ごとの値のばらつきから計算。
    """
    out = []
    for name, f in prep.factors.items():
        qrets, ics = [], []
        for d in f.index:
            r = prep.returns.loc[d]
            x = f.loc[d].where(prep.mask.loc[d])
            ok = x.notna() & r.notna()
            if ok.sum() < quantiles * 5:
                continue
            x, y = x[ok], r[ok]
            q = pd.qcut(x.rank(method="first"), quantiles, labels=False)
            qrets.append((y.groupby(q).mean() - y.mean()).values)
            ics.append(x.rank().corr(y.rank()))
        if not qrets:
            continue
        qr, ic = np.array(qrets), pd.Series(ics)
        spread = qr[:, -1] - qr[:, 0]
        row = {"factor": name, "weeks": len(qr)}
        row.update({f"Q{i + 1}": qr[:, i].mean() for i in range(quantiles)})
        row["Q5-Q1" if quantiles == 5 else "top-bottom"] = spread.mean()
        row["t"] = spread.mean() / (spread.std() / np.sqrt(len(spread)))
        row["IC"] = ic.mean()
        row["IC_t"] = ic.mean() / (ic.std() / np.sqrt(len(ic)))
        out.append(row)
    return pd.DataFrame(out)
