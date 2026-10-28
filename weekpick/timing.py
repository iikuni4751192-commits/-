"""特定銘柄の週末の買いタイミング判定（シグナルと、同時に出ている条件）。

判定の型は weekpick の検証と同じ: 週の最終営業日の終値で判定 → 翌週初日の寄りで買い → その週の最終営業日の引けで売り。
シグナルは SIGNALS（RSI(2)<10 / ウィリアムズ%R(14)<-80）から選び、同時に見る条件の組は
CONDITIONS（RSI(2)・ANA/あおぞら銀行向け）または WR_CONDITIONS（%R・SOMPO 向け）を使う。
条件の選び方は 2008〜2026年の検証に基づくが、いずれも統計的に確かとは言えないため、判定は「参考」として扱う。
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
WR_THRESHOLD = -80.0


@dataclass(frozen=True)
class Signal:
    key: str      # indicators() の値の名前
    label: str
    column: str   # 履歴・判定表での列名
    threshold: float

    def on(self, value) -> bool:
        return bool(value < self.threshold)


SIGNALS = {
    "rsi2": Signal("rsi2", "RSI(2) < 10", "RSI2", RSI2_THRESHOLD),
    "wr14": Signal("wr14", "ウィリアムズ%R(14) < -80", "%R14", WR_THRESHOLD),
}


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

# ウィリアムズ%R(14)<-80 と同時に見る条件（2008〜2026/10 の SOMPO・他の保険株・全銘柄の検証に基づく）
WR_CONDITIONS = [
    Condition("up3", "陽線3日連続", "+", "全銘柄で+6.0ポイント（前半+5.8・後半+5.9）。SOMPOは該当2回のみ"),
    Condition("dev25", "25日線乖離率 ≤ -10%", "+", "全銘柄で+3.0ポイント（前半+2.1・後半+4.2）。SOMPOは10回で-4.2ポイント"),
    Condition("sar_flip", "パラボリックSARが買い転換（週内）", "+", "全銘柄で+2.2ポイント（前半+3.3・後半+1.2）"),
    Condition("mkt_drop", "市場全体（全銘柄平均）が週間-3%以下", "+", "全銘柄で+6.2ポイント。ただし効果はほぼ2018年以降"),
    Condition("above75", "終値 > 75日線", "-", "他の保険株で-7.8ポイント、全銘柄で-1.2ポイント、SOMPOで-6.5ポイント"),
    Condition("mkt_above75", "市場全体が75日線より上", "-", "全銘柄で-3.5ポイント（後半-6.9）。SOMPOでは差なし"),
    Condition("shallow", "週間の下げが3%未満（浅い下げ）", "?", "SOMPOで71.7%（+11.8ポイント）。全銘柄では逆向き"),
    Condition("mkt_up", "市場全体は週間で上昇（この銘柄だけ下落）", "?", "SOMPOで72.2%（+10.5ポイント）。全銘柄では逆向き"),
    Condition("peer_wr", "同業種の他銘柄の半数以上も%R<-80（業種ぐるみの下げ）", "?", "他の保険株で+5.0ポイント。SOMPOは+1.2ポイント"),
    Condition("solo_wr", "同業種の他銘柄で%R<-80は25%未満（この銘柄だけの下げ）", "?", "SOMPOで70.6%（34回）。回数が少ない"),
]


def _sar_bull_flip(h: np.ndarray, l: np.ndarray, af0: float = 0.02, step: float = 0.02, amax: float = 0.2) -> np.ndarray:
    """パラボリックSAR（Wilder）が売り→買いに転換した日に True。"""
    out = np.zeros(len(h), dtype=bool)
    idx = np.where(np.isfinite(h) & np.isfinite(l))[0]
    if len(idx) < 3:
        return out
    hh, ll = h[idx], l[idx]
    bull, af, ep, sar = True, af0, hh[0], ll[0]
    for i in range(1, len(idx)):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, ll[i - 1], ll[max(i - 2, 0)])
            if ll[i] < sar:
                bull, sar, ep, af = False, ep, ll[i], af0
            elif hh[i] > ep:
                ep, af = hh[i], min(af + step, amax)
        else:
            sar = max(sar, hh[i - 1], hh[max(i - 2, 0)])
            if hh[i] > sar:
                bull, sar, ep, af = True, ep, hh[i], af0
                out[idx[i]] = True
            elif ll[i] < ep:
                ep, af = ll[i], min(af + step, amax)
    return out


def _rsi(x: pd.DataFrame, n: int) -> pd.DataFrame:
    d = x.diff()
    w = lambda s: s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + w(d.clip(lower=0)) / w((-d).clip(lower=0)))


def indicators(p: Panel, sectors: pd.Series) -> dict[str, pd.DataFrame]:
    """日次の指標（終値基準）と、CONDITIONS の各条件の真偽。sectors: ティッカー → 業種。"""
    C, O, H, L = p.close, p.open, p.high, p.low
    sma = lambda n: C.rolling(n, min_periods=n).mean()
    m25, m75, m200 = sma(25), sma(75), sma(200)
    wret = C / C.shift(5) - 1
    mkt = wret.mean(axis=1)  # 全銘柄の週間騰落率の単純平均
    sec_mean = wret.T.groupby(sectors.reindex(C.columns)).transform("mean").T
    bcast = lambda s: pd.DataFrame(np.repeat(s.to_numpy()[:, None], C.shape[1], axis=1), index=C.index, columns=C.columns)
    down = C.diff() < 0
    hh14, ll14 = H.rolling(14).max(), L.rolling(14).min()
    wr14 = -100 * (hh14 - C) / (hh14 - ll14)
    wr_on = (wr14 < WR_THRESHOLD).astype(float).where(wr14.notna())
    sec = sectors.reindex(C.columns)
    sec_sum = wr_on.T.groupby(sec).transform("sum").T
    sec_cnt = wr_on.notna().astype(float).T.groupby(sec).transform("sum").T
    peer_share = (sec_sum - wr_on.fillna(0)) / (sec_cnt - wr_on.notna()).replace(0, np.nan)  # 自分を除く同業種の割合
    mlev = (1 + C.pct_change(fill_method=None).mean(axis=1).fillna(0)).cumprod()  # 全銘柄の日次平均騰落の累積
    sar = pd.DataFrame({t: _sar_bull_flip(H[t].to_numpy(), L[t].to_numpy()) for t in C.columns}, index=C.index)
    inweek = lambda df: df.astype(float).rolling(5, min_periods=1).max() > 0
    out = {
        "rsi2": _rsi(C, 2),
        "wr14": wr14,
        "peer_wr_share_pct": peer_share * 100,
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
        "up3": (C > O) & (C.shift(1) > O.shift(1)) & (C.shift(2) > O.shift(2)),
        "sar_flip": inweek(sar),
        "above75": C > m75,
        "mkt_above75": bcast(mlev > mlev.rolling(75).mean()),
        "shallow": wret > -0.03,
        "peer_wr": peer_share >= 0.5,
        "solo_wr": peer_share < 0.25,
    }
    out.update({f"flag_{k}": v for k, v in flags.items()})
    return out


def history(p: Panel, ind: dict[str, pd.DataFrame], tickers: list[str], start: str = "2008-01-01",
            signal: str = "rsi2", conditions: list[Condition] = CONDITIONS) -> pd.DataFrame:
    """過去のシグナル週（判定日にシグナルあり）の一覧と、翌週の騰落率（配当込み・コスト控除前）。"""
    sg = SIGNALS[signal]
    mask = eligible(p)
    ws = [w for w in weeks(p.close.index) if w.entry >= pd.Timestamp(start)]
    ret, _ = week_returns(p, ws, cost=0.0)
    rows = []
    for w in ws:
        for t in tickers:
            v = ind[sg.key].at[w.signal, t]
            r = ret.at[w.entry, t]
            if not sg.on(v) or pd.isna(r) or not mask.at[w.signal, t]:
                continue
            row = {"ticker": t, "判定日": w.signal, "買い": w.entry, "売り": w.exit, sg.column: v, "翌週騰落率": r}
            row.update({c.key: bool(ind[f"flag_{c.key}"].at[w.signal, t]) for c in conditions})
            row["参考点"] = score(row, conditions)
            rows.append(row)
    return pd.DataFrame(rows)


def score(flags: dict, conditions: list[Condition] = CONDITIONS) -> int:
    """「+」の条件数 - 「-」の条件数（「?」は数えない）。"""
    return sum(bool(flags[c.key]) for c in conditions if c.side == "+") - sum(
        bool(flags[c.key]) for c in conditions if c.side == "-")


def conditional_table(hist: pd.DataFrame, conditions: list[Condition] = CONDITIONS) -> pd.DataFrame:
    """銘柄・条件ごとに、条件が出ていた/出ていなかったシグナル週の上昇確率。"""
    rows = []
    for t, g in hist.groupby("ticker"):
        up = g["翌週騰落率"] > 0
        for c in conditions:
            on = g[c.key]
            rows.append({
                "ticker": t, "条件": c.label, "向き": c.side,
                "あり_回数": int(on.sum()), "あり_上昇確率": up[on].mean() if on.any() else np.nan,
                "なし_回数": int((~on).sum()), "なし_上昇確率": up[~on].mean() if (~on).any() else np.nan,
            })
    return pd.DataFrame(rows)


def latest(ind: dict[str, pd.DataFrame], tickers: list[str], day: pd.Timestamp | None = None,
           signal: str = "rsi2", conditions: list[Condition] = CONDITIONS) -> pd.DataFrame:
    """判定日（既定は最新日）の状態。参考点 = 「+」の条件数 - 「-」の条件数（シグナルが出ていない銘柄は空欄）。"""
    sg = SIGNALS[signal]
    day = day or ind[sg.key].index[-1]
    rows = []
    for t in tickers:
        sig = sg.on(ind[sg.key].at[day, t])
        row = {"ticker": t, "判定日": day.date(), "RSI2": ind["rsi2"].at[day, t], "%R14": ind["wr14"].at[day, t],
               "シグナル": sig, "RSI14": ind["rsi14"].at[day, t], "25日線乖離%": ind["dev25_pct"].at[day, t],
               "週間騰落%": ind["wret_pct"].at[day, t], "市場の週間騰落%": ind["mkt_wret_pct"].at[day, t],
               "業種平均との差%": ind["sector_rel_pct"].at[day, t],
               "同業種で%R<-80の割合%": ind["peer_wr_share_pct"].at[day, t]}
        flags = {c.key: bool(ind[f"flag_{c.key}"].at[day, t]) for c in conditions}
        row.update({c.label: flags[c.key] for c in conditions})
        row["+条件の数"] = sum(flags[c.key] for c in conditions if c.side == "+")
        row["-条件の数"] = sum(flags[c.key] for c in conditions if c.side == "-")
        row["参考点"] = score(flags, conditions) if sig else np.nan
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


def condition_effect(p: Panel, ind: dict[str, pd.DataFrame], signal: str, conditions: list[Condition],
                     tickers: list[str] | None = None, start: str = "2008-01-01", split: str = "2018-01-01") -> pd.DataFrame:
    """シグナル週を条件の有無で分け、翌週の上昇確率を比べる（既定は全銘柄）。

    差 = 条件あり − 条件なし（ポイント）。前半/後半は split の前後。z は2つの割合の差の検定（同じ週の銘柄は
    独立でないため、市場全体の条件では z が大きく出すぎる点に注意）。
    """
    sg = SIGNALS[signal]
    mask = eligible(p)
    ws = [w for w in weeks(p.close.index) if w.entry >= pd.Timestamp(start)]
    ret, _ = week_returns(p, ws, cost=0.0)
    sig_days = [w.signal for w in ws]
    cols = tickers or list(p.close.columns)
    R = ret[cols].to_numpy()
    ok = mask.loc[sig_days, cols].to_numpy() & np.isfinite(R)
    E = (ind[sg.key].loc[sig_days, cols].to_numpy() < sg.threshold) & ok
    UP = R > 0
    late = np.array([w.entry >= pd.Timestamp(split) for w in ws])[:, None]
    rows = []
    for c in conditions:
        X = ind[f"flag_{c.key}"].loc[sig_days, cols].fillna(False).to_numpy().astype(bool)
        row = {"条件": c.label, "向き": c.side}
        for lab, m in [("", np.ones_like(late)), ("前半_", ~late), ("後半_", late)]:
            e = E & m
            n1, n0 = int((e & X).sum()), int((e & ~X).sum())
            k1, k0 = int((e & X & UP).sum()), int((e & ~X & UP).sum())
            p1 = k1 / n1 if n1 else np.nan
            p0 = k0 / n0 if n0 else np.nan
            if lab == "":
                pp = (k1 + k0) / max(n1 + n0, 1)
                se = np.sqrt(pp * (1 - pp) * (1 / max(n1, 1) + 1 / max(n0, 1)))
                row.update({"あり_回数": n1, "あり_上昇確率": p1, "なし_上昇確率": p0,
                            "z": (p1 - p0) / se if n1 and n0 and se > 0 else np.nan})
            row[f"{lab}差"] = (p1 - p0) * 100 if n1 and n0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
