"""手仕舞いの規則ごとに、資金を使い回すポートフォリオのトータルリターンを比べる
・手順0に従い、警戒サイン2個以上の週は買わない（1個の週は金額を半分）
・候補が多い週は配当利回りの高い順に、空きの数まで買う。1銘柄＝資産の 1/N（半額の週は 1/(2N)）"""
import sys, numpy as np, pandas as pd
from profit_lab import *
X, SETS = events(); X = X[(X.e + 189 + 2 < len(idx))].copy(); X["nw"] = [NW[pos[d]] for d in X.date]; X = X[X.nw <= 1].copy()
RULES = {"固定 21日": dict(hold=21), "固定 42日": dict(hold=42), "固定 63日": dict(hold=63), "固定 126日": dict(hold=126), "固定 189日": dict(hold=189),
 "25日線を回復で売り(上限63日)": dict(hold=63, sig=0.0), "25日線+6%で売り(上限126日)": dict(hold=126, sig=.06), "25日線+10%で売り(上限189日)": dict(hold=189, sig=.10),
 "損切り-25% 126日": dict(hold=126, stop=.25), "利確+30% 126日": dict(hold=126, tp=.30), "追従12%(+30%超で有効) 189日": dict(hold=189, trail=.12, trail_act=.30),
 "警戒サイン2個以上で売り 126日": dict(hold=126, crisis=True), "63日で-10%なら見切り 126日": dict(hold=126, tstop=(63, .10))}
dates_idx = idx
def simulate(trades, N, half=True):
    ent = {}
    for tr in trades: ent.setdefault(tr["e"], []).append(tr)
    t0 = min(ent); t1 = max(tr["e"] + len(tr["v"]) for tr in trades) + 1; t1 = min(t1, len(idx) - 1)
    cash = 1.0; open_ = []; eq = np.full(len(idx), np.nan); prev_eq = 1.0; taken = 0; inv = []
    for t in range(t0, t1 + 1):
        if t in ent and len(open_) < N:
            for tr in sorted(ent[t], key=lambda z: -z["prio"]):
                if len(open_) >= N: break
                alloc = min(cash, prev_eq / N * (0.5 if (half and tr["nw"] == 1) else 1.0))
                if alloc <= 1e-9: break
                cash -= alloc; open_.append([alloc, tr["e"], tr["v"]]); taken += 1
        tot = 0.0; keep = []
        for a, e, v in open_:
            k = t - e; val = a * v[min(k, len(v) - 1)]
            if k >= len(v) - 1: cash += val
            else: keep.append([a, e, v]); tot += val
        open_ = keep; prev_eq = cash + tot; eq[t] = prev_eq; inv.append(tot / prev_eq if prev_eq > 0 else 0)
    s = pd.Series(eq, index=idx).dropna(); return s, taken, float(np.mean(inv))
def metrics(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25; r = s.pct_change().dropna(); dd = (s / s.cummax() - 1).min()
    return dict(最終倍率=s.iloc[-1], 年率=(s.iloc[-1] ** (1 / yrs) - 1) * 100, 最大下落=dd * 100, 年率の変動=r.std() * np.sqrt(250) * 100, シャープ=r.mean() / r.std() * np.sqrt(250) if r.std() > 0 else np.nan)
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
out = {}
for sname, m in list(SETS.items())[:2]:
    S = X[m.reindex(X.index).fillna(False)]
    print("=" * 190); print(sname, f"（事例 {len(S)}件、2008年〜）  1銘柄=資産の1/N、候補が多い週は配当の高い順")
    for N in (10, 20):
        rows = []
        for rn, spec in RULES.items():
            trades = []
            for _, r in S.iterrows():
                o = run(int(r.e), int(r.j), with_path=True, **spec)
                if o is None: continue
                trades.append(dict(e=int(r.e), v=o[2], prio=r["配当利回り%"], nw=r.nw))
            s, taken, expo = simulate(trades, N); mt = metrics(s)
            # 前半・後半
            a = s[s.index < "2018-01-01"]; b = s[s.index >= "2018-01-01"]
            mt.update(rule=rn, 買った数=taken, 平均の投資割合=expo * 100, 前半の年率=(a.iloc[-1] / a.iloc[0]) ** (365.25 / (a.index[-1] - a.index[0]).days) * 100 - 100, 後半の年率=(b.iloc[-1] / b.iloc[0]) ** (365.25 / (b.index[-1] - b.index[0]).days) * 100 - 100)
            rows.append(mt); out[(sname, N, rn)] = s
        R = pd.DataFrame(rows).set_index("rule"); print(f"\n--- N={N} ---"); print(R.round(2).to_string())
bm = ew.reindex(next(iter(out.values())).index); yrs = (bm.index[-1] - bm.index[0]).days / 365.25
print("\n参考：対象銘柄の均等平均（持ち続けた場合）の年率 %.1f%%、最大下落 %.1f%%（%s〜%s）" % ((bm.iloc[-1] / bm.iloc[0]) ** (1 / yrs) * 100 - 100, (bm / bm.cummax() - 1).min() * 100, bm.index[0].date(), bm.index[-1].date()))
pd.to_pickle(out, SP + "profit_port_eq.pkl")
