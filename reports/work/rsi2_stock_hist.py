"""RSI2<10 戦術：指定銘柄の過去の同様のシグナル（必須条件を満たす事例）の3か月・6か月後の成績"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/-")
from weekpick.universe import load_universe
from weekpick import data as WD
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
codes = sys.argv[1:]
X = pd.read_pickle(SP + "rsi2_extra_events.pkl")
for c in X.columns:
    if X[c].dtype == object and c not in ("t", "sector"): X[c] = pd.to_numeric(X[c], errors="coerce")
base = pd.read_pickle(SP + "wf_base_by_date.pkl")
X["b_win"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[1]); X["b_mean"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[0])
u = load_universe("topix500"); p = WD.load_panel(list(u.ticker)); idx = p.close.index; Ca = (p.close * p.adjf).ffill(limit=5); Oa = (p.open * p.adjf)
pos = {d: i for i, d in enumerate(idx)}
COST = 0.001
def r6(t, d):
    e = pos[d] + 1
    if e + 125 >= len(idx): return np.nan
    return ((Ca[t].iloc[e + 125] / Oa[t].iloc[e]) * (1 - COST) / (1 + COST) - 1) * 100
rt = X["米10年金利20日変化"]
B = X[(X["直近4週に決算"] == 0) & (X["配当利回り%"] >= 2.45)].copy()
name = u.set_index("ticker")["name"]
rows = []; det = {}
for c in codes:
    t = c + ".T"; A = X[X.t == t].copy(); M = B[B.t == t].copy()
    M["r6"] = [r6(t, pd.Timestamp(d)) for d in M.date]
    d = pd.Series(sorted(M.date.unique())); ep = (d.diff().dt.days > 91).cumsum(); M["ep"] = M.date.map(dict(zip(d, ep)))
    E = M.groupby("ep").first()  # 各局面の最初のシグナル
    def wr(s): s = s.dropna(); return f"{(s>0).mean()*100:.0f}%" if len(s) else "-"
    rows.append({"コード": c, "銘柄": name[t][:10], "RSI2<10の全件": len(A), "全件の3か月勝率": wr(A.r3m),
                 "必須条件を満たす件数": len(M), "独立した局面": len(E), "3か月勝率": wr(M.r3m), "3か月平均%": round(M.r3m.mean(), 1),
                 "同じ週の全銘柄との差(勝率pt)": round((M.up * 100 - M.b_win).mean(), 1), "6か月勝率": wr(M.r6), "6か月平均%": round(M.r6.mean(), 1),
                 "局面の最初の週だけ3か月": wr(E.r3m), "米金利低下時3か月": wr(M[rt.reindex(M.index) <= -0.13].r3m) + f"({int((rt.reindex(M.index)<=-0.13).sum())}件)",
                 "前半(〜2017)": wr(M[M.date < "2018"].r3m), "後半(2018〜)": wr(M[M.date >= "2018"].r3m), "-10%以下": f"{(M.r3m<=-10).mean()*100:.0f}%"})
    det[c] = E[["date", "r3m", "r6", "配当利回り%", "米10年金利20日変化"]].assign(局面の件数=M.groupby("ep").size().values)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 30)
R = pd.DataFrame(rows); print(R.to_string(index=False))
print("\n参考：この戦術の必須条件を満たす全事例 3か月勝率 %.1f%%、配当4%%以上 %.1f%%" % (B.up.mean() * 100, B[B["配当利回り%"] >= 4].up.mean() * 100))
print("\n【局面ごと（各局面の最初のシグナル週）】")
for c, E in det.items():
    print(f"== {c} {name[c+'.T']}")
    print(E.round(2).to_string(index=False))
