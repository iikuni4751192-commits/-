"""主戦略（25日乖離≤-10%）：指定銘柄の過去の同様のシグナルの1・3・6か月後の成績"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/-")
from weekpick.universe import load_universe
from weekpick import data as WD
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
T = sys.argv[1]
X = pd.read_pickle(SP + "wf_main_events.pkl"); X = X[X.t == T].copy()
u = load_universe("topix500"); p = WD.load_panel([T]); idx = p.close.index; Ca = (p.close * p.adjf)[T].ffill(limit=5); Oa = (p.open * p.adjf)[T]
pos = {d: i for i, d in enumerate(idx)}; COST = 0.001
def r(d, h):
    e = pos[d] + 1
    return ((Ca.iloc[e + h - 1] / Oa.iloc[e]) * (1 - COST) / (1 + COST) - 1) * 100 if e + h - 1 < len(idx) else np.nan
X["r1"] = [r(d, 21) for d in X.date]; X["r6"] = [r(d, 126) for d in X.date]
X["rt"] = pd.to_numeric(X["米10年金利20日変化"], errors="coerce")
noe = (X["直近4週に決算"] == 0) & (X["2週以内に決算"] == 0); ind = (X["業種内のシグナル銘柄割合%"] >= 30) | (X["業種比20日%"] > -5)
d0 = pd.Series(sorted(X.date.unique())); epm = dict(zip(d0, (d0.diff().dt.days > 91).cumsum())); X["ep"] = X.date.map(epm)
def st(D):
    if len(D) == 0: return "（0件）"
    E = D.groupby("ep").first()
    w = lambda s: f"{(s.dropna()>0).mean()*100:3.0f}%" if s.notna().any() else " - "
    return (f"{len(D):3d}件・{D.ep.nunique():2d}局面｜1か月 {w(D.r1)}（平均{D.r1.mean():+5.1f}%）｜3か月 {w(D.r3m)}（平均{D.r3m.mean():+5.1f}%、-10%以下{(D.r3m<=-10).mean()*100:3.0f}%）"
            f"｜6か月 {w(D.r6)}（平均{D.r6.mean():+5.1f}%）｜局面の最初の週の3か月 {w(E.r3m)}｜基準との差 {D.wedge.mean():+5.1f}pt")
print(f"{T} {u.set_index('ticker').at[T,'name']}  25日乖離≤-10%のシグナル（2008年〜2026年7月）")
for lab, m in [("すべて", pd.Series(True, index=X.index)), ("決算の前後でない", noe), ("決算の前後でない＋業界ぐるみ", noe & ind),
               ("＋米国金利が20日で0.13以上低下", noe & ind & (X.rt <= -0.13)), ("参考：決算の前後（除外される事例）", ~noe)]:
    print(f"  {lab:28s}", st(X[m]))
print("\n【局面ごと（決算の前後でない事例、各局面の最初の週）】")
D = X[noe]; E = D.groupby("ep").agg(最初=("date", "min"), 件数=("up", "size"), 乖離=("25日乖離%", "first"), 業界ぐるみ=("業種内のシグナル銘柄割合%", "first"),
                                     配当=("配当利回り%", "first"), 米金利=("rt", "first"), 一か月=("r1", "first"), 三か月=("r3m", "first"), 六か月=("r6", "first"))
print(E.round(1).to_string(index=False))
