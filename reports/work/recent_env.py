"""RSI2/RSI14戦術：直近（2023年〜2026年7月）に効きやすかった条件（各戦術の必須条件の中）"""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
base = pd.read_pickle(SP + "wf_base_by_date.pkl")
def load(fn):
    X = pd.read_pickle(SP + fn)
    for c in X.columns:
        if X[c].dtype == object and c not in ("t", "sector"): X[c] = pd.to_numeric(X[c], errors="coerce")
    X["b_win"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[1]); X["b_mean"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[0])
    X = X.dropna(subset=["b_win", "r3m"]); X["wedge"] = X.up * 100 - X.b_win; X["edge"] = X.r3m - X.b_mean
    return X
def row(lab, d, full):
    r = lambda q: f"{q.up.mean()*100:5.1f}%({len(q):4d}) 差{q.wedge.mean():+5.1f}pt 平均{q.r3m.mean():+5.1f}%" if len(q) >= 15 else f"（{len(q)}件）"
    print(f"   {lab:26s} 直近 {r(d)}  ｜全期間 {r(full)}")
for which, fn in [("RSI2<10 戦術", "rsi2_extra_events.pkl"), ("RSI14<30 戦術", "RSI14lt30_events.pkl")]:
    X = load(fn); dv = X["配当利回り%"]; n4 = X["直近4週に決算"] == 0; n6 = n4 & (X["2週以内に決算"] == 0)
    ind30 = (X["業種内のシグナル銘柄割合%"] >= 30) | (X["業種比20日%"] > -5)
    B = X[(n4 & (dv >= 2.45))] if which.startswith("RSI2") else X[(n6 & ind30 & (dv >= 2.45))]
    R = B[B.date >= "2023-01-01"]
    print("=" * 100); print(which, f"必須条件の中。直近=2023年1月〜2026年7月のシグナル（{len(R)}件、基準＝同じ週の全銘柄）")
    print(f"   {'全体':26s} 直近 {R.up.mean()*100:5.1f}%({len(R)}) 差{R.wedge.mean():+5.1f}pt 平均{R.r3m.mean():+5.1f}%")
    rt = "米10年金利20日変化"
    if which.startswith("RSI2"):
        C = [("配当2.45〜3%", (dv < 3)), ("配当3〜4%", (dv >= 3) & (dv < 4)), ("配当4%以上", dv >= 4), ("配当5%以上", dv >= 5),
             ("米金利が20日で0.13以上低下", B[rt] <= -0.13), ("米金利が20日で0.06超上昇", B[rt] > 0.06),
             ("VIX 22超", B.VIX > 22), ("VIX 16未満", B.VIX < 16), ("25日乖離≤-10%も同時", B["25日乖離%"] <= -10),
             ("業界ぐるみ（業種内50%以上）", B["業種内のシグナル銘柄割合%"] >= 50), ("市場全体の急落（全体36%以上）", B["全体のシグナル銘柄割合%"] >= 36),
             ("配当4%以上×米金利低下", (dv >= 4) & (B[rt] <= -0.13)), ("配当4%以上×25日乖離も同時", (dv >= 4) & (B["25日乖離%"] <= -10))]
    else:
        C = [("配当2.45〜3%", (dv < 3)), ("配当3〜4%", (dv >= 3) & (dv < 4)), ("配当4%以上", dv >= 4), ("配当3.5%以上", dv >= 3.5),
             ("米金利が20日で0.27以上低下", B[rt] <= -0.27), ("米金利が20日で上昇(-0.03超)", B[rt] > -0.03),
             ("VIX 26.7超", B.VIX > 26.7), ("全体の23.5%以上にシグナル", B["全体のシグナル銘柄割合%"] >= 23.5),
             ("25日乖離≤-10%も同時", B["25日乖離%"] <= -10), ("25日乖離≤-10%が同時でない", B["25日乖離%"] > -10),
             ("配当3.5%以上×米金利が上昇でない", (dv >= 3.5) & (B[rt] <= -0.03))]
    for lab, m in C:
        m = m.reindex(B.index).fillna(False).astype(bool)
        row(lab, B[m & (B.date >= "2023-01-01")], B[m])
