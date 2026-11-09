"""RSI2/RSI14戦術：年ごとの基準との差（wf_tactic.py と同じ事例・基準）"""
import sys, numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
base = pd.read_pickle(SP + "wf_base_by_date.pkl")
for which, fn in [("rsi2", "rsi2_extra_events.pkl"), ("rsi14", "RSI14lt30_events.pkl")]:
    X = pd.read_pickle(SP + fn)
    for c in X.columns:
        if X[c].dtype == object and c not in ("t", "sector"): X[c] = pd.to_numeric(X[c], errors="coerce")
    X["b_mean"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[0]); X["b_win"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[1])
    X = X.dropna(subset=["b_mean", "r3m"]); X["wedge"] = X.up * 100 - X.b_win; X["edge"] = X.r3m - X.b_mean; X["y"] = X.date.dt.year
    dv = X["配当利回り%"]; n4 = X["直近4週に決算"] == 0; n6 = n4 & (X["2週以内に決算"] == 0); ind = (X["業種内のシグナル銘柄割合%"] >= 30) | (X["業種比20日%"] > -5)
    V = {"最終ルール": (n4 & (dv >= 2.45)) if which == "rsi2" else (n6 & ind & (dv >= 2.45))}
    V["配当4%以上"] = (n4 & (dv >= 4)) if which == "rsi2" else (n6 & ind & (dv >= 4))
    print("=" * 60, which)
    rows = []
    for y in range(2013, 2027):
        r = {"年": y}
        for k, m in V.items():
            d = X[m & (X.y == y)]; r[f"{k} 件数"] = len(d); r[f"{k} 勝率"] = d.up.mean() * 100; r[f"{k} 勝率の差"] = d.wedge.mean(); r[f"{k} 平均の差"] = d.edge.mean()
        rows.append(r)
    T = pd.DataFrame(rows).set_index("年"); print(T.round(1).to_string())
    for k in V: print(f"  {k}: 勝率の差がプラスの年 {(T[k+' 勝率の差']>0).sum()}/{T[k+' 勝率の差'].notna().sum()}、平均の差がプラスの年 {(T[k+' 平均の差']>0).sum()}/{T[k+' 平均の差'].notna().sum()}")
