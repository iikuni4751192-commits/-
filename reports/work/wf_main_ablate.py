"""主戦略：最終ルールから条件を1つずつ外した場合の、期間別の成績（wf_main.py の後に実行）"""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
X = pd.read_pickle(SP + "wf_main_events.pkl")
noe = (X["直近4週に決算"] == 0) & (X["2週以内に決算"] == 0); ind = (X["業種内のシグナル銘柄割合%"] >= 30) | (X["業種比20日%"] > -5)
hot = (X.sec1y > 60) | (X.r1y > 100); dv = X["配当利回り%"]
V = {"最終ルール": noe & ind & (dv >= 2.6) & ~hot, "業界ぐるみを外す": noe & (dv >= 2.6) & ~hot, "上がりすぎ除外を外す": noe & ind & (dv >= 2.6),
     "配当条件を外す": noe & ind & ~hot, "決算の除外を外す": ind & (dv >= 2.6) & ~hot, "配当3.5%以上に上げる": noe & ind & (dv >= 3.5) & ~hot, "全シグナル": pd.Series(True, index=X.index)}
def blk(d):
    w = d.groupby("date").edge.mean(); b = w.groupby((w.index - pd.Timestamp("2000-01-07")).days // 91).mean()
    return b.mean() / (b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 2 else np.nan
rows = []
for nm, m in V.items():
    r = {"ルール": nm}
    for lab, a, b in [("2008-12", 2008, 2012), ("2013-17", 2013, 2017), ("2018-22", 2018, 2022), ("2023-26", 2023, 2026)]:
        d = X[m & X.year.between(a, b)]
        r[f"{lab} 件数"] = len(d); r[f"{lab} 勝率"] = d.up.mean() * 100; r[f"{lab} 勝率の差"] = d.wedge.mean(); r[f"{lab} 平均の差"] = d.edge.mean()
    d = X[m & (X.year >= 2023)]; r["2023-26 t"] = blk(d); r["2023-26 週数"] = d.date.nunique()
    rows.append(r)
pd.set_option("display.width", 300); print(pd.DataFrame(rows).set_index("ルール").round(1).to_string())
