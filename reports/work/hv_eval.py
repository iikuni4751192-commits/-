import numpy as np, pandas as pd, itertools
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
D = pd.read_pickle(SP + "hv_panel.pkl"); D["date"] = pd.to_datetime(D.date)
skip = {"date", "j", "t", "r63", "r126"}; feats = [c for c in D.columns if c not in skip]
mkt = [c for c in feats if c.startswith("【市場】")]
# 条件の定義：銘柄特徴は同日・高ボラ群内の順位で下位1/3・上位1/3、市場特徴は過去1年の位置(下位/上位1/3)
D = D.sort_values(["date"]).reset_index(drop=True); R = {}
for c in feats:
    if c in mkt:
        s = D.groupby("date")[c].first(); pr = s.rolling(250, min_periods=150).rank(pct=True); R[c] = D.date.map(pr)
    else: R[c] = D.groupby("date")[c].rank(pct=True)
cond = {}
for c in feats:
    cond[f"{c} 低(下位1/3)"] = (R[c] <= 1 / 3).fillna(False).to_numpy(); cond[f"{c} 高(上位1/3)"] = (R[c] >= 2 / 3).fillna(False).to_numpy()
for lab, a, b in [("決算直後(0〜28日)", 0, 28), ("決算まで14日以内", None, 14)]:
    pass
cond["決算が直近4週以内"] = (D["前回決算からの日数"] <= 28).to_numpy(); cond["決算が次の2週以内"] = (D["次の決算までの日数"] <= 14).to_numpy()
cond["決算の前後なし(直近4週・次2週とも無し)"] = ((D["前回決算からの日数"] > 28) & (D["次の決算までの日数"] > 14)).to_numpy()
cond["配当利回り3%以上"] = (D["配当利回り%"] >= 3).to_numpy()
for h in (63, 126): D[f"w{h}"] = D[f"r{h}"] - D.groupby("date")[f"r{h}"].transform("mean")
dates = np.array(sorted(D.date.unique())); wk13 = pd.Series(np.arange(len(dates)) // 13, index=dates); blk = D.date.map(wk13).to_numpy()
y = D.date.dt.year.to_numpy(); first = y <= 2017
def ev(m, h):
    r = D[f"r{h}"].to_numpy(); w = D[f"w{h}"].to_numpy(); ok = m & ~np.isnan(r)
    if ok.sum() < 300: return None
    bw = pd.Series(w[ok]).groupby(blk[ok]).mean(); t = bw.mean() / (bw.std() / np.sqrt(len(bw))) if len(bw) > 3 and bw.std() > 0 else np.nan
    a = ok & first; b = ok & ~first
    return dict(n=int(ok.sum()), 週数=D.date[ok].nunique(), 平均=r[ok].mean(), 勝率=(r[ok] > 0).mean() * 100, 同週比=w[ok].mean(), t=t, 前半同週比=w[a].mean() if a.sum() > 100 else np.nan, 後半同週比=w[b].mean() if b.sum() > 100 else np.nan, 下位5pct=np.nanpercentile(r[ok], 5))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 80)
allm = np.ones(len(D), bool)
print("高ボラ全体:", {h: {k: round(v, 2) for k, v in ev(allm, h).items()} for h in (63, 126)})
rows = []
for k, m in cond.items():
    for h in (63, 126):
        e = ev(m, h)
        if e: rows.append(dict(条件=k, 期間=h, **e))
S = pd.DataFrame(rows); S.to_pickle(SP + "hv_single.pkl")
for h in (63, 126):
    g = S[S.期間 == h].sort_values("同週比", ascending=False)
    print(f"\n=== 単独条件 {h}日 同週比の上位 ==="); print(g.head(25).round(2).to_string(index=False))
    print(f"\n=== 単独条件 {h}日 同週比の下位(避けるべき) ==="); print(g.tail(12).round(2).to_string(index=False))
import pickle; pickle.dump(cond, open(SP + "hv_cond.pkl", "wb"))
