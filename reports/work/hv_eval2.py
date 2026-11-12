import numpy as np, pandas as pd, itertools, pickle
exec(open("hv_eval.py").read().split("pd.set_option")[0])
cond = pickle.load(open(SP + "hv_cond.pkl", "rb"))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
# --- 市場条件(絶対水準、高ボラ群の平均との差)
base = {h: D[f"r{h}"].mean() for h in (63, 126)}
def evabs(m, h):
    r = D[f"r{h}"].to_numpy(); ok = m & ~np.isnan(r)
    if ok.sum() < 3000: return None
    wkm = pd.Series(r[ok]).groupby(blk[ok]).mean(); t = (wkm.mean() - np.nanmean(r)) / (wkm.std() / np.sqrt(len(wkm)))
    a = ok & first; b = ok & ~first
    return dict(n=int(ok.sum()), 週数=D.date[ok].nunique(), 平均=r[ok].mean(), 勝率=(r[ok] > 0).mean() * 100, 前半平均=r[a].mean(), 後半平均=r[b].mean(), 下位5pct=np.nanpercentile(r[ok], 5), t_vs全体=t)
rows = []
for k, m in cond.items():
    if k.startswith("【市場】"):
        for h in (63, 126):
            e = evabs(m, h)
            if e: rows.append(dict(条件=k, 期間=h, **e))
M = pd.DataFrame(rows)
for h in (63, 126): print(f"\n=== 市場条件 {h}日(高ボラ全体 平均{base[h]:.2f}) ==="); print(M[M.期間 == h].sort_values("平均", ascending=False).round(2).to_string(index=False))
# --- 組み合わせ探索：単独の候補（銘柄特徴のみ）
S = pd.read_pickle(SP + "hv_single.pkl"); S = S[~S.条件.str.startswith("【市場】")]
cand = S.groupby("条件").apply(lambda g: (g.同週比 > 0).all() and (g.前半同週比 > 0).all() and (g.後半同週比 > 0).all()); cand = list(cand[cand].index)
cand = [c for c in cand if "ボラ" not in c or "ボラ20" in c]  # 60日ボラ水準そのものは外す(高ボラ群内で更に高ボラは別扱い)
print("\n候補の単独条件数:", len(cand)); 
def evw(m, h, lo=None):
    r = D[f"r{h}"].to_numpy(); w = D[f"w{h}"].to_numpy(); ok = m & ~np.isnan(r)
    if lo is not None: ok = ok & lo
    if ok.sum() < 1500: return None
    bw = pd.Series(w[ok]).groupby(blk[ok]).mean(); t = bw.mean() / (bw.std() / np.sqrt(len(bw)))
    return dict(n=int(ok.sum()), 週数=D.date[ok].nunique(), 平均=r[ok].mean(), 勝率=(r[ok] > 0).mean() * 100, 同週比=w[ok].mean(), t=t, 前半=w[ok & first].mean(), 後半=w[ok & ~first].mean(), 下位5=np.nanpercentile(r[ok], 5))
res = []; ntest = 0
for a, b in itertools.combinations(cand, 2):
    if a.rsplit(" ", 1)[0] == b.rsplit(" ", 1)[0]: continue
    m = cond[a] & cond[b]; ntest += 1
    for h in (63, 126):
        e = evw(m, h)
        if e: res.append(dict(条件=a + " ＋ " + b, 期間=h, **e))
P = pd.DataFrame(res); print("検定した組み合わせ数:", ntest)
P.to_pickle(SP + "hv_pairs.pkl")
for h in (63, 126):
    g = P[(P.期間 == h) & (P.前半 > 0) & (P.後半 > 0)].sort_values("平均", ascending=False)
    print(f"\n=== 2条件の組み合わせ {h}日 (前半・後半とも同週比>0、平均順) ==="); print(g.head(20).round(2).to_string(index=False))
    # ホールドアウト: 前半(〜2017)で選び後半で検証
pickle.dump(cand, open(SP + "hv_cand.pkl", "wb"))
