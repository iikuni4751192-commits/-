import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
def blk_t(s):
    s = s.dropna(); b = s.groupby(((s.index - pd.Timestamp("2000-01-07")).days // 91)).mean(); return b.mean() / (b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 3 else np.nan
def within(D, f):
    d = D.dropna(subset=[f]); vals = {}
    for w, g_ in d.groupby("date"):
        if g_.up.any() and (~g_.up).any() and g_[f].nunique() > 1: vals[w] = g_.loc[g_.up, f].mean() - g_.loc[~g_.up, f].mean()
    wk = pd.Series(vals, dtype=float); wk.index = pd.to_datetime(wk.index)
    rk = d.groupby("date")[f].rank(pct=True); n = d.groupby("date")[f].transform("size"); m = n >= 6
    return blk_t(wk), blk_t(wk.loc[:"2017"]), blk_t(wk.loc["2018":]), d[m & (rk > 2/3)].up.mean() * 100, d[m & (rk <= 1/3)].up.mean() * 100
pd.set_option("display.width", 330); pd.set_option("display.max_rows", 200)
SETS = {}
X = pd.read_pickle(SP + "free_main.pkl"); SETS["主戦略（25日乖離≤-10%・決算の前後でない）"] = (X, X[(X["直近4週に決算"] == 0) & (X["2週以内に決算"] == 0)])
Y = pd.read_pickle(SP + "free_cci.pkl"); SETS["CCI20<-100戦術（配当3%以上・直後2週に決算なし）"] = (Y, Y[(Y["配当利回り%"] >= 3) & (Y["2週以内に決算"] == 0)])
res = {}
for nm, (full, D) in SETS.items():
    ST, MK, CAL = full.attrs["stock"], full.attrs["market"], full.attrs["cal"]
    for f in ST + MK + CAL: D[f] = pd.to_numeric(D[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
    print("=" * 150); print(f"{nm}  {len(D)}件・基準の勝率 {D.up.mean()*100:.1f}%（前半{D[D.date<'2018'].up.mean()*100:.1f}% 後半{D[D.date>='2018'].up.mean()*100:.1f}%）")
    rows = []
    for f in ST:
        if D[f].notna().sum() < 500: continue
        t, t1, t2, hi, lo = within(D, f); rows.append({"銘柄側の要因": f, "週内差t": t, "前半t": t1, "後半t": t2, "同じ週の上位1/3": hi, "下位1/3": lo, "差": hi - lo, "同向き": "○" if np.sign(t1) == np.sign(t2) else "×"})
    R = pd.DataFrame(rows).sort_values("週内差t", key=abs, ascending=False); print(R.round(2).to_string(index=False)); res[nm + "_stock"] = R
    rows = []; W = D.groupby("date").agg(up=("up", "mean"), wedge=("wedge", "mean"), n=("up", "size"))
    for f in MK + CAL:
        d = D.dropna(subset=[f]); wkv = d.groupby("date")[f].first()
        if wkv.nunique() < 6 or len(wkv) < 60: continue
        q = pd.qcut(wkv.rank(method="first"), 3, labels=["低", "中", "高"]) if wkv.nunique() > 3 else pd.cut(wkv, 3, labels=["低", "中", "高"])
        d = d.assign(q=d.date.map(q)); wr = d.groupby("q", observed=True).up.mean() * 100; we = d.groupby("q", observed=True).wedge.mean()
        if not {"低", "高"} <= set(wr.index): continue
        h1 = d[d.date < "2018"].groupby("q", observed=True).up.mean() * 100; h2 = d[d.date >= "2018"].groupby("q", observed=True).up.mean() * 100
        yr = d.groupby([d.date.dt.year, "q"], observed=True).up.mean().unstack(); yrs = yr.dropna(subset=["低", "高"]); sgn = np.sign(wr["高"] - wr["低"])
        ww = pd.DataFrame({"f": wkv, "up": W.up}).dropna(); b = ww.groupby((ww.index - pd.Timestamp("2000-01-07")).days // 91).mean()
        rows.append({"相場・カレンダー要因": f, "低": wr["低"], "中": wr.get("中", np.nan), "高": wr["高"], "高−低": wr["高"] - wr["低"], "前半": h1.get("高", np.nan) - h1.get("低", np.nan), "後半": h2.get("高", np.nan) - h2.get("低", np.nan),
                     "差(基準との)低/高": f"{we['低']:+.1f}/{we['高']:+.1f}", "年で同向き": f"{int((np.sign(yrs['高'] - yrs['低']) == sgn).sum())}/{len(yrs)}", "13週ブロック順位相関": b.corr(method="spearman").iloc[0, 1], "境目": f"{wkv.quantile(1/3):.2f}/{wkv.quantile(2/3):.2f}"})
    R2 = pd.DataFrame(rows); R2["安定"] = np.where((np.sign(R2["前半"]) == np.sign(R2["後半"])) & (R2["前半"].abs() >= 4) & (R2["後半"].abs() >= 4), "○", "")
    R2 = R2.sort_values("高−低", key=abs, ascending=False); print("\n" + R2.round(1).to_string(index=False)); res[nm + "_mk"] = R2
pd.to_pickle(res, SP + "free_eval_res.pkl")
