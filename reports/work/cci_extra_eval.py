import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
X = pd.read_pickle(SP + "cci_extra_events.pkl"); ST, MK = X.attrs["stock"], X.attrs["market"]
base = pd.read_pickle(SP + "wf_base_by_date.pkl")
X["b_win"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[1]); X["wedge"] = X.up * 100 - X.b_win
for f in ST + MK: X[f] = pd.to_numeric(X[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
def blk_t(s):
    s = s.dropna(); b = s.groupby(((s.index - pd.Timestamp("2000-01-07")).days // 91)).mean(); return b.mean() / (b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 3 else np.nan
def within(D, f):
    d = D.dropna(subset=[f]); vals = {}
    for w, g_ in d.groupby("date"):
        if g_.up.any() and (~g_.up).any(): vals[w] = g_.loc[g_.up, f].mean() - g_.loc[~g_.up, f].mean()
    wk = pd.Series(vals, dtype=float); wk.index = pd.to_datetime(wk.index)
    rk = d.groupby("date")[f].rank(pct=True); n = d.groupby("date")[f].transform("size"); m = n >= 6
    return blk_t(wk), blk_t(wk.loc[:"2017"]), blk_t(wk.loc["2018":]), d[m & (rk > 2/3)].up.mean() * 100, d[m & (rk <= 1/3)].up.mean() * 100
pd.set_option("display.width", 330); pd.set_option("display.max_rows", 100)
print(f"事例 {len(X)}／銘柄側の要因 {len(ST)}（目安：|t|>3.1＝約{len(ST)+len(MK)}要因の補正、かつ前半・後半で同じ向き）")
rows = []
for f in ST:
    t, t1, t2, hi, lo = within(X, f); rows.append({"銘柄側の要因": f, "週内差t": t, "前半t": t1, "後半t": t2, "同じ週の上位1/3の勝率": hi, "下位1/3の勝率": lo, "差": hi - lo})
R = pd.DataFrame(rows).sort_values("週内差t", key=abs, ascending=False); print(R.round(2).to_string(index=False))
# 配当と独立か（配当3%以上＋決算なしの中で）
B = X[(X["配当利回り%"] >= 3) & (X["直近4週に決算"] == 0) & (X["2週以内に決算"] == 0)]
print(f"\n【配当3%以上＋決算なし（{len(B)}件）の中での上位要因】")
for f in R["銘柄側の要因"].head(8):
    t, t1, t2, hi, lo = within(B, f); print(f"  {f:28s} t {t:+.2f}（前半{t1:+.2f}／後半{t2:+.2f}） 上位1/3 {hi:.1f}% 下位1/3 {lo:.1f}%")
# 配当との相関
print("\n配当利回りとの順位相関:", {f[:14]: round(X[[f, "配当利回り%"]].corr(method="spearman").iloc[0, 1], 2) for f in R["銘柄側の要因"].head(8)})
print("\n=== 相場側の要因（週ごとに同じ値。3等分の勝率と、同じ週の全銘柄との差、前半・後半） ===")
rows = []
W = X.groupby("date").agg(up=("up", "mean"), wedge=("wedge", "mean"), n=("up", "size"))
for f in MK:
    d = X.dropna(subset=[f]); wkv = d.groupby("date")[f].first()
    q = pd.qcut(wkv.rank(method="first"), 3, labels=["低", "中", "高"]); d = d.assign(q=d.date.map(q))
    wr = d.groupby("q", observed=True).up.mean() * 100; we = d.groupby("q", observed=True).wedge.mean()
    h1 = d[d.date < "2018"].groupby("q", observed=True).up.mean() * 100; h2 = d[d.date >= "2018"].groupby("q", observed=True).up.mean() * 100
    # 週単位の相関（13週ブロック平均で）
    ww = pd.DataFrame({"f": wkv, "up": W.up}).dropna(); b = ww.groupby((ww.index - pd.Timestamp("2000-01-07")).days // 91).mean(); rb = b.corr(method="spearman").iloc[0, 1]
    rows.append({"相場側の要因": f, "低": wr["低"], "中": wr["中"], "高": wr["高"], "高−低": wr["高"] - wr["低"], "前半 高−低": h1["高"] - h1["低"], "後半 高−低": h2["高"] - h2["低"],
                 "基準との差 低/高": f"{we['低']:+.1f}/{we['高']:+.1f}", "13週ブロックの順位相関": rb, "境目": f"{wkv.quantile(1/3):.2f}/{wkv.quantile(2/3):.2f}"})
print(pd.DataFrame(rows).sort_values("高−低", key=abs, ascending=False).round(1).to_string(index=False))
print("\n月別:", (X.groupby("月").up.mean() * 100).round(1).to_dict())
print("月別（基準との差）:", X.groupby("月").wedge.mean().round(1).to_dict())
print("月内（上旬/中旬/下旬）:", {k: round(v, 1) for k, v in (X.groupby(pd.cut(X["月内の日"], [0, 10, 20, 31])).up.mean() * 100).items()})
