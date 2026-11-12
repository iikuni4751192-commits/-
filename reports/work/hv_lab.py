"""高ボラ銘柄(60日年率ボラ・同日TOPIX500内の上位1/3)で、組み合わせると期待値・勝率が上がるシグナル探索（週次サンプリング、翌営業日始値で買い、63/126営業日後の終値）"""
import sys, numpy as np, pandas as pd, itertools
sys.path.insert(0, "/home/user/-")
from weekpick.universe import load_universe
from weekpick import data as WD
from weekpick.factors import eligible
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
COST = 0.001
u = load_universe("topix500"); p = WD.load_panel(list(u.ticker)); idx = p.close.index; mask = eligible(p)
A = p.adjf.ffill(); Ca = (p.close * A).ffill(limit=5); O = (p.open * A); Hh = p.high * A; Ll = p.low * A; V = p.volume; cols = list(Ca.columns)
dr = Ca.pct_change(fill_method=None); mkt = dr.where(mask.shift(1, fill_value=False)).mean(axis=1).fillna(0); ew = (1 + mkt).cumprod()
vol60 = dr.rolling(60, min_periods=45).std() * np.sqrt(250) * 100
wk = pd.Series(idx, index=idx).groupby([idx.isocalendar().year.values, idx.isocalendar().week.values]).max().values
wk = [d for d in wk if idx.get_loc(d) + 1 < len(idx)]; wi = np.array([idx.get_loc(d) for d in wk])
F = {}
dev = lambda n: (Ca / Ca.rolling(n).mean() - 1) * 100
F["25日乖離%"] = dev(25); F["75日乖離%"] = dev(75); F["200日乖離%"] = dev(200)
F["250日高値からの下落%"] = (Ca / Ca.rolling(250, min_periods=150).max() - 1) * 100
F["5日騰落%"] = Ca.pct_change(5) * 100; F["20日騰落%"] = Ca.pct_change(20) * 100; F["60日騰落%"] = Ca.pct_change(60) * 100; F["250日騰落%"] = Ca.pct_change(250) * 100
def rsi(n):
    d = Ca.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean(); return 100 * up / (up + dn).replace(0, np.nan)
F["RSI2"] = rsi(2); F["RSI14"] = rsi(14)
tp = (Hh + Ll + Ca * 1) / 3; sm = tp.rolling(20).mean(); mad = pd.DataFrame(np.nan, index=idx, columns=cols); tv = tp.to_numpy()
for i in wi:
    if i >= 19: w = tv[i - 19:i + 1]; mad.iloc[i] = np.nanmean(np.abs(w - np.nanmean(w, axis=0)), axis=0)
F["CCI20"] = (tp - sm) / (0.015 * mad)
F["出来高5日/60日"] = V.rolling(5).mean() / V.rolling(60).mean(); F["ボラ20日/250日"] = dr.rolling(20).std() / dr.rolling(250, min_periods=150).std()
F["60日ボラの水準"] = vol60; F["1日最大上昇%(20日)"] = dr.rolling(20).max() * 100; F["1日最大下落%(20日)"] = dr.rolling(20).min() * 100
F["20日中の下落日数"] = (dr < 0).rolling(20).sum()
rng = Hh.rolling(5).max() - Ll.rolling(5).min(); F["5日レンジ内位置"] = (Ca - Ll.rolling(5).min()) / rng.replace(0, np.nan)
F["寄り付きギャップ%(当日)"] = (O / Ca.shift(1) - 1) * 100
g = p.adjf / p.adjf.shift(1); Dv = (p.close.shift(1) * (1 - 1 / g)).where(g > 1 + 1e-6).fillna(0); F["配当利回り%"] = Dv.rolling("365D").sum() / p.close * 100
cov = lambda z, n=60: z.rolling(n, min_periods=40).mean(); mm = pd.DataFrame(np.repeat(mkt.values[:, None], len(cols), 1), index=idx, columns=cols)
F["市場との相関(60日)"] = (cov(dr * mm) - cov(dr) * cov(mm)) / (dr.rolling(60, min_periods=40).std() * mm.rolling(60, min_periods=40).std())
F["対市場20日超過%"] = F["20日騰落%"].sub(ew.pct_change(20) * 100, axis=0)
Ed = pd.read_csv("/home/user/-/reports/work/earnings_dates.csv"); Ed["d"] = pd.to_datetime(Ed.jst).dt.normalize(); Eg = {t: np.sort(g_.d.unique()) for t, g_ in Ed.groupby("ticker")}
# 市場
x = pd.concat([pd.read_pickle(SP + "crisis_ext.pkl"), pd.read_pickle(SP + "ext2.pkl")], axis=1).ffill(); x = x.loc[:, ~x.columns.duplicated()]; x = x.reindex(x.index.union(idx)).ffill().reindex(idx)
sh = ((dev(25) <= -10) & mask).sum(axis=1) / mask.sum(axis=1) * 100
M = pd.DataFrame(index=idx)
M["【市場】急落銘柄の割合%"] = sh; M["【市場】日本株の20日騰落%"] = ew.pct_change(20) * 100; M["【市場】日本株の200日乖離%"] = (ew / ew.rolling(200).mean() - 1) * 100
M["【市場】VIX"] = x["^VIX"]; M["【市場】VIXの5日変化"] = x["^VIX"].diff(5); M["【市場】米10年金利の20日変化"] = x["^TNX"].diff(20); M["【市場】ドル円の20日騰落%"] = x["JPY=X"].pct_change(20) * 100
M["【市場】S&P500の200日乖離%"] = (x["^GSPC"] / x["^GSPC"].rolling(200).mean() - 1) * 100; M["【市場】日本株の250日高値からの下落%"] = (ew / ew.rolling(250).max() - 1) * 100
# 組み立て
elig = (mask & vol60.notna() & O.notna()).to_numpy(); v = vol60.to_numpy(); Cn = Ca.to_numpy(); On = O.to_numpy()
rows = []
for k, i in enumerate(wi):
    ok = elig[i]; vv = np.where(ok, v[i], np.nan); thr = np.nanquantile(vv, 2 / 3); sel = np.flatnonzero(ok & (vv >= thr)); d = idx[i]
    r = {"date": d, "j": sel, "nuniv": ok.sum()}
    for h in (63, 126):
        e = i + 1; x_ = e + h - 1
        if x_ < len(idx):
            p0 = On[e, sel]; p1 = Cn[x_, sel]; r[f"r{h}"] = ((p1 / p0) * (1 - COST) / (1 + COST) - 1) * 100
        else: r[f"r{h}"] = np.full(len(sel), np.nan)
    rows.append(r)
recs = []
for r in rows:
    n = len(r["j"]); df = pd.DataFrame({"date": r["date"], "j": r["j"], "t": [cols[q] for q in r["j"]], "r63": r["r63"], "r126": r["r126"]}); recs.append(df)
D = pd.concat(recs, ignore_index=True); ri = idx.get_indexer(D.date); 
for k, df in F.items(): D[k] = df.to_numpy()[ri, D.j.to_numpy()]
for k in M.columns: D[k] = M[k].to_numpy()[ri]
nd = []; ld = []
for t, d in zip(D.t, D.date):
    a = Eg.get(t)
    if a is None: nd.append(np.nan); ld.append(np.nan); continue
    q = np.searchsorted(a, np.datetime64(d), side="right")
    nd.append((a[q] - np.datetime64(d)).astype("timedelta64[D]").astype(int) if q < len(a) else np.nan); ld.append((np.datetime64(d) - a[q - 1]).astype("timedelta64[D]").astype(int) if q > 0 else np.nan)
D["次の決算までの日数"] = nd; D["前回決算からの日数"] = ld
D.to_pickle(SP + "hv_panel.pkl"); print(D.shape, D.date.min(), D.date.max(), D.r63.notna().sum(), D.r126.notna().sum())
print("全期間の高ボラ群 平均 63日 %.2f 勝率 %.1f / 126日 %.2f 勝率 %.1f" % (D.r63.mean(), (D.r63.dropna() > 0).mean() * 100, D.r126.mean(), (D.r126.dropna() > 0).mean() * 100))
