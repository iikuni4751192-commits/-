"""個別銘柄：13週後に上昇した週と下落した週の『その時点の状態』を比べ、現在がどちらに近いかを見る"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/-")
from weekpick.universe import load_universe
from weekpick import data as WD
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
u = load_universe("topix500"); p = WD.load_panel(list(u.ticker)); C = p.close * p.adjf.ffill(); idx = C.index
info = u.set_index("ticker"); sec = info.reindex(C.columns)["sector"]
x = pd.read_pickle(SP + "crisis_ext.pkl").ffill(); e1 = pd.read_pickle(SP + "ext.pkl").ffill()
ext = pd.concat([x, e1[["CL=F"]]], axis=1); ext = ext.loc[:, ~ext.columns.duplicated()]; xd = ext.reindex(ext.index.union(idx)).ffill().reindex(idx)
g = p.adjf / p.adjf.shift(1); Dv = (p.close.shift(1) * (1 - 1 / g)).where(g > 1 + 1e-6).fillna(0); YLD = Dv.rolling("365D").sum() / p.close * 100
wild = lambda s, n: s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
def rsi(s, n): d = s.diff(); return 100 - 100 / (1 + wild(d.clip(lower=0), n) / wild((-d).clip(lower=0), n))
mkt = C.pct_change(fill_method=None).mean(axis=1).fillna(0).add(1).cumprod()
for T in sys.argv[1:]:
    s = C[T].dropna(); peers = [c for c in C.columns if sec[c] == sec[T] and c != T]
    secI = C[peers].pct_change(fill_method=None).mean(axis=1).fillna(0).add(1).cumprod().reindex(s.index)
    F = pd.DataFrame(index=s.index)
    F["25日乖離%"] = (s / s.rolling(25).mean() - 1) * 100; F["200日乖離%"] = (s / s.rolling(200).mean() - 1) * 100
    F["RSI14"] = rsi(s, 14); F["20日騰落%"] = s.pct_change(20) * 100; F["60日騰落%"] = s.pct_change(60) * 100; F["1年騰落%"] = s.pct_change(250) * 100
    F["52週高値比%"] = (s / s.rolling(250).max() - 1) * 100
    F["業種比60日%"] = F["60日騰落%"] - secI.pct_change(60) * 100; F["業種200日乖離%"] = (secI / secI.rolling(200).mean() - 1) * 100
    m = mkt.reindex(s.index); F["市場200日乖離%"] = (m / m.rolling(200).mean() - 1) * 100; F["市場20日%"] = m.pct_change(20) * 100
    F["配当利回り%"] = YLD[T].reindex(s.index); F["20日ボラ%"] = s.pct_change().rolling(20).std() * np.sqrt(250) * 100
    F["VIX"] = xd["^VIX"].reindex(s.index); F["米金利20日変化"] = xd["^TNX"].diff(20).reindex(s.index); F["ドル円20日%"] = xd["JPY=X"].pct_change(20).reindex(s.index) * 100
    F["原油20日%"] = xd["CL=F"].pct_change(20).reindex(s.index) * 100; F["銀行株20日%"] = xd["1615.T"].pct_change(20).reindex(s.index) * 100
    fut = (s.shift(-63) / s - 1) * 100
    W = F.resample("W-FRI").last(); fw = fut.resample("W-FRI").last(); W = W[W.index >= "2009-01-01"]
    now = W.iloc[-1]; Wd = W.iloc[:-14].copy(); Wd["fut"] = fw.reindex(Wd.index); Wd = Wd.dropna(subset=["fut"])
    up = Wd[Wd.fut > 0]; dn = Wd[Wd.fut <= 0]
    print(f"\n######## {T} {info.at[T,'name']}  対象 {len(Wd)}週（2009年〜）  13週後に上昇 {len(up)}週（{len(up)/len(Wd)*100:.0f}%）／下落 {len(dn)}週")
    rows = []
    for c in F.columns:
        a, b = up[c].dropna(), dn[c].dropna()
        # 13週ブロックで重なりを考慮した t（ブロック平均の差）
        blk = lambda z: z.groupby((z.index - pd.Timestamp("2009-01-02")).days // 91).mean()
        A, B = blk(a), blk(b); diff = a.mean() - b.mean()
        se = np.sqrt(A.var() / len(A) + B.var() / len(B)); t = diff / se if se > 0 else np.nan
        # 今の値の位置：上昇時と下落時の分布のどちらに近いか（z距離）
        za = abs(now[c] - a.median()) / (a.quantile(.75) - a.quantile(.25) + 1e-9); zb = abs(now[c] - b.median()) / (b.quantile(.75) - b.quantile(.25) + 1e-9)
        # 今の値に近い週（同じ分位の範囲）の上昇率
        q = Wd[c].rank(pct=True); qn = (Wd[c] < now[c]).mean(); near = Wd[(q > qn - 0.1) & (q < qn + 0.1)]
        rows.append({"指標": c, "上昇した週の中央値": round(a.median(), 1), "下落した週の中央値": round(b.median(), 1), "差のt(粗い)": round(t, 1),
                     "現在": round(now[c], 1), "現在は": "上昇側" if za < zb else "下落側", "現在に近い値の週の13週後上昇率": f"{(near.fut>0).mean()*100:.0f}%（{len(near)}）",
                     "前半/後半の差の向き": ("同" if np.sign(up.loc[:'2017', c].mean() - dn.loc[:'2017', c].mean()) == np.sign(up.loc['2018':, c].mean() - dn.loc['2018':, c].mean()) else "逆")})
    pd.set_option("display.width", 300); print(pd.DataFrame(rows).to_string(index=False))
