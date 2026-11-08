"""RSI2<10戦術・RSI14<30戦術：200日線との上下関係（銘柄・市場）の影響を、各戦術の必須条件を満たす事例で検証"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/-")
from weekpick.universe import load_universe
from weekpick import data as WD
from weekpick.factors import eligible
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
COST = 0.001; net = lambda r: (1 + r) * (1 - COST) / (1 + COST) - 1
u = load_universe("topix500"); p = WD.load_panel(list(u.ticker)); idx = p.close.index; mask = eligible(p)
A = p.adjf.ffill(); Oa = (p.open * A).to_numpy(); Ca = (p.close * A).ffill(limit=5); Cl = Ca.to_numpy()
pos = {d: i for i, d in enumerate(idx)}; col = {t: j for j, t in enumerate(Ca.columns)}
ma = Ca.rolling(200, min_periods=150).mean()
dist = ((Ca / ma - 1) * 100).to_numpy(); slope = ((ma / ma.shift(20) - 1) * 100).to_numpy()
ew = (1 + Ca.pct_change(fill_method=None).where(mask.shift(1, fill_value=False)).mean(axis=1).fillna(0)).cumprod()
mkt = (ew / ew.rolling(200).mean() - 1) * 100
x = pd.read_pickle(SP + "crisis_ext.pkl").ffill(); x = x.reindex(x.index.union(idx)).ffill().reindex(idx)
spx = (x["^GSPC"] / x["^GSPC"].rolling(200).mean() - 1) * 100
# 全銘柄の200日線より上の割合（市場の広がり）
breadth = ((Ca > ma).where(mask & ma.notna()).mean(axis=1) * 100)
def load(name):
    X = pd.read_pickle(SP + name)
    for c in X.columns:
        if X[c].dtype == object and c not in ("t", "sector"): X[c] = pd.to_numeric(X[c], errors="coerce")
    return X
X2 = load("rsi2_extra_events.pkl"); B2 = X2[(X2["直近4週に決算"] == 0) & (X2["配当利回り%"] >= 2.45)].copy()
X14 = load("RSI14lt30_events.pkl")
noe = (X14["直近4週に決算"] == 0) & (X14["2週以内に決算"] == 0); ind = (X14["業種内のシグナル銘柄割合%"] >= 30) | (X14["業種比20日%"] > -5)
B14 = X14[noe & ind & (X14["配当利回り%"] >= 2.45)].copy()
def prep(B):
    B = B.copy(); B["i"] = [pos[d] for d in B.date]; B["e"] = B.i + 1; B = B[B.e + 126 < len(idx)].copy(); B["j"] = [col[t] for t in B.t]
    b0 = Oa[B.e, B.j]; B["r3"] = net(Cl[B.e + 62, B.j] / b0 - 1) * 100; B["r6"] = net(Cl[B.e + 125, B.j] / b0 - 1) * 100; B["up"] = B.r3 > 0
    B["dist"] = dist[B.i, B.j]; B["slope"] = slope[B.i, B.j]
    B["mkt"] = mkt.reindex(B.date).values; B["spx"] = spx.reindex(B.date).values; B["breadth"] = breadth.reindex(B.date).values
    return B.dropna(subset=["dist"])
def s(d, c="r3"):
    x_ = d[c].dropna()
    if len(x_) < 20: return f"（{len(x_)}件：少なすぎ）"
    a = d[d.date < "2018"][c]; b = d[d.date >= "2018"][c]
    return f"{(x_>0).mean()*100:5.1f}%（{len(x_):5d}件） 平均{x_.mean():+5.1f}% -10%以下{(x_<=-10).mean()*100:3.0f}%  前半{(a>0).mean()*100:5.1f}%({len(a)}) 後半{(b>0).mean()*100:5.1f}%({len(b)})"
def blk_t(s_):
    s_ = s_.dropna(); b = s_.groupby(((s_.index - pd.Timestamp("2000-01-07")).days // 91)).mean(); return b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))
def within(d, f):
    d = d.dropna(subset=[f]); wk = pd.Series({w: g.loc[g.up, f].mean() - g.loc[~g.up, f].mean() for w, g in d.groupby("date") if g.up.any() and (~g.up).any() and g[f].nunique() > 1}).sort_index()
    rk = d.groupby("date")[f].rank(pct=True); n = d.groupby("date")[f].transform("size"); m = n >= 3
    return f"同じ週の中で比べた t {blk_t(wk):+.2f}（前半{blk_t(wk.loc['2008':'2017']):+.2f}／後半{blk_t(wk.loc['2018':'2026']):+.2f}）  同じ週の上位1/3 {d[m & (rk>2/3)].up.mean()*100:.1f}% 下位1/3 {d[m & (rk<=1/3)].up.mean()*100:.1f}%"
for name, B in [("RSI2<10 戦術（配当2.45%以上・直近4週に決算なし）", prep(B2)), ("RSI14<30 戦術（決算なし前4週・後2週・業界ぐるみ・配当2.45%以上）", prep(B14))]:
    print("=" * 100); print(name); print("  全体（3か月）", s(B))
    print("\n[1] 銘柄の株価が200日線より上か下か（3か月）")
    print("    上 ", s(B[B.dist > 0])); print("    下 ", s(B[B.dist <= 0]))
    print("    6か月：上", s(B[B.dist > 0], "r6"), "\n           下", s(B[B.dist <= 0], "r6"))
    print("\n[2] 200日線からの距離（3か月）")
    for lab, a, b in [("-20%未満", -999, -20), ("-20〜-10%", -20, -10), ("-10〜0%", -10, 0), ("0〜+10%", 0, 10), ("+10%超", 10, 999)]:
        print(f"    {lab:9s}", s(B[(B.dist > a) & (B.dist <= b)]))
    print("    ", within(B, "dist"), "← 距離（大きいほど200日線より上）")
    print("\n[3] 200日線の向き（20日間の変化）")
    print("    上向き", s(B[B.slope > 0])); print("    下向き", s(B[B.slope <= 0])); print("    ", within(B, "slope"))
    print("    上かつ上向き  ", s(B[(B.dist > 0) & (B.slope > 0)])); print("    下かつ下向き  ", s(B[(B.dist <= 0) & (B.slope <= 0)]))
    print("    下だが上向き  ", s(B[(B.dist <= 0) & (B.slope > 0)])); print("    上だが下向き  ", s(B[(B.dist > 0) & (B.slope <= 0)]))
    print("\n[4] 市場（日本株の均等平均）が200日線より上か下か")
    print("    上", s(B[B.mkt > 0])); print("    下", s(B[B.mkt <= 0]))
    for lab, a, b in [("市場 -10%未満", -999, -10), ("市場 -10〜-5%", -10, -5), ("市場 -5〜0%", -5, 0), ("市場 0〜+5%", 0, 5), ("市場 +5%超", 5, 999)]:
        print(f"      {lab:12s}", s(B[(B.mkt > a) & (B.mkt <= b)]))
    print("    S&P500が200日線より上", s(B[B.spx > 0])); print("    S&P500が200日線より下", s(B[B.spx <= 0]))
    print("\n[5] 市場で200日線より上にある銘柄の割合")
    for lab, a, b in [("30%未満", -1, 30), ("30〜50%", 30, 50), ("50〜70%", 50, 70), ("70%以上", 70, 101)]:
        print(f"    {lab:8s}", s(B[(B.breadth > a) & (B.breadth <= b)]))
    print("\n[6] 銘柄×市場")
    for lab, m in [("銘柄上・市場上", (B.dist > 0) & (B.mkt > 0)), ("銘柄下・市場上", (B.dist <= 0) & (B.mkt > 0)), ("銘柄上・市場下", (B.dist > 0) & (B.mkt <= 0)), ("銘柄下・市場下", (B.dist <= 0) & (B.mkt <= 0))]:
        print(f"    {lab}", s(B[m]))
    print("\n[7] 金融危機型の警戒サイン0〜1個の週に限った場合（2008年の影響を除く確認）")
    a1 = ((spx.shift(20) < -5) | (mkt.shift(20) < -5)); a2 = ew.rolling(252).apply(lambda v: (len(v) - 1 - np.argmax(v)) / 5, raw=True) > 40; a3 = x["HG=F"].pct_change(20) * 100 <= -8
    nw = (a1.astype(int) + a2.astype(int) + a3.astype(int)).reindex(B.date).values; C = B[nw <= 1]
    print("    銘柄 上", s(C[C.dist > 0])); print("    銘柄 下", s(C[C.dist <= 0]))
    print("    市場 上", s(C[C.mkt > 0])); print("    市場 下", s(C[C.mkt <= 0]))
    print("    ", within(C, "dist"))
