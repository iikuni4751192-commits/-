"""CCI20<-100 戦術：必須条件の候補の比較と、保有期間・損切り利確・警戒サイン・買い方・RSI2との重なりの検証"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/-")
from weekpick.universe import load_universe
from weekpick import data as WD
from weekpick.factors import eligible
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
COST = 0.001; net = lambda r: (1 + r) * (1 - COST) / (1 + COST) - 1
X = pd.read_pickle(SP + "cci_extra_events.pkl"); base = pd.read_pickle(SP + "wf_base_by_date.pkl")
X["b_win"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[1]); X["wedge"] = X.up * 100 - X.b_win
dv = X["配当利回り%"]; n4 = X["直近4週に決算"] == 0; n2 = X["2週以内に決算"] == 0
def s(d, c="r3m"):
    x_ = d[c].dropna(); a = d[d.date < "2018"][c].dropna(); b = d[d.date >= "2018"][c].dropna()
    return f"{(x_>0).mean()*100:5.1f}%（{len(x_):6d}件） 平均{x_.mean():+5.1f}% -10%以下{(x_<=-10).mean()*100:3.0f}% 前半{(a>0).mean()*100:5.1f}% 後半{(b>0).mean()*100:5.1f}%"
print("[必須条件の候補]（3か月、差＝同じ週の全銘柄との勝率の差）")
for lab, m in [("配当2.45%以上", dv >= 2.45), ("配当3%以上", dv >= 3), ("配当3%以上＋直後2週に決算なし", (dv >= 3) & n2), ("配当3%以上＋前4週・後2週に決算なし", (dv >= 3) & n4 & n2),
               ("配当3%以上で直前4週に決算あり", (dv >= 3) & ~n4 & n2), ("配当3%以上で直後2週に決算あり", (dv >= 3) & ~n2),
               ("配当3.5%以上＋直後2週に決算なし", (dv >= 3.5) & n2), ("配当4%以上＋直後2週に決算なし", (dv >= 4) & n2)]:
    print(f"  {lab:30s}", s(X[m]), f"差{X[m].wedge.mean():+.1f}pt 年{len(X[m])/18.6:.0f}件")
B = X[(dv >= 3) & n2].copy()
u = load_universe("topix500"); p = WD.load_panel(list(u.ticker)); idx = p.close.index; mask = eligible(p)
A = p.adjf.ffill(); Oa = (p.open * A).to_numpy(); Ha = (p.high * A).to_numpy(); La = (p.low * A).to_numpy(); Ca = (p.close * A).ffill(limit=5); Cl = Ca.to_numpy()
pos = {d: i for i, d in enumerate(idx)}; col = {t: j for j, t in enumerate(Ca.columns)}
B["e"] = [pos[d] + 1 for d in B.date]; B = B[B.e + 126 < len(idx)].copy(); B["j"] = [col[t] for t in B.t]; b0 = Oa[B.e, B.j]
for lab, h in [("r1", 21), ("r3", 63), ("r6", 126)]: B[lab] = net(Cl[B.e + h - 1, B.j] / b0 - 1) * 100
B["r3c"] = net(Cl[B.e + 62, B.j] / Cl[B.e, B.j] - 1) * 100
print(f"\n=== 必須条件：配当3%以上＋直後2週に決算なし（{len(B)}件） ===")
print("[保有期間] 1か月", s(B, "r1")); print("           3か月", s(B, "r3")); print("           6か月", s(B, "r6"))
print("[配当] 3〜4%", s(B[B['配当利回り%'] < 4], "r3")); print("       4〜5%", s(B[(B['配当利回り%'] >= 4) & (B['配当利回り%'] < 5)], "r3")); print("       5%以上", s(B[B['配当利回り%'] >= 5], "r3"))
x = pd.read_pickle(SP + "crisis_ext.pkl").ffill(); x = x.reindex(x.index.union(idx)).ffill().reindex(idx)
ew = (1 + Ca.pct_change(fill_method=None).where(mask.shift(1, fill_value=False)).mean(axis=1).fillna(0)).cumprod()
a1 = (((x["^GSPC"] / x["^GSPC"].rolling(200).mean() - 1) * 100).shift(20) < -5) | (((ew / ew.rolling(200).mean() - 1) * 100).shift(20) < -5)
a2 = ew.rolling(252).apply(lambda v: (len(v) - 1 - np.argmax(v)) / 5, raw=True) > 40; a3 = x["HG=F"].pct_change(20) * 100 <= -8
B["nw"] = (a1.astype(int) + a2.astype(int) + a3.astype(int)).reindex(B.date).values
print("[警戒サイン] 0個", s(B[B.nw == 0], "r3")); print("             1個", s(B[B.nw == 1], "r3")); print("             2個以上", s(B[B.nw >= 2], "r3"))
jy, r60, r20 = B["ドル円の60日騰落%"], B["米10年金利の60日変化"], B["米10年金利20日変化"]
print("[相場環境] 円高(60日≤-1.5%)", s(B[jy <= -1.46], "r3")); print("           円安(60日>+2.3%)", s(B[jy > 2.3], "r3"))
print("           米金利60日≤-0.16", s(B[r60 <= -0.16], "r3")); print("           米金利60日>+0.17", s(B[r60 > 0.17], "r3"))
print("           米金利20日≤-0.13", s(B[r20 <= -0.13], "r3")); print("           米金利20日>+0.06", s(B[r20 > 0.06], "r3"))
print("           円安×金利60日上昇", s(B[(jy > 2.3) & (r60 > 0.17)], "r3")); print("           円高×金利60日低下", s(B[(jy <= -1.46) & (r60 <= -0.16)], "r3"))
print("           警戒0〜1個に限り 円安×金利上昇", s(B[(B.nw <= 1) & (jy > 2.3) & (r60 > 0.17)], "r3"))
def sim(tp, sl, Hn=63):
    out = []
    for e, j in zip(B.e.values, B.j.values):
        bb = Oa[e, j]; r = None
        for k in range(Hn):
            o, h, l = Oa[e + k, j] / bb - 1, Ha[e + k, j] / bb - 1, La[e + k, j] / bb - 1
            if k > 0 and np.isfinite(o) and (o <= -sl or o >= tp): r = o; break
            if np.isfinite(l) and l <= -sl: r = -sl; break
            if np.isfinite(h) and h >= tp: r = tp; break
        out.append(net(r if r is not None else Cl[e + Hn - 1, j] / bb - 1) * 100)
    return np.array(out)
print("[損切り・利確（最長63営業日）]")
for tp, sl in [(0.02, 0.015), (0.10, 0.05), (np.inf, 0.10), (np.inf, np.inf)]:
    r = sim(tp, sl); print(f"  利確{tp*100 if np.isfinite(tp) else 'なし'}/損切{sl*100 if np.isfinite(sl) else 'なし'}: 勝率{(r>0).mean()*100:.1f}% 平均{r.mean():+.2f}% 悪い方から5% {np.percentile(r,5):+.1f}%")
print("[買い方] 寄り成行 平均%.2f%% ／ 月曜の引け 平均%.2f%%（勝率%.1f%%）" % (B.r3.mean(), B.r3c.mean(), (B.r3c > 0).mean() * 100))
ref = Cl[B.e - 1, B.j]
for lim in [0.0, -0.02]:
    lv = ref * (1 + lim); fill = np.full(len(B), np.nan)
    for i, (e, j) in enumerate(zip(B.e.values, B.j.values)):
        for k in range(5):
            o, l = Oa[e + k, j], La[e + k, j]
            if np.isfinite(o) and o <= lv[i]: fill[i] = o; break
            if np.isfinite(l) and l <= lv[i]: fill[i] = lv[i]; break
    r = net(Cl[B.e + 62, B.j] / fill - 1) * 100; f = np.isfinite(r); print(f"  前週末終値{lim*100:+.0f}%の指値: 約定率{f.mean()*100:.0f}% 機会あたり平均{np.nan_to_num(r).mean():+.2f}%")
R2 = pd.read_pickle(SP + "rsi2_extra_events.pkl")[["date", "t"]].assign(r2=1); R14 = pd.read_pickle(SP + "RSI14lt30_events.pkl")[["date", "t"]].assign(r14=1)
Bm = B.merge(R2, on=["date", "t"], how="left").merge(R14, on=["date", "t"], how="left"); Bm["dev"] = Bm["25日乖離%"] <= -10
print(f"[ほかのシグナルとの重なり] RSI2<10も同時 {Bm.r2.notna().mean()*100:.0f}% ／ RSI14<30も同時 {Bm.r14.notna().mean()*100:.0f}% ／ 25日乖離≤-10%も同時 {Bm.dev.mean()*100:.0f}%")
only = Bm[Bm.r2.isna() & Bm.r14.isna() & ~Bm.dev]; print("  CCIだけ（ほかの3つは出ていない）", s(only, "r3"), f"差{only.wedge.mean():+.1f}pt"); both = Bm[Bm.r2.notna()]; print("  RSI2<10も同時              ", s(both, "r3"), f"差{both.wedge.mean():+.1f}pt")
yr = B.groupby(B.date.dt.year).r3.agg(lambda v: (v > 0).mean() * 100).round(0); print("[年別]", yr.to_dict(), " 年の件数", B.groupby(B.date.dt.year).size().min(), "〜", B.groupby(B.date.dt.year).size().max())
sc = B.groupby("sector").r3.agg(["size", lambda v: (v > 0).mean() * 100]); sc.columns = ["n", "w"]; sc = sc[sc.n >= 150].sort_values("w"); print("[業種] 低", sc.head(3).round(1).to_dict("index"), "高", sc.tail(3).round(1).to_dict("index"))
print("悪い方から5pct 3か月", round(np.percentile(B.r3.dropna(), 5), 1))
