"""主戦略（25日乖離≤-10%、決算の前後でない事例）で、条件の組み合わせによる高確率の局面を探す（cci_combo.py と同じ方法）"""
import sys, itertools, numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
X = pd.read_pickle(SP + "wf_main_events.pkl")
B = X[(X["直近4週に決算"] == 0) & (X["2週以内に決算"] == 0)].copy()
Cx = pd.read_pickle(SP + "cci_extra_events.pkl"); MK = Cx.attrs["market"]; dm = Cx.groupby("date")[MK].first()
for k in MK: B[k] = dm[k].reindex(B.date).values
x = pd.concat([pd.read_pickle(SP + "crisis_ext.pkl"), pd.read_pickle(SP + "ext2.pkl")], axis=1).ffill(); x = x.loc[:, ~x.columns.duplicated()]
def at(s): return s.reindex(B.date, method="ffill").values
pc = lambda k, n: x[k].pct_change(n) * 100; g = lambda c: pd.to_numeric(B[c], errors="coerce").values
gs200 = (x["^GSPC"] / x["^GSPC"].rolling(200).mean() - 1) * 100
V = at(x["^VIX"]); sig = g("全体のシグナル銘柄割合%"); hot = (g("sec1y") > 60) | (g("r1y") > 100)
C = {
 "円高(60日≤-1.5%)": g("ドル円の60日騰落%") <= -1.46, "円安(60日>+2.3%)": g("ドル円の60日騰落%") > 2.3, "円高(20日≤-1.6%)": g("ドル円20日%") <= -1.62, "直近5日は円安に反転(>+0.5%)": at(pc("JPY=X", 5)) > 0.5,
 "米金利60日低下(≤-0.16)": g("米10年金利の60日変化") <= -0.16, "米金利60日上昇(>+0.17)": g("米10年金利の60日変化") > 0.17, "米金利20日低下(≤-0.13)": g("米10年金利20日変化") <= -0.13,
 "米金利20日は横ばい(-0.13〜+0.06)": (g("米10年金利20日変化") > -0.13) & (g("米10年金利20日変化") <= 0.06), "米長期債が60日で+3%超": at(pc("TLT", 60)) > 3,
 "VIX22超": V > 22, "VIX30超": V > 30, "VIX16未満": V < 16, "VIX22超で5日前より低下": (V > 22) & (at(x["^VIX"].diff(5)) < 0), "VIXが20日高値から15%以上低下": at(x["^VIX"] / x["^VIX"].rolling(20).max()) <= 0.85,
 "VIX÷VIX3M>1": g("VIX÷VIX3M(1超は目先の不安が強い)") > 1,
 "S&P500が20日で-5%以下": at(pc("^GSPC", 20)) <= -5, "S&P500が20日でプラス": at(pc("^GSPC", 20)) > 0, "S&P500が直近5日で+2%超": at(pc("^GSPC", 5)) > 2, "S&P500は200日線より上": at(gs200) > 0,
 "S&P500は高値から-3%以内": at((x["^GSPC"] / x["^GSPC"].rolling(250).max() - 1) * 100) > -3, "S&P500は高値から-10%以下": at((x["^GSPC"] / x["^GSPC"].rolling(250).max() - 1) * 100) <= -10,
 "日本株だけ弱い(対S&P500 20日≤-1.8%)": g("日本株−S&P500の20日差%") <= -1.79, "日本株が20日で-8%以下": g("市場20日騰落%") <= -8, "日本株の20日の下げは-3%より小さい": g("市場20日騰落%") > -3,
 "日本株は200日線より上": g("市場200日乖離%") > 0, "日本株が200日線を5%超下回る": g("市場200日乖離%") < -5, "日本株の1年高値から-10%以下": g("日本株の高値からの下げ%(1年)") <= -10,
 "日本株の1年高値から-5%以内": g("日本株の高値からの下げ%(1年)") > -5, "200日線より上の銘柄が5割未満": g("200日線より上の銘柄の割合%") < 50,
 "全体の15%以上にシグナル(市場全体の急落)": sig >= 15, "全体の30%以上にシグナル": sig >= 30, "シグナルは全体の5%未満(個別・業種の下げ)": sig < 5,
 "SOXが20日で-8%以下": at(pc("^SOX", 20)) <= -8, "金が60日で+8%超": at(pc("GC=F", 60)) > 8, "ドル安(ドル指数60日≤-2%)": at(pc("DX-Y.NYB", 60)) <= -2, "ドル高(ドル指数60日>+3%)": at(pc("DX-Y.NYB", 60)) > 3,
 "原油が60日で-15%以下": at(pc("CL=F", 60)) <= -15, "原油が60日で+15%超": at(pc("CL=F", 60)) > 15, "銅が20日で+3%超": at(pc("HG=F", 20)) > 3, "銅が20日で-5%以下": at(pc("HG=F", 20)) <= -5,
 "上海が60日で-8%以下": at(pc("000001.SS", 60)) <= -8, "新興国株が20日で-5%以下": at(pc("EEM", 20)) <= -5, "米社債が悪化(LQD/IEF 20日≤-0.3%)": g("米投資適格社債(LQD/IEF)20日%") <= -0.26,
 "銀行株が20日で-5%以下": g("銀行株(1615)の20日騰落%") <= -5, "大型株優位(日経÷均等 20日>+0.4%)": g("日経平均÷均等平均の20日変化%(大型株優位)") > 0.4,
 "長短金利差がマイナス(逆イールド)": g("米国の長短金利差(10年−3か月)") < 0, "長短金利差が2%超": g("米国の長短金利差(10年−3か月)") > 2,
 "S&P500が20日前に200日線より上(上昇相場の中の急落)": at(gs200.shift(20)) > 0,
 # 銘柄側（戦略の条件を含む）
 "業界ぐるみ": (g("業種内のシグナル銘柄割合%") >= 30) | (g("業種比20日%") > -5), "業種の6割以上にシグナル": g("業種内のシグナル銘柄割合%") >= 60, "配当2.6%以上": g("配当利回り%") >= 2.6, "配当3.5%以上": g("配当利回り%") >= 3.5,
 "配当4%以上": g("配当利回り%") >= 4, "上がりすぎでない": ~hot, "25日乖離≤-15%": g("25日乖離%") <= -15, "RSI14<30": g("RSI14") < 30, "52週高値から-30%以下": g("52週高値比%") <= -30,
 "200日線より上の銘柄": g("200日乖離%") > 0, "シグナル1週目": g("連続週(何週目)") <= 1, "シグナル3週目以降": g("連続週(何週目)") >= 3, "上位12のシグナルが6個以上": g("シグナル数(上位12)") >= 6,
}
C = {k: np.nan_to_num(np.asarray(v, dtype=float)).astype(bool) for k, v in C.items()}
up = B.up.values.astype(float); r3 = B.r3m.values; wed = B.wedge.values; dates = B.date.values; first = (B.date < "2018").values; yrs = B.date.dt.year.values; bw = B.b_win.values
def stat(m, full=False):
    n = int(m.sum()); wk = np.unique(dates[m]); d = pd.Series(np.sort(wk)); epi = (d.diff().dt.days > 91).cumsum()
    a, b = m & first, m & ~first
    yw = pd.Series(up[m]).groupby(yrs[m]).agg(["mean", "size"]); yw = yw[yw["size"] >= 15]
    s = dict(件数=n, 週=len(wk), 局面=int(epi.max() + 1) if len(d) else 0, 勝率=up[m].mean() * 100, 平均=r3[m].mean(), 大負け=(r3[m] <= -10).mean() * 100, 差=wed[m].mean(),
             前半=up[a].mean() * 100 if a.sum() >= 40 else np.nan, 後半=up[b].mean() * 100 if b.sum() >= 40 else np.nan, 前半件数=int(a.sum()), 後半件数=int(b.sum()),
             年数=len(yw), 勝率6割以上の年=int((yw["mean"] >= 0.6).sum()))
    if full:
        D = pd.DataFrame({"d": dates[m], "u": up[m], "r": r3[m], "b": bw[m]}); D["ep"] = D.d.map(dict(zip(d, epi)))
        E = D.groupby("ep").agg(w=("u", "mean"), r=("r", "mean"), n=("u", "size"), s=("d", "min")); W = D.groupby("d").agg(w=("u", "mean"), b=("b", "first"))
        wo = E.sort_values("r").head(2)
        s.update(週平均=W.w.mean() * 100, 全銘柄=W.b.mean(), 勝ち越し局面=f"{int((E.w > 0.5).sum())}/{len(E)}", 最悪の局面=" / ".join(f"{pd.Timestamp(a_).date()}:{b_:+.0f}%({c_})" for a_, b_, c_ in zip(wo.s, wo.r, wo.n)))
    return s
allm = np.ones(len(B), bool); s0 = stat(allm, True)
print(f"基準（25日乖離≤-10%・決算の前後でない）: {s0['件数']}件 勝率{s0['勝率']:.1f}% 前半{s0['前半']:.1f}% 後半{s0['後半']:.1f}% 週平均{s0['週平均']:.1f}% 全銘柄{s0['全銘柄']:.1f}%  条件数{len(C)}")
names = list(C); rows = []
for k in names: s = stat(C[k], True); s["条件"] = k; s["数"] = 1; rows.append(s)
for a, b in itertools.combinations(names, 2):
    m = C[a] & C[b]
    if m.sum() < 150: continue
    s = stat(m); s["条件"] = f"{a} × {b}"; s["数"] = 2; rows.append(s)
R = pd.DataFrame(rows); n2 = int((R.数 == 2).sum())
good2 = R[(R.数 == 2) & (R.局面 >= 8) & (R.勝率 >= 78)].条件.tolist(); rows3 = []; seen = set()
for nm in good2:
    a, b = nm.split(" × ")
    for c in names:
        key = frozenset((a, b, c))
        if c in (a, b) or key in seen: continue
        seen.add(key); m = C[a] & C[b] & C[c]
        if m.sum() < 150: continue
        s = stat(m); s["条件"] = f"{a} × {b} × {c}"; s["数"] = 3; rows3.append(s)
R = pd.concat([R, pd.DataFrame(rows3)], ignore_index=True); print(f"試した数：単独{len(names)}、2つの組{n2}、3つの組{len(rows3)}")
pd.set_option("display.width", 420); pd.set_option("display.max_colwidth", 95); pd.set_option("display.max_rows", 200)
c1 = ["条件", "件数", "週", "局面", "勝率", "週平均", "全銘柄", "勝ち越し局面", "平均", "大負け", "差", "前半", "後半"]
print("\n=== 単独（勝率順） ==="); print(R[R.数 == 1].sort_values("勝率", ascending=False)[c1].round(1).to_string(index=False))
def robust(D, wmin): return D[(D.局面 >= 10) & (D.前半件数 >= 60) & (D.後半件数 >= 60) & (D.前半 >= wmin) & (D.後半 >= wmin) & (D.勝率6割以上の年 >= D.年数 - 1)]
out = []
for k, wmin, top in [(2, 75, 25), (3, 80, 20)]:
    Q = robust(R[R.数 == k], wmin).assign(弱い方=lambda d: d[["前半", "後半"]].min(axis=1)).sort_values("弱い方", ascending=False)
    print(f"\n=== {k}つの組：前半・後半とも{wmin}%以上、局面10以上（{len(Q)}組） ===")
    for nm in Q.条件.head(top):
        m = np.ones(len(B), bool)
        for kk in nm.split(" × "): m &= C[kk]
        s = stat(m, True); s["条件"] = nm; out.append(s)
    print(pd.DataFrame(out[-min(top, len(Q)):])[c1].round(1).to_string(index=False) if len(Q) else "なし")
P = R[(R.数 == 2) & (R.前半件数 >= 100) & (R.後半件数 >= 100)]
for sel, oth in [("前半", "後半"), ("後半", "前半")]:
    t = P.sort_values(sel, ascending=False).head(30)
    print(f"\n[水増し] {sel}で選んだ上位30組：{sel}{t[sel].mean():.1f}%（基準{s0[sel]:.1f}%）→ {oth}{t[oth].mean():.1f}%（基準{s0[oth]:.1f}%）、{oth}で基準超え {int((t[oth] > s0[oth]).sum())}/30")
print("前半と後半の順位相関:", round(P[["前半", "後半"]].corr(method="spearman").iloc[0, 1], 2))
print("\n=== 危険な組み合わせ（2つの組、勝率の低い順、局面8以上・前半後半とも55%未満） ===")
Q = R[(R.数 == 2) & (R.局面 >= 8) & (R.前半 < 55) & (R.後半 < 55) & (R.前半件数 >= 60) & (R.後半件数 >= 60)].sort_values("勝率").head(15)
print(Q[["条件", "件数", "週", "局面", "勝率", "平均", "大負け", "差", "前半", "後半"]].round(1).to_string(index=False))
t = x.index[-1]; print(f"\n現在（{t.date()}）: 金60日{pc('GC=F',60).iloc[-1]:+.1f}% ドル指数60日{pc('DX-Y.NYB',60).iloc[-1]:+.1f}% 銅20日{pc('HG=F',20).iloc[-1]:+.1f}% 原油60日{pc('CL=F',60).iloc[-1]:+.1f}% VIX{x['^VIX'].iloc[-1]:.1f} S&P500高値比{(x['^GSPC'].iloc[-1]/x['^GSPC'].rolling(250).max().iloc[-1]-1)*100:+.1f}%")
