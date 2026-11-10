"""CCI20<-100 戦術（配当3%以上・直後2週に決算なし）の事例で、条件の組み合わせによる高確率の局面を探す
・条件はあらかじめ決めた固定の閾値（約60個）。2つ・3つの組み合わせを総当たり
・前半（〜2017）で選んだ上位が後半でも通用するか（とその逆）で、探索による水増しを測る"""
import sys, itertools, numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
X = pd.read_pickle(SP + "cci_extra_events.pkl"); base = pd.read_pickle(SP + "wf_base_by_date.pkl")
X["b_win"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[1]); X["wedge"] = X.up * 100 - X.b_win
B = X[(X["配当利回り%"] >= 3) & (X["2週以内に決算"] == 0)].dropna(subset=["b_win"]).copy()
x = pd.concat([pd.read_pickle(SP + "crisis_ext.pkl"), pd.read_pickle(SP + "ext2.pkl")], axis=1).ffill(); x = x.loc[:, ~x.columns.duplicated()]
def at(s): return s.reindex(B.date, method="ffill").values
pc = lambda k, n: x[k].pct_change(n) * 100
M = {}
M["SOX20"] = at(pc("^SOX", 20)); M["SOX60"] = at(pc("^SOX", 60)); M["金60"] = at(pc("GC=F", 60)); M["ドル指数60"] = at(pc("DX-Y.NYB", 60)); M["原油60"] = at(pc("CL=F", 60)); M["原油20"] = at(pc("CL=F", 20))
M["上海60"] = at(pc("000001.SS", 60)); M["新興国株20"] = at(pc("EEM", 20)); M["米小型−大型20"] = at((x["^RUT"] / x["^GSPC"]).pct_change(20) * 100)
M["米ハイテク−公益20"] = at((x["XLK"] / x["XLU"]).pct_change(20) * 100); M["米長期債60"] = at(pc("TLT", 60)); M["円建て日本株ETF−現地20"] = at((x["EWJ"] / x["^N225"]).pct_change(20) * 100)
M["VIX"] = at(x["^VIX"]); M["VIX5日変化"] = at(x["^VIX"].diff(5)); M["VIX20日高値比"] = at(x["^VIX"] / x["^VIX"].rolling(20).max()); M["VIX1年分位"] = at(x["^VIX"].rolling(250).rank(pct=True))
M["SP20"] = at(pc("^GSPC", 20)); M["SP5"] = at(pc("^GSPC", 5)); M["SP200乖離"] = at((x["^GSPC"] / x["^GSPC"].rolling(200).mean() - 1) * 100); M["SP高値比"] = at((x["^GSPC"] / x["^GSPC"].rolling(250).max() - 1) * 100)
M["銅20"] = at(pc("HG=F", 20)); M["米5年金利60"] = at(x["^FVX"].diff(60)); M["金利20"] = B["米10年金利20日変化"].values; M["金利60"] = B["米10年金利の60日変化"].values
M["円20"] = B["ドル円20日%"].values; M["円60"] = B["ドル円の60日騰落%"].values; M["円5"] = at(pc("JPY=X", 5))
g = lambda c: B[c].values
C = {  # 相場側
 "円高(60日≤-1.5%)": M["円60"] <= -1.46, "円安(60日>+2.3%)": M["円60"] > 2.3, "円高(20日≤-1.6%)": M["円20"] <= -1.62, "直近5日は円安に反転(>+0.5%)": M["円5"] > 0.5,
 "米金利60日低下(≤-0.16)": M["金利60"] <= -0.16, "米金利60日上昇(>+0.17)": M["金利60"] > 0.17, "米金利20日低下(≤-0.13)": M["金利20"] <= -0.13, "米5年金利60日低下(≤-0.2)": M["米5年金利60"] <= -0.2,
 "米長期債が60日で+3%超": M["米長期債60"] > 3,
 "VIX22超": M["VIX"] > 22, "VIX30超": M["VIX"] > 30, "VIX16未満": M["VIX"] < 16, "VIXが1年で上位2割": M["VIX1年分位"] >= 0.8, "VIXが20日高値から15%以上低下": M["VIX20日高値比"] <= 0.85,
 "VIX22超で5日前より低下": (M["VIX"] > 22) & (M["VIX5日変化"] < 0), "VIX÷VIX3M>1": g("VIX÷VIX3M(1超は目先の不安が強い)") > 1,
 "S&P500が20日で-5%以下": M["SP20"] <= -5, "S&P500が20日でプラス": M["SP20"] > 0, "S&P500が直近5日で+2%超": M["SP5"] > 2, "S&P500は200日線より上": M["SP200乖離"] > 0,
 "S&P500は高値から-3%以内": M["SP高値比"] > -3, "S&P500は高値から-10%以下": M["SP高値比"] <= -10,
 "日本株だけ弱い(対S&P500 20日≤-1.8%)": g("日本株−S&P500の20日差%") <= -1.79, "日本株が20日で-3.7%以下": g("市場20日騰落%") <= -3.69, "日本株が20日で-8%以下": g("市場20日騰落%") <= -8,
 "日本株は200日線より上": g("市場200日乖離%") > 0, "日本株が200日線を5%超下回る": g("市場200日乖離%") < -5, "日本株の1年高値から-10%以下": g("日本株の高値からの下げ%(1年)") <= -10,
 "日本株の1年高値から-3%以内": g("日本株の高値からの下げ%(1年)") > -3, "200日線より上の銘柄が5割未満": g("200日線より上の銘柄の割合%") < 50,
 "全体の43%以上にシグナル": g("全体のシグナル銘柄割合%") >= 43.3, "全体の60%以上にシグナル": g("全体のシグナル銘柄割合%") >= 60, "シグナルは全体の22%未満(個別の下げ)": g("全体のシグナル銘柄割合%") < 22.3,
 "SOXが20日で-8%以下": M["SOX20"] <= -8, "SOXが60日で+10%超": M["SOX60"] > 10, "金が60日で+8%超": M["金60"] > 8, "ドル指数が60日で-2%以下": M["ドル指数60"] <= -2, "ドル指数が60日で+3%超": M["ドル指数60"] > 3,
 "原油が60日で-15%以下": M["原油60"] <= -15, "原油が60日で+15%超": M["原油60"] > 15, "銅が20日で+3%超": M["銅20"] > 3, "銅が20日で-5%以下": M["銅20"] <= -5,
 "上海が60日で-8%以下": M["上海60"] <= -8, "新興国株が20日で-5%以下": M["新興国株20"] <= -5, "米小型株が大型株に20日で+2%超": M["米小型−大型20"] > 2,
 "米ハイテクが公益株に20日で-5%以下": M["米ハイテク−公益20"] <= -5, "米社債が悪化(LQD/IEF 20日≤-0.3%)": g("米投資適格社債(LQD/IEF)20日%") <= -0.26,
 "銀行株が20日で-5%以下": g("銀行株(1615)の20日騰落%") <= -5, "銀行株が20日で+4%超": g("銀行株(1615)の20日騰落%") > 4.2, "大型株優位(日経÷均等 20日>+0.4%)": g("日経平均÷均等平均の20日変化%(大型株優位)") > 0.4,
 "長短金利差がマイナス(逆イールド)": g("米国の長短金利差(10年−3か月)") < 0, "長短金利差が2%超": g("米国の長短金利差(10年−3か月)") > 2,
 # 銘柄側
 "配当4%以上": g("配当利回り%") >= 4, "配当5%以上": g("配当利回り%") >= 5, "25日乖離≤-10%": g("25日乖離%") <= -10, "RSI14<30": g("RSI14") < 30, "直前4週に決算なし": g("直近4週に決算") == 0,
 "業種の6割以上にシグナル": g("業種内のシグナル銘柄割合%") >= 60, "200日線より上の銘柄": g("200日乖離%") > 0, "52週高値から-30%以下": g("52週高値比%") <= -30, "直近20日に権利落ち": g("直近20日に権利落ち(1)") == 1,
 "1年で業種より20%超上昇": g("業種比1年%(直近1か月除く)") > 20,
}
C = {k: np.nan_to_num(np.asarray(v, dtype=float)).astype(bool) for k, v in C.items()}
STOCK = {"配当4%以上", "配当5%以上", "25日乖離≤-10%", "RSI14<30", "直前4週に決算なし", "業種の6割以上にシグナル", "200日線より上の銘柄", "52週高値から-30%以下", "直近20日に権利落ち", "1年で業種より20%超上昇"}
up = B.up.values.astype(float); r3 = B.r3m.values; wed = B.wedge.values; dates = B.date.values; first = (B.date < "2018").values; yrs = B.date.dt.year.values
wkid = pd.factorize(B.date)[0]; udates = pd.Series(sorted(B.date.unique()))
def stat(m):
    n = int(m.sum())
    if n == 0: return None
    wk = np.unique(dates[m]); d = pd.Series(np.sort(wk)); ep = int((d.diff().dt.days > 91).sum() + 1)
    a, b = m & first, m & ~first
    yw = pd.Series(up[m]).groupby(yrs[m]).agg(["mean", "size"]); yw = yw[yw["size"] >= 20]
    return dict(件数=n, 週=len(wk), 局面=ep, 勝率=up[m].mean() * 100, 平均=r3[m].mean(), 大負け=(r3[m] <= -10).mean() * 100, 差=wed[m].mean(),
                前半=up[a].mean() * 100 if a.sum() >= 50 else np.nan, 後半=up[b].mean() * 100 if b.sum() >= 50 else np.nan, 前半件数=int(a.sum()), 後半件数=int(b.sum()),
                年数=len(yw), 勝率6割以上の年=int((yw["mean"] >= 0.6).sum()), 最悪の年=yw["mean"].min() * 100 if len(yw) else np.nan)
allm = np.ones(len(B), bool); s0 = stat(allm)
print(f"基準（戦術の必須条件のみ）: {s0['件数']}件 勝率{s0['勝率']:.1f}% 平均{s0['平均']:+.1f}% 前半{s0['前半']:.1f}% 後半{s0['後半']:.1f}%  条件数 {len(C)}")
names = list(C); rows = []
for k in names:
    s = stat(C[k]); s["条件"] = k; s["数"] = 1; rows.append(s)
for a, b in itertools.combinations(names, 2):
    m = C[a] & C[b]
    if m.sum() < 200: continue
    s = stat(m); s["条件"] = f"{a} × {b}"; s["数"] = 2; rows.append(s)
R = pd.DataFrame(rows); n_pairs = (R.数 == 2).sum()
# 3つ：2つの組で局面8以上・勝率72%以上のものに1条件を追加
good2 = R[(R.数 == 2) & (R.局面 >= 8) & (R.勝率 >= 72)].条件.tolist(); rows3 = []; seen = set()
for nm in good2:
    a, b = nm.split(" × ")
    for c in names:
        if c in (a, b): continue
        key = frozenset((a, b, c))
        if key in seen: continue
        seen.add(key); m = C[a] & C[b] & C[c]
        if m.sum() < 200: continue
        s = stat(m); s["条件"] = f"{a} × {b} × {c}"; s["数"] = 3; rows3.append(s)
R = pd.concat([R, pd.DataFrame(rows3)], ignore_index=True)
R.to_pickle(SP + "cci_combo.pkl"); print(f"試した数：単独{len(names)}、2つの組{n_pairs}、3つの組{len(rows3)}")
pd.set_option("display.width", 400); pd.set_option("display.max_colwidth", 90); pd.set_option("display.max_rows", 200)
cols = ["条件", "件数", "週", "局面", "勝率", "平均", "大負け", "差", "前半", "後半", "年数", "勝率6割以上の年", "最悪の年"]
print("\n=== 単独の条件（勝率の高い順） ==="); print(R[R.数 == 1].sort_values("勝率", ascending=False)[cols].round(1).to_string(index=False))
def robust(D, wmin): return D[(D.局面 >= 10) & (D.前半件数 >= 100) & (D.後半件数 >= 100) & (D.前半 >= wmin) & (D.後半 >= wmin) & (D.年数 >= 6) & (D.勝率6割以上の年 >= D.年数 - 1)]
for k, wmin in [(2, 72), (3, 76)]:
    Q = robust(R[R.数 == k], wmin).assign(弱い方=lambda d: d[["前半", "後半"]].min(axis=1)).sort_values("弱い方", ascending=False)
    print(f"\n=== {k}つの組：前半・後半とも勝率{wmin}%以上、局面10以上、勝率6割未満の年は1年まで（{len(Q)}組。前半・後半の低い方の順） ==="); print(Q[cols].head(30).round(1).to_string(index=False))
# 探索による水増しの測定：前半で選んだ上位が後半でどうだったか（とその逆）
P = R[(R.数 == 2) & (R.前半件数 >= 150) & (R.後半件数 >= 150)]
for sel, oth in [("前半", "後半"), ("後半", "前半")]:
    top = P.sort_values(sel, ascending=False).head(30)
    print(f"\n[水増しの測定] {sel}の勝率で選んだ上位30組：{sel} 平均{top[sel].mean():.1f}%（基準{s0[sel]:.1f}%）→ {oth} 平均{top[oth].mean():.1f}%（基準{s0[oth]:.1f}%）、{oth}で基準を上回った組 {int((top[oth] > s0[oth]).sum())}/30")
print("順位相関（2つの組の前半の勝率と後半の勝率）:", round(P[["前半", "後半"]].corr(method="spearman").iloc[0, 1], 2))
