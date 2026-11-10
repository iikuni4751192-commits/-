"""cci_extra の補足：市場ベータ等の3要因（計算のやり直し）と、ドル円60日・米金利60日の頑健性"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/-")
from weekpick.universe import load_universe
from weekpick import data as WD
from weekpick.factors import eligible
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
X = pd.read_pickle(SP + "cci_extra_events.pkl")
u = load_universe("topix500"); p = WD.load_panel(list(u.ticker)); idx = p.close.index; mask = eligible(p)
Ca = (p.close * p.adjf.ffill()).ffill(limit=5); dr = Ca.pct_change(fill_method=None); mkt = dr.where(mask.shift(1, fill_value=False)).mean(axis=1)
mm = pd.DataFrame(np.repeat(mkt.values[:, None], dr.shape[1], 1), index=idx, columns=dr.columns)
ro = lambda z: z.rolling(120, min_periods=80).mean()
beta = (ro(dr * mm) - ro(dr) * ro(mm)) / (ro(mm * mm) - ro(mm) ** 2); resid = dr - beta * mm
G = {"市場ベータ(120日)": beta, "固有の値動きの荒さ(60日,年率%)": resid.rolling(60, min_periods=40).std() * np.sqrt(250) * 100, "20日の下げのうち市場で説明できない部分%": resid.rolling(20).sum() * 100}
def blk_t(s):
    s = s.dropna(); b = s.groupby(((s.index - pd.Timestamp("2000-01-07")).days // 91)).mean(); return b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))
for k, df in G.items():
    X[k] = [df.at[d, t] for d, t in zip(X.date, X.t)]; d = X.dropna(subset=[k]); vals = {}
    for w, g_ in d.groupby("date"):
        if g_.up.any() and (~g_.up).any(): vals[w] = g_.loc[g_.up, k].mean() - g_.loc[~g_.up, k].mean()
    wk = pd.Series(vals); wk.index = pd.to_datetime(wk.index); rk = d.groupby("date")[k].rank(pct=True)
    print(f"{k}: t {blk_t(wk):+.2f}（前半{blk_t(wk.loc[:'2017']):+.2f}／後半{blk_t(wk.loc['2018':]):+.2f}） 同じ週の上位1/3 {d[rk>2/3].up.mean()*100:.1f}% 下位1/3 {d[rk<=1/3].up.mean()*100:.1f}%")
base = pd.read_pickle(SP + "wf_base_by_date.pkl"); X["b_win"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[1]); X["wedge"] = X.up * 100 - X.b_win
def ep(d):
    s = pd.Series(sorted(d.date.unique())); return int((s.diff().dt.days > 91).sum() + 1)
def st(d): return f"{d.up.mean()*100:5.1f}%（{len(d):6d}件・{d.date.nunique():3d}週・{ep(d):2d}局面） -10%以下{(d.r3m<=-10).mean()*100:3.0f}% 差{d.wedge.mean():+5.1f}pt｜前半{d[d.date<'2018'].up.mean()*100:5.1f}% 後半{d[d.date>='2018'].up.mean()*100:5.1f}% 2008年を除く{d[d.date.dt.year!=2008].up.mean()*100:5.1f}%"
jy, r60 = X["ドル円の60日騰落%"], X["米10年金利の60日変化"]
print("\n【ドル円の60日騰落】")
for lab, m in [("円高 −1.5%以下", jy <= -1.46), ("横ばい", (jy > -1.46) & (jy <= 2.3)), ("円安 +2.3%超", jy > 2.3), ("円安 +5%超", jy > 5)]: print(f"  {lab:12s}", st(X[m]))
print("【米10年金利の60日変化】")
for lab, m in [("低下 −0.16以下", r60 <= -0.16), ("横ばい", (r60 > -0.16) & (r60 <= 0.17)), ("上昇 +0.17超", r60 > 0.17)]: print(f"  {lab:12s}", st(X[m]))
print("【組み合わせ】")
for lab, m in [("円高×金利低下", (jy <= -1.46) & (r60 <= -0.16)), ("円高×金利は低下でない", (jy <= -1.46) & (r60 > -0.16)), ("円高でない×金利低下", (jy > -1.46) & (r60 <= -0.16)),
               ("円安×金利上昇", (jy > 2.3) & (r60 > 0.17)), ("配当4%以上×決算なし×円高", (X["配当利回り%"] >= 4) & (X["直近4週に決算"] == 0) & (X["2週以内に決算"] == 0) & (jy <= -1.46)),
               ("配当4%以上×決算なし×円安", (X["配当利回り%"] >= 4) & (X["直近4週に決算"] == 0) & (X["2週以内に決算"] == 0) & (jy > 2.3))]: print(f"  {lab:20s}", st(X[m]))
print("【年ごと：円高の週と円安の週の勝率】")
T = pd.DataFrame({"円高": X[jy <= -1.46].groupby(X.date.dt.year).up.mean() * 100, "円安": X[jy > 2.3].groupby(X.date.dt.year).up.mean() * 100, "円高件数": X[jy <= -1.46].groupby(X.date.dt.year).size(), "円安件数": X[jy > 2.3].groupby(X.date.dt.year).size()})
print(T.round(0).T.to_string()); print("円高>円安だった年:", int((T.円高 > T.円安).sum()), "/", int(T.dropna().shape[0]))
x = pd.read_pickle(SP + "crisis_ext.pkl").ffill(); print("現在（", x.index[-1].date(), "）ドル円60日 %.1f%%  米金利60日 %+.2f" % (x["JPY=X"].pct_change(60).iloc[-1] * 100, x["^TNX"].ffill().diff(60).iloc[-1]))
