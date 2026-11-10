"""CCI20<-100：これまで調べていない要因（銘柄側・相場側）で3か月後の勝率が変わるかを探す"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/-")
from weekpick.universe import load_universe
from weekpick import data as WD
from weekpick.factors import eligible
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
X = pd.read_pickle(SP + "CCI20lt-100_events.pkl")
u = load_universe("topix500"); p = WD.load_panel(list(u.ticker)); idx = p.close.index; mask = eligible(p)
sec = u.set_index("ticker")["sector"].reindex(p.close.columns)
A = p.adjf.ffill(); O, H, L, Cr = p.open, p.high, p.low, p.close; Ca = (Cr * A).ffill(limit=5); V = p.volume
dr = Ca.pct_change(fill_method=None); mkt = dr.where(mask.shift(1, fill_value=False)).mean(axis=1)
G = {}
tp = (H + L + Cr) / 3; tpm = tp.rolling(20).mean(); mad = (tp - tpm).abs().rolling(20).mean()
G["CCIの値"] = (tp - tpm) / (0.015 * mad)
rng = (H - L).replace(0, np.nan)
G["引けの位置(0=安値,1=高値)"] = (Cr - L) / rng
G["下ヒゲの長さ(値幅比)"] = (np.minimum(O, Cr) - L) / rng
G["当日の騰落%"] = dr * 100
down = (dr < 0).astype(int); G["連続下落日数"] = down.apply(lambda c: c.groupby((c == 0).cumsum()).cumsum())
gap = (O / Cr.shift(1) - 1) * 100; G["直近5日の最大の窓開け下落%"] = gap.rolling(5).min()
r5 = Ca / Ca.shift(5) - 1; r20 = Ca / Ca.shift(20) - 1
G["20日の下げのうち直近5日の割合"] = (r5 / r20.where(r20 < -0.02)).clip(-1, 2)
G["下落日の出来高÷上昇日の出来高(20日)"] = (V.where(dr < 0).rolling(20, min_periods=3).mean() / V.where(dr > 0).rolling(20, min_periods=3).mean()).clip(0, 5)
G["日次騰落の歪み(60日)"] = dr.rolling(60, min_periods=40).skew()
G["60日安値を当日更新(1)"] = (Ca <= Ca.rolling(60).min() * 1.0001).astype(float)
G["1年安値からの距離%"] = (Ca / Ca.rolling(250, min_periods=200).min() - 1) * 100
hi = Ca.rolling(250, min_periods=200)
G["1年高値からの営業日数"] = Ca.rolling(250, min_periods=200).apply(lambda v: len(v) - 1 - np.argmax(v), raw=True)
G["3年騰落%(直近1年除く)"] = (Ca.shift(250) / Ca.shift(750) - 1) * 100
G["6か月の勢い%(直近1週除く)"] = (Ca.shift(5) / Ca.shift(126) - 1) * 100
beta = dr.rolling(120, min_periods=80).cov(mkt) / mkt.rolling(120, min_periods=80).var(); G["市場ベータ(120日)"] = beta
resid = dr.sub(beta.mul(mkt, axis=0)); G["固有の値動きの荒さ(60日,年率%)"] = resid.rolling(60, min_periods=40).std() * np.sqrt(250) * 100
G["20日の下げのうち市場で説明できない部分%"] = resid.rolling(20).sum() * 100
G["売買代金の回転(5日/250日)"] = (Cr * V).rolling(5, min_periods=3).mean() / (Cr * V).rolling(250, min_periods=120).mean()
g = p.adjf / p.adjf.shift(1); Dv = (Cr.shift(1) * (1 - 1 / g)).where(g > 1 + 1e-6).fillna(0)
d1 = Dv.rolling("365D").sum(); d0 = d1.shift(250)
G["配当の前年比%"] = ((d1 / d0.where(d0 > 0) - 1) * 100).clip(-100, 200)
exd = (g > 1 + 1e-6)
nx = exd.iloc[::-1].apply(lambda c: (~c).groupby(c.cumsum()).cumsum()).iloc[::-1].where(exd.iloc[::-1].cumsum().iloc[::-1] > 0)
G["次の権利落ちまでの営業日数"] = nx.clip(0, 130)
G["直近20日に権利落ち(1)"] = exd.rolling(20).max().astype(float)
sig = G["CCIの値"] < -100
G["過去1年にシグナルが出た日数"] = sig.rolling(250, min_periods=200).sum()
r20s = r20.where(mask); G["業種内での20日騰落の順位(0=最下位)"] = r20s.T.groupby(sec).rank(pct=True).T
sec1y = (Ca.shift(21) / Ca.shift(273) - 1).where(mask).T.groupby(sec).transform("mean").T; G["業種の1年騰落%"] = sec1y * 100
G["業種の60日騰落%"] = (Ca / Ca.shift(60) - 1).where(mask).T.groupby(sec).transform("mean").T * 100
for k, df in G.items(): X[k] = [df.at[d, t] for d, t in zip(X.date, X.t)]
# 前回このシグナルが出たときの結果（3か月前より前に確定したもの）
X = X.sort_values(["t", "date"]); prev = []
for t, g_ in X.groupby("t", sort=False):
    ds = g_.date.values; ups = g_.up.values
    for i in range(len(g_)):
        j = np.searchsorted(ds, ds[i] - np.timedelta64(100, "D")) - 1
        prev.append(float(ups[max(j - 2, 0):j + 1].mean()) if j >= 0 else np.nan)
X["この銘柄の直近3回のシグナルの勝率"] = prev
E = pd.read_csv(SP + "surprise.csv"); E["d"] = pd.to_datetime(E.jst).dt.normalize(); E = E.dropna(subset=["sur"]).sort_values("d")
Eg = {t: g_ for t, g_ in E.groupby("t")}
def lastsur(t, d):
    g_ = Eg.get(t)
    if g_ is None: return np.nan
    g_ = g_[g_.d <= d - pd.Timedelta(days=3)]
    return g_.sur.iloc[-1] if len(g_) and (d - g_.d.iloc[-1]).days < 200 else np.nan
X["直近のEPSサプライズ%"] = [lastsur(t, d) for t, d in zip(X.t, X.date)]; X["直近のEPSサプライズ%"] = X["直近のEPSサプライズ%"].clip(-100, 100)
# 相場側
x = pd.read_pickle(SP + "crisis_ext.pkl").ffill(); x = x.reindex(x.index.union(idx)).ffill().reindex(idx)
ew = (1 + mkt.fillna(0)).cumprod(); M = pd.DataFrame(index=idx)
M["米国の長短金利差(10年−3か月)"] = x["^TNX"] - x["^IRX"]
M["VIX÷VIX3M(1超は目先の不安が強い)"] = x["^VIX"] / x["^VIX3M"]
M["VIXの20日変化"] = x["^VIX"].diff(20)
M["S&P500の20日騰落%"] = x["^GSPC"].pct_change(20) * 100
M["日本株−S&P500の20日差%"] = ew.pct_change(20) * 100 - M["S&P500の20日騰落%"]
M["日経平均÷均等平均の20日変化%(大型株優位)"] = (x["^N225"] / ew).pct_change(20) * 100
M["米投資適格社債(LQD/IEF)20日%"] = (x["LQD"] / x["IEF"]).pct_change(20) * 100
M["銅の60日騰落%"] = x["HG=F"].pct_change(60) * 100
M["米金融株−S&P500の20日差%"] = (x["XLF"] / x["SPY"]).pct_change(20) * 100
M["銀行株(1615)の20日騰落%"] = x["1615.T"].pct_change(20) * 100
ma200 = Ca.rolling(200, min_periods=150).mean(); M["200日線より上の銘柄の割合%"] = (Ca > ma200).where(mask & ma200.notna()).mean(axis=1) * 100
M["銘柄間の20日騰落のばらつき%"] = r20.where(mask).std(axis=1) * 100
M["市場の20日の値動きの大きさ(年率%)"] = mkt.rolling(20).std() * np.sqrt(250) * 100
M["日本株の1年騰落%"] = ew.pct_change(250) * 100
M["日本株の高値からの下げ%(1年)"] = (ew / ew.rolling(250).max() - 1) * 100
M["米10年金利の水準%"] = x["^TNX"]; M["米10年金利の60日変化"] = x["^TNX"].diff(60); M["ドル円の60日騰落%"] = x["JPY=X"].pct_change(60) * 100
for k in M.columns: X[k] = M[k].reindex(X.date).values
X["月"] = X.date.dt.month; X["月内の日"] = X.date.dt.day
X.attrs["stock"] = list(G) + ["この銘柄の直近3回のシグナルの勝率", "直近のEPSサプライズ%"]; X.attrs["market"] = list(M.columns)
X.to_pickle(SP + "cci_extra_events.pkl"); print("saved", X.shape)
