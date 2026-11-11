"""未検証の新しい要因（銘柄側・相場側・カレンダー）を計算し、主戦略とCCI戦術の事例に付ける"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/-")
from weekpick.universe import load_universe
from weekpick import data as WD
from weekpick.factors import eligible
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
u = load_universe("topix500"); p = WD.load_panel(list(u.ticker)); idx = p.close.index; mask = eligible(p)
A = p.adjf.ffill(); O, H, L, Cr, V = p.open, p.high, p.low, p.close, p.volume; Ca = (Cr * A).ffill(limit=5)
dr = Ca.pct_change(fill_method=None); mkt = dr.where(mask.shift(1, fill_value=False)).mean(axis=1).fillna(0); ew = (1 + mkt).cumprod()
cols = list(Ca.columns); sec = u.set_index("ticker")["sector"].reindex(cols)
G = {}
g = p.adjf / p.adjf.shift(1); Dv = (Cr.shift(1) * (1 - 1 / g)).where(g > 1 + 1e-6).fillna(0); YLD = Dv.rolling("365D").sum() / Cr * 100
G["配当利回りの対3年中央値比"] = (YLD / YLD.rolling(750, min_periods=500).median()).clip(0, 5)
secy = YLD.where(mask).T.groupby(sec).transform("median").T; G["配当利回りの業種中央値比"] = (YLD / secy).clip(0, 6)
hi20 = Ca.rolling(20).max(); ds = pd.DataFrame(np.nan, index=idx, columns=cols)
for k in range(19, -1, -1): ds = ds.mask(Ca.shift(k) >= hi20 * 0.99999, k)
G["20日高値から何日前か"] = ds
G["20日高値からの下落%÷経過日数"] = ((Ca / hi20 - 1) * 100 / (ds + 1))
G["20日中の下落日数"] = (dr < 0).rolling(20).sum()
G["出来高の枯れ(5日/60日)"] = V.rolling(5).mean() / V.rolling(60).mean()
G["値動きの大きさ(20日÷250日)"] = dr.rolling(20).std() / dr.rolling(250, min_periods=150).std()
rng5 = H.rolling(5).max() - L.rolling(5).min(); G["5日レンジ内の位置(0=安値,1=高値)"] = (Cr - L.rolling(5).min()) / rng5.replace(0, np.nan)
G["5日安値からの反発%"] = (Cr / L.rolling(5).min() - 1) * 100
G["直近5日で終値が安値圏の日数"] = (((Cr - L) / (H - L).replace(0, np.nan)) < 0.2).rolling(5).sum()
G["下落日の対市場の強さ(250日平均%)"] = (dr.sub(mkt, axis=0)).where(mkt < 0, np.nan).rolling(250, min_periods=60).mean() * 100
mm = pd.DataFrame(np.repeat(mkt.values[:, None], len(cols), 1), index=idx, columns=cols); ro = lambda z, n=60: z.rolling(n, min_periods=int(n * .6)).mean()
cov = ro(dr * mm) - ro(dr) * ro(mm); G["市場との相関(60日)"] = cov / (np.sqrt((ro(dr * dr) - ro(dr) ** 2).clip(lower=1e-12)) * np.sqrt((ro(mm * mm) - ro(mm) ** 2).clip(lower=1e-12)))
G["1日で最大の上昇%(20日)"] = dr.rolling(20).max() * 100
G["半年の最高値からの日数"] = Ca.rolling(126).apply(lambda v: len(v) - 1 - np.argmax(v), raw=True) if False else np.nan
exd = (g > 1 + 1e-6).to_numpy(); nx = np.full(exd.shape, np.nan)
for j in range(exd.shape[1]):
    pos = np.flatnonzero(exd[:, j])
    if len(pos): k = np.searchsorted(pos, np.arange(exd.shape[0]), side="right"); ok = k < len(pos); nx[ok, j] = pos[k[ok]] - np.arange(exd.shape[0])[ok]
G["次の権利落ちまでの営業日"] = pd.DataFrame(nx, index=idx, columns=cols); G["63営業日内に権利落ちあり"] = (G["次の権利落ちまでの営業日"] <= 63).astype(float).where(G["次の権利落ちまでの営業日"].notna())
Ed = pd.read_csv("/home/user/-/reports/work/earnings_dates.csv"); Ed["d"] = pd.to_datetime(Ed.jst).dt.normalize(); Eg = {t: np.sort(g_.d.unique()) for t, g_ in Ed.groupby("ticker")}
# 市場（日次）
x = pd.concat([pd.read_pickle(SP + "crisis_ext.pkl"), pd.read_pickle(SP + "ext2.pkl"), pd.read_pickle(SP + "ext3.pkl")], axis=1).ffill(); x = x.loc[:, ~x.columns.duplicated()]
x = x.reindex(x.index.union(idx)).ffill().reindex(idx)
dev25 = (Ca / Ca.rolling(25).mean() - 1) * 100; sh = ((dev25 <= -10) & mask).sum(axis=1) / mask.sum(axis=1) * 100
M = pd.DataFrame(index=idx)
M["市場全体の急落割合の前週比"] = sh.diff(5); M["急落割合÷20日最大(1=ピーク中)"] = sh / sh.rolling(20).max().replace(0, np.nan)
M["急落割合15%以上が続いた日数"] = (sh >= 15).astype(int).groupby((sh < 15).cumsum()).cumsum()
M["日本株の5日騰落%"] = ew.pct_change(5) * 100; M["日本株の1日騰落%"] = ew.pct_change() * 100; M["日本株の20日安値からの反発%"] = (ew / ew.rolling(20).min() - 1) * 100
M["日本株の60日高値からの下げ%"] = (ew / ew.rolling(60).max() - 1) * 100; M["日本株の下落日数(20日)"] = (mkt < 0).rolling(20).sum()
pcx = lambda k, n: x[k].pct_change(n) * 100
for k, n in [("^KS11", 20), ("^TWII", 20), ("^GDAXI", 20), ("^FTSE", 20), ("^STOXX50E", 20), ("^IXIC", 20), ("BTC-USD", 20), ("SI=F", 60), ("NG=F", 60), ("AUDJPY=X", 20), ("AUDJPY=X", 60), ("EURJPY=X", 20), ("CNY=X", 60)]:
    M[f"{k}の{n}日騰落%"] = pcx(k, n)
M["韓国株−日本株の20日差%"] = M["^KS11の20日騰落%"] - ew.pct_change(20) * 100; M["台湾株−日本株の20日差%"] = M["^TWIIの20日騰落%"] - ew.pct_change(20) * 100
M["欧州株(DAX)−日本株の20日差%"] = M["^GDAXIの20日騰落%"] - ew.pct_change(20) * 100
M["豪ドル円−ドル円の20日差%(リスク選好)"] = M["AUDJPY=Xの20日騰落%"] - pcx("JPY=X", 20)
for k in ["^SKEW", "^MOVE", "^OVX", "^GVZ", "^VXN"]: M[f"{k}の水準"] = x[k]; M[f"{k}の1年分位"] = x[k].rolling(250, min_periods=150).rank(pct=True)
M["^MOVEの20日変化"] = x["^MOVE"].diff(20); M["VXN÷VIX"] = x["^VXN"] / x["^VIX"]
M["銅÷金の60日変化%"] = ((x["HG=F"] / x["GC=F"]).pct_change(60)) * 100; M["銀÷金の60日変化%"] = ((x["SI=F"] / x["GC=F"]).pct_change(60)) * 100; M["原油÷金の60日変化%"] = ((x["CL=F"] / x["GC=F"]).pct_change(60)) * 100
M["米30年−10年金利差"] = x["^TYX"] - x["^TNX"]; M["米30年−10年金利差の20日変化"] = M["米30年−10年金利差"].diff(20); M["米3か月金利の60日変化"] = x["^IRX"].diff(60)
M["米ハイイールド(HYG/IEF)60日%"] = (x["HYG"] / x["IEF"]).pct_change(60) * 100
cnt = Ed.groupby("d").size().reindex(pd.date_range("2006-01-01", "2027-03-31"), fill_value=0); M["決算発表の集中度(直近14日の発表銘柄の割合%)"] = (cnt.rolling(14).sum() / len(cols) * 100).reindex(idx).values
def attach(X):
    X = X.copy(); r = idx.get_indexer(pd.to_datetime(X.date)); c = pd.Index(cols).get_indexer(X.t)
    for k, df in G.items():
        if isinstance(df, pd.DataFrame): X[k] = df.to_numpy()[r, c]
    for k in M.columns: X[k] = M[k].to_numpy()[r]
    nd = []
    for t, d in zip(X.t, pd.to_datetime(X.date)):
        a = Eg.get(t)
        if a is None: nd.append(np.nan); continue
        k = np.searchsorted(a, np.datetime64(d), side="right"); nd.append((a[k] - np.datetime64(d)).astype("timedelta64[D]").astype(int) if k < len(a) else np.nan)
    X["次の決算までの日数"] = nd; d = pd.to_datetime(X.date)
    X["月"] = d.dt.month; X["月内の週(0〜4)"] = (d.dt.day - 1) // 7; X["SQ週(第2金曜を含む週)"] = ((d.dt.day >= 8) & (d.dt.day <= 14)).astype(float)
    X["3月・9月の最終週(権利付き最終日前後)"] = (d.dt.month.isin([3, 9]) & (d.dt.day >= 24)).astype(float); X["12月の後半"] = ((d.dt.month == 12) & (d.dt.day >= 15)).astype(float)
    X["お盆週(8/10〜16)"] = ((d.dt.month == 8) & d.dt.day.between(10, 16)).astype(float); X["GW前後(4/25〜5/8)"] = (((d.dt.month == 4) & (d.dt.day >= 25)) | ((d.dt.month == 5) & (d.dt.day <= 8))).astype(float)
    X["年の何週目"] = d.dt.isocalendar().week.astype(float).values
    return X
base = pd.read_pickle(SP + "wf_base_by_date.pkl")
for name, fn in [("cci", "cci_extra_events.pkl"), ("main", "wf_main_events.pkl")]:
    X = pd.read_pickle(SP + fn)
    X["b_win"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[1]); X["wedge"] = X.up * 100 - X.b_win
    X = attach(X); X.attrs["stock"] = [k for k, v in G.items() if isinstance(v, pd.DataFrame)] + ["次の決算までの日数"]; X.attrs["market"] = list(M.columns)
    X.attrs["cal"] = ["月内の週(0〜4)", "SQ週(第2金曜を含む週)", "3月・9月の最終週(権利付き最終日前後)", "12月の後半", "お盆週(8/10〜16)", "GW前後(4/25〜5/8)", "年の何週目"]
    X.to_pickle(SP + f"free_{name}.pkl"); print(name, X.shape, len(X.attrs["stock"]), len(X.attrs["market"]), len(X.attrs["cal"]))
