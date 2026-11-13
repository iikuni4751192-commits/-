"""ネットで言われる上昇シグナルの、1か月(21営業日)後の上昇確率。毎週末の終値でシグナル判定→翌営業日の始値で買い→21営業日目の終値、コスト片道0.1%"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user")
sys.path.insert(0, "/home/user/-")
from weekpick.universe import load_universe
from weekpick import data as WD
from weekpick.factors import eligible
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
COST = 0.001; HOLD = 21
u = load_universe("topix500"); p = WD.load_panel(list(u.ticker)); idx = p.close.index; mask = eligible(p)
A = p.adjf.ffill(); Ca = (p.close * A).ffill(limit=5); O = p.open * A; V = p.volume; cols = list(Ca.columns)
dr = Ca.pct_change(fill_method=None)
wk = pd.Series(idx, index=idx).groupby([idx.isocalendar().year.values, idx.isocalendar().week.values]).max().values
wi = np.array([idx.get_loc(d) for d in wk if idx.get_loc(d) + 1 + HOLD < len(idx)])
S = {}
def rsi(n):
    d = Ca.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean(); return 100 * up / (up + dn).replace(0, np.nan)
ma = lambda n: Ca.rolling(n).mean()
dev25 = (Ca / ma(25) - 1) * 100; r2 = rsi(2); r14 = rsi(14)
hi, lo = p.high * A, p.low * A; tp = (hi + lo + Ca) / 3; sm = tp.rolling(20).mean(); mad = pd.DataFrame(np.nan, index=idx, columns=cols); tv = tp.to_numpy()
for i in wi:
    if i >= 19: w = tv[i - 19:i + 1]; mad.iloc[i] = np.nanmean(np.abs(w - np.nanmean(w, axis=0)), axis=0)
cci = (tp - sm) / (0.015 * mad)
sd20 = Ca.rolling(20).std(); bb = (Ca < ma(20) - 2 * sd20)
cross = lambda a, b, n=5: ((a > b) & (a.shift(n) <= b.shift(n))) & ((a.shift(1) <= b.shift(1)).rolling(n).max().astype(bool) | True)
def crossed_up(a, b, n=5):   # 直近n日以内に a が b を下から上へ抜けた
    up = (a > b) & (a.shift(1) <= b.shift(1)); return up.rolling(n).max().fillna(0).astype(bool)
def crossed_dn(a, b, n=5):
    dn = (a < b) & (a.shift(1) >= b.shift(1)); return dn.rolling(n).max().fillna(0).astype(bool)
ema = lambda n: Ca.ewm(span=n, adjust=False).mean(); macd = ema(12) - ema(26); sig = macd.ewm(span=9, adjust=False).mean()
S["【売られすぎ】RSI2が10未満"] = r2 < 10
S["【売られすぎ】RSI14が30未満"] = r14 < 30
S["【売られすぎ】CCI20が-100未満"] = cci < -100
S["【売られすぎ】25日線からの乖離が-10%以下"] = dev25 <= -10
S["【売られすぎ】ボリンジャー(20日,-2σ)割れ"] = bb
S["【売られすぎ】3日連続の陰線(終値が前日比マイナス)"] = (dr < 0).rolling(3).sum() == 3
S["【売られすぎ】当日の寄り付きギャップが-3%以下"] = (O / Ca.shift(1) - 1) * 100 <= -3
S["【トレンド】5日線が25日線を上抜け(直近5日, ゴールデンクロス)"] = crossed_up(ma(5), ma(25))
S["【トレンド】25日線が75日線を上抜け(直近5日, ゴールデンクロス)"] = crossed_up(ma(25), ma(75))
S["【トレンド】5日線が25日線を下抜け(直近5日, デッドクロス)"] = crossed_dn(ma(5), ma(25))
S["【トレンド】MACDがシグナルを上抜け(直近5日)"] = crossed_up(macd, sig)
S["【トレンド】終値が200日線を上抜け(直近5日)"] = crossed_up(Ca, ma(200))
S["【トレンド】終値が200日線の上にある"] = Ca > ma(200)
S["【勢い】終値が過去250日の最高値を更新"] = Ca >= Ca.rolling(250, min_periods=200).max()
S["【勢い】RSI14が70超(買われすぎ)"] = r14 > 70
S["【勢い】出来高が60日平均の2倍超(5日平均)かつ週上昇"] = (V.rolling(5).mean() > 2 * V.rolling(60).mean()) & (Ca.pct_change(5) > 0)
g = p.adjf / p.adjf.shift(1); Dv = (p.close.shift(1) * (1 - 1 / g)).where(g > 1 + 1e-6).fillna(0); yld = Dv.rolling("365D").sum() / p.close * 100
S["【その他】配当利回り3%以上"] = yld >= 3
S["【その他】シグナルなし(全銘柄・基準)"] = pd.DataFrame(True, index=idx, columns=cols)
elig = (mask & O.notna() & Ca.notna()).to_numpy(); On = O.to_numpy(); Cn = Ca.to_numpy()
rows = []
base_w = {}
for k, i in enumerate(wi):
    e = i + 1; x_ = e + HOLD - 1
    p0 = On[e]; p1 = Cn[x_]; r = ((p1 / p0) * (1 - COST) / (1 + COST) - 1) * 100
    ok = elig[i] & ~np.isnan(r); base_w[idx[i]] = (r[ok] > 0).mean() * 100
    for name, df in S.items():
        m = df.to_numpy()[i]; m = np.where(pd.isna(m), False, m).astype(bool) & ok
        if m.any(): rows.append((idx[i], name, int(m.sum()), (r[m] > 0).sum(), r[m].sum()))
R = pd.DataFrame(rows, columns=["date", "sig", "n", "up", "sumr"]); R["bw"] = R.date.map(base_w)
out = []
for name, g_ in R.groupby("sig", sort=False):
    n = g_.n.sum(); up = g_.up.sum() / n * 100; mean = g_.sumr.sum() / n
    wd = (g_.up / g_.n * 100 - g_.bw); wavg = (wd * g_.n).sum() / n
    a = g_[g_.date < "2018-01-01"]; b = g_[g_.date >= "2018-01-01"]
    out.append(dict(シグナル=name, 件数=n, 出現週数=len(g_), 上昇割合=up, 同週の全銘柄との差pt=wavg, 平均リターン=mean, 前半=a.up.sum() / max(a.n.sum(), 1) * 100, 後半=b.up.sum() / max(b.n.sum(), 1) * 100))
O_ = pd.DataFrame(out); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
O_.to_pickle(SP + "sig_1m.pkl"); print(O_.round(1).sort_values("上昇割合", ascending=False).to_string(index=False))
