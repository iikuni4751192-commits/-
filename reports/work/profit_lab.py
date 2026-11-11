"""期待値・トータルリターンを最大化する検証の共通部品（主戦略の事例：翌営業日の寄り付きで買い、手仕舞いの規則を変えて比べる）"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/-")
from weekpick.universe import load_universe
from weekpick import data as WD
from weekpick.factors import eligible
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
COST = 0.001
u = load_universe("topix500"); p = WD.load_panel(list(u.ticker)); idx = p.close.index; mask = eligible(p)
A_ = p.adjf.ffill(); Cdf = (p.close * A_).ffill(limit=5)
C = Cdf.to_numpy().copy(); O = (p.open * A_).to_numpy().copy(); H = (p.high * A_).to_numpy().copy(); L = (p.low * A_).to_numpy().copy()
for arr in (O, H, L): m = np.isnan(arr); arr[m] = C[m]
MA25 = Cdf.rolling(25).mean().to_numpy(); cols = list(Cdf.columns); colpos = {t: j for j, t in enumerate(cols)}; pos = {d: i for i, d in enumerate(idx)}
ew = (1 + Cdf.pct_change(fill_method=None).where(mask.shift(1, fill_value=False)).mean(axis=1).fillna(0)).cumprod()
x = pd.read_pickle(SP + "crisis_ext.pkl").ffill(); x = x.reindex(x.index.union(idx)).ffill().reindex(idx)
a1 = (((x["^GSPC"] / x["^GSPC"].rolling(200).mean() - 1) * 100).shift(20) < -5) | (((ew / ew.rolling(200).mean() - 1) * 100).shift(20) < -5)
a2 = ew.rolling(252).apply(lambda v: (len(v) - 1 - np.argmax(v)) / 5, raw=True) > 40; a3 = x["HG=F"].pct_change(20) * 100 <= -8
NW = (a1.astype(int) + a2.astype(int) + a3.astype(int)).to_numpy()
def events():
    X = pd.read_pickle(SP + "free_main.pkl"); X["date"] = pd.to_datetime(X.date)
    noe = (X["直近4週に決算"] == 0) & (X["2週以内に決算"] == 0); ind = (X["業種内のシグナル銘柄割合%"] >= 30) | (X["業種比20日%"] > -5)
    hot = (X.sec1y > 60) | (X.r1y > 100); dv = X["配当利回り%"]
    X["e"] = [pos[d] + 1 for d in X.date]; X["j"] = [colpos[t] for t in X.t]
    return X, {"A 最終ルール(決算なし・業界ぐるみ・配当2.6%以上・上がりすぎ除外)": noe & ind & (dv >= 2.6) & ~hot, "B 手順3まで(決算なし・業界ぐるみ)": noe & ind, "C 手順2まで(決算なし)": noe}
def net(price, p0): return (price / p0) * (1 - COST) / (1 + COST) - 1
def run(e, j, hold=63, stop=None, tp=None, trail=None, trail_act=0.0, sig=None, sig_min=3, tstop=None, crisis=False, with_path=False):
    Hm = hold; o = O[e:e + Hm, j]; h = H[e:e + Hm, j]; l = L[e:e + Hm, j]; c = C[e:e + Hm, j]; n = len(c)
    if n < Hm or np.isnan(o[0]): return None
    p0 = o[0]; best_k = Hm - 1; best_p = c[Hm - 1]
    if np.isnan(best_p): best_p = c[~np.isnan(c)][-1]
    cand = []
    if stop is not None:
        lv = p0 * (1 - stop); hit = np.flatnonzero(l <= lv)
        if len(hit): k = hit[0]; cand.append((k, lv if (k == 0 or o[k] > lv) else o[k]))
    if tp is not None:
        lv = p0 * (1 + tp); hit = np.flatnonzero(h >= lv)
        if len(hit): k = hit[0]; cand.append((k, lv if (k == 0 or o[k] < lv) else o[k]))
    if trail is not None:
        cm = np.maximum.accumulate(h); pk = np.concatenate([[-np.inf], cm[:-1]]); act = pk >= p0 * (1 + trail_act); lv = pk * (1 - trail)
        hit = np.flatnonzero(act & (l <= lv) & (np.arange(Hm) >= 1))
        if len(hit): k = hit[0]; cand.append((k, lv[k] if o[k] > lv[k] else o[k]))
    nxt = lambda k: (k + 1, o[k + 1]) if k + 1 < Hm else (Hm - 1, c[Hm - 1])
    if sig is not None:
        m = MA25[e:e + Hm, j]; hit = np.flatnonzero((c >= m * (1 + sig)) & (np.arange(Hm) >= sig_min) & ~np.isnan(m))
        if len(hit): cand.append(nxt(hit[0]))
    if tstop is not None:
        md, b = tstop
        if md + 1 < Hm and c[md] / p0 - 1 <= -b: cand.append(nxt(md))
    if crisis:
        hit = np.flatnonzero((NW[e:e + Hm] >= 2) & (np.arange(Hm) >= 1))
        if len(hit): cand.append(nxt(hit[0]))
    if cand:
        k, pr = min(cand, key=lambda t: t[0])
        if k < best_k: best_k, best_p = k, pr
    r = net(best_p, p0) * 100
    if not with_path: return r, best_k + 1
    pathv = np.array([(C[e + k, j] / p0) if k < best_k else (best_p / p0) for k in range(best_k + 1)]) * (1 - COST) / (1 + COST)
    return r, best_k + 1, pathv
def stats(r, d, dates):
    r = np.asarray(r); d = np.asarray(d); dates = pd.to_datetime(dates)
    a, b = r[dates < "2018"], r[dates >= "2018"]
    return dict(件数=len(r), 平均=r.mean(), 中央値=np.median(r), 勝率=(r > 0).mean() * 100, 保有日数=d.mean(), 日あたり年率換算=r.mean() / d.mean() * 250, 下位5pct=np.percentile(r, 5), 大負け10=(r <= -10).mean() * 100, 前半平均=a.mean() if len(a) else np.nan, 後半平均=b.mean() if len(b) else np.nan)
def paired_t(r1, r0, dates):
    df = pd.DataFrame({"d": pd.to_datetime(dates), "x": np.asarray(r1) - np.asarray(r0)}); w = df.groupby("d").x.mean()
    b = w.groupby((w.index - pd.Timestamp("2000-01-07")).days // 91).mean(); return b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))
