"""RSI14<30 戦術のスクリーニング（判定日を引数で指定、省略時は最新の終値）"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/-")
from weekpick.universe import load_universe
from weekpick import data as WD
from weekpick.factors import eligible
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
u = load_universe("topix500"); p = WD.load_panel(list(u.ticker)); C = p.close; idx = C.index
d = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else idx[-1]; d = idx[idx <= d][-1]
info = u.set_index("ticker"); sec = info.reindex(C.columns)["sector"]; mask = eligible(p); el = mask.loc[d]
Ca = (C * p.adjf).ffill(limit=5)
w = lambda x, n: x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
def rsi(x, n): dd = x.diff(); return 100 - 100 / (1 + w(dd.clip(lower=0), n) / w((-dd).clip(lower=0), n))
r14 = rsi(Ca, 14).loc[d]; dev = ((Ca / Ca.rolling(25).mean() - 1) * 100).loc[d]
sig25 = (dev <= -10) & el; share = sig25.groupby(sec).transform("sum") / el.groupby(sec).transform("sum") * 100
r20 = Ca.pct_change(20).loc[d]; sec20 = r20.where(el).groupby(sec).transform("mean")
g = p.adjf / p.adjf.shift(1); Dv = (C.shift(1) * (1 - 1 / g)).where(g > 1 + 1e-6).fillna(0); yld = (Dv.rolling("365D").sum() / C).loc[d] * 100
x = pd.read_pickle(SP + "crisis_ext.pkl").ffill(); x = x.reindex(x.index.union(idx)).ffill().reindex(idx)
ew = (1 + Ca.pct_change(fill_method=None).where(mask.shift(1, fill_value=False)).mean(axis=1).fillna(0)).cumprod()
s1 = ((x["^GSPC"] / x["^GSPC"].rolling(200).mean() - 1) * 100).shift(20).loc[d]; j1 = ((ew / ew.rolling(200).mean() - 1) * 100).shift(20).loc[d]
k = idx.get_loc(d); wk = (251 - np.argmax(ew.iloc[k - 251:k + 1].values)) / 5; cu = x["HG=F"].pct_change(20).loc[d] * 100
n_sign = int((s1 < -5) or (j1 < -5)) + int(wk > 40) + int(cu <= -8); tnx = x["^TNX"].diff(20).loc[d]
print(f"判定日 {d.date()}  警戒サイン {n_sign}個（S&P {s1:+.1f}%、日本株 {j1:+.1f}%、高値から {wk:.1f}週、銅 {cu:+.1f}%）  米10年金利20日変化 {tnx:+.2f}"
      f" → {'金額を半分' if tnx > -0.03 else '通常の金額'}  VIX {x['^VIX'].loc[d]:.1f}")
E = pd.read_csv(SP + "surprise.csv"); E["d0"] = pd.to_datetime(E.jst).dt.normalize()
rows = []
for t in C.columns[((r14 < 30) & el).values]:
    e = E[E.t == t]; nxt = e[e.d0 > d].d0.min(); ypast = e[(e.d0 <= d) & (e.d0 > d - pd.Timedelta(days=28))].d0.max()
    rows.append({"コード": t[:4], "銘柄": info.at[t, "name"], "業種": sec[t], "終値": C.at[d, t], "RSI14": r14[t], "25日乖離%": dev[t],
                 "業種内25日乖離割合%": share[t], "業種比20日%": (r20[t] - sec20[t]) * 100, "配当利回り%": yld[t], "Yahoo次の決算": nxt, "Yahoo直近4週": ypast})
R = pd.DataFrame(rows); R["業界ぐるみ"] = (R["業種内25日乖離割合%"] >= 30) | (R["業種比20日%"] > -5)
print("RSI14<30:", len(R), "銘柄  うち配当2.45%以上:", int((R["配当利回り%"] >= 2.45).sum()), " 業界ぐるみ:", int(R.業界ぐるみ.sum()))
R.to_csv(f"/home/user/-/reports/work/screen_rsi14_{d.date()}.csv", index=False)
