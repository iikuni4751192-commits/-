"""設計案のまとめ比較（ポートフォリオ、1銘柄=資産の1/N、警戒サイン0〜1個の週のみ買い、1個の週は半額）"""
import sys, numpy as np, pandas as pd
from port_lib import *
ewr = ew.pct_change().fillna(0).to_numpy(); flag = (NW <= 1).astype(float)
def prio_fn(name):
    if name == "配当": return lambda r: r["配当利回り%"]
    if name == "ボラ": return lambda r: r["ボラ(60日,年率%)"]
    if name == "順位合計": return lambda r: r["rk"]
DES = [("現行に近い案: A・配当順・固定63日", "A", "配当", dict(hold=63), False), ("A・配当順・固定126日", "A", "配当", dict(hold=126), False), ("A・ボラ順・固定126日", "A", "ボラ", dict(hold=126), False),
       ("B・配当順・固定126日", "B", "配当", dict(hold=126), False), ("B・ボラ順・固定126日", "B", "ボラ", dict(hold=126), False), ("B・順位合計・損切り-25% 126日", "B", "順位合計", dict(hold=126, stop=.25), False),
       ("B・ボラ順・損切り-25% 126日", "B", "ボラ", dict(hold=126, stop=.25), False), ("C・ボラ順・損切り-25% 126日", "C", "ボラ", dict(hold=126, stop=.25), False),
       ("B・ボラ順・損切り-25% 126日＋待機資金を日本株の均等平均で運用", "B", "ボラ", dict(hold=126, stop=.25), True), ("B・順位合計・損切り-25% 126日＋待機資金を日本株の均等平均で運用", "B", "順位合計", dict(hold=126, stop=.25), True)]
key = {"A": "A 最終ルール(決算なし・業界ぐるみ・配当2.6%以上・上がりすぎ除外)", "B": "B 手順3まで(決算なし・業界ぐるみ)", "C": "C 手順2まで(決算なし)"}
cache = {}; rows = []
for nm, pool, pr, spec, overlay in DES:
    S = X[SETS[key[pool]].reindex(X.index).fillna(False)].copy()
    S["rk"] = S.groupby("date")["配当利回り%"].rank(pct=True) + S.groupby("date")["ボラ(60日,年率%)"].rank(pct=True) + S.groupby("date")["25日乖離%"].rank(pct=True, ascending=False)
    ck = (pool, tuple(sorted(spec.items())))
    if ck not in cache:
        cache[ck] = [(r, run(int(r.e), int(r.j), with_path=True, **spec)) for _, r in S.iterrows()]
    base = [(r, o) for r, o in cache[ck] if o is not None]; fn = prio_fn(pr)
    for N in (10, 20):
        trades = [dict(e=int(r.e), v=o[2], prio=float(np.nan_to_num(fn(r))), nw=r.nw) for r, o in base]
        s, taken, expo = simulate(trades, N, cash_ret=(ewr * flag if overlay else None)); mt = metrics(s)
        a = s[s.index < "2018-01-01"]; b = s[s.index >= "2018-01-01"]; ya = lambda z: (z.iloc[-1] / z.iloc[0]) ** (365.25 / (z.index[-1] - z.index[0]).days) * 100 - 100
        mt.update(設計=nm, N=N, 買った数=taken, 投資割合=expo * 100, 前半の年率=ya(a), 後半の年率=ya(b), 前半の最大下落=(a / a.cummax() - 1).min() * 100, 後半の最大下落=(b / b.cummax() - 1).min() * 100); rows.append(mt)
R = pd.DataFrame(rows); pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 70)
print(R[["設計", "N", "年率", "最終倍率", "最大下落", "シャープ", "前半の年率", "後半の年率", "前半の最大下落", "後半の最大下落", "買った数", "投資割合"]].round(2).to_string(index=False))
R.to_pickle(SP + "profit_final.pkl")
