"""ポートフォリオ：候補の優先順位・候補の範囲・資金の遊びが、トータルリターンに与える影響（port_lib.py の simulate を使用）"""
import sys, numpy as np, pandas as pd
from port_lib import *
EXITS = {"固定126日": dict(hold=126), "損切り-25% 126日": dict(hold=126, stop=.25), "固定63日": dict(hold=63)}
PRIO = {"配当の高い順": lambda r: r["配当利回り%"], "ボラの高い順": lambda r: r["ボラ(60日,年率%)"], "25日乖離の深い順": lambda r: -r["25日乖離%"], "52週高値から遠い順": lambda r: -r["52週高値比%"],
        "順位の合計(配当＋ボラ＋深さ)": None, "ランダム": None}
S_ = {k: v for k, v in SETS.items()}
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
rng = np.random.default_rng(0); res = []
ewr = ew.pct_change().fillna(0).to_numpy()
for sname in ["B 手順3まで(決算なし・業界ぐるみ)", "C 手順2まで(決算なし)"]:
    S = X[S_[sname].reindex(X.index).fillna(False)].copy()
    S["rk"] = S.groupby("date")["配当利回り%"].rank(pct=True) + S.groupby("date")["ボラ(60日,年率%)"].rank(pct=True) + S.groupby("date")["25日乖離%"].rank(pct=True, ascending=False)
    for en, spec in EXITS.items():
        base = []
        for _, r in S.iterrows():
            o = run(int(r.e), int(r.j), with_path=True, **spec)
            if o is not None: base.append((r, o[2]))
        for pn, fn in PRIO.items():
            for N in (10, 20):
                if pn == "順位の合計(配当＋ボラ＋深さ)": pr = [r["rk"] for r, v in base]
                elif pn == "ランダム": pr = list(rng.random(len(base)))
                else: pr = [fn(r) for r, v in base]
                trades = [dict(e=int(r.e), v=v, prio=float(np.nan_to_num(q)), nw=r.nw) for (r, v), q in zip(base, pr)]
                s, taken, expo = simulate(trades, N); mt = metrics(s); mt.update(候補=sname[:2], 手仕舞い=en, 優先順位=pn, N=N, 買った数=taken, 投資割合=expo * 100)
                a = s[s.index < "2018-01-01"]; b = s[s.index >= "2018-01-01"]
                mt["前半の年率"] = (a.iloc[-1] / a.iloc[0]) ** (365.25 / (a.index[-1] - a.index[0]).days) * 100 - 100; mt["後半の年率"] = (b.iloc[-1] / b.iloc[0]) ** (365.25 / (b.index[-1] - b.index[0]).days) * 100 - 100
                res.append(mt)
R = pd.DataFrame(res); R.to_pickle(SP + "profit_port2.pkl")
for (c, e), g in R.groupby(["候補", "手仕舞い"]):
    print("=" * 150); print(f"候補={c}  手仕舞い={e}"); print(g.pivot(index="優先順位", columns="N", values="年率").round(2).join(g.pivot(index="優先順位", columns="N", values="最大下落").round(1), rsuffix="(最大下落)").join(g.pivot(index="優先順位", columns="N", values="シャープ").round(2), rsuffix="(シャープ)").to_string())
# 資金の遊び：待機資金を日本株の均等平均で運用（警戒サイン2個以上の日は現金）
print("\n" + "=" * 150); print("待機資金の使い方（候補B・損切り-25% 126日・配当の高い順）")
S = X[S_["B 手順3まで(決算なし・業界ぐるみ)"].reindex(X.index).fillna(False)]
trades = []
for _, r in S.iterrows():
    o = run(int(r.e), int(r.j), with_path=True, hold=126, stop=.25)
    if o is not None: trades.append(dict(e=int(r.e), v=o[2], prio=r["配当利回り%"], nw=r.nw))
flag = (NW <= 1).astype(float)
for N in (10, 20):
    for lab, cr in [("待機資金は現金のまま", None), ("待機資金を日本株の均等平均で運用(警戒サイン2個以上の日は現金)", ewr * flag), ("待機資金を日本株の均等平均で運用(常に)", ewr)]:
        s, taken, expo = simulate(trades, N, cash_ret=cr); mt = metrics(s); print(f"  N={N:2d} {lab:48s} 年率{mt['年率']:6.2f}% 最大下落{mt['最大下落']:7.1f}% 変動{mt['年率の変動']:5.1f}% シャープ{mt['シャープ']:.2f} 最終倍率{mt['最終倍率']:.2f}")
bm = ew.reindex(s.index); yrs = (bm.index[-1] - bm.index[0]).days / 365.25
print("  参考：日本株の均等平均を持ち続けた場合 年率%.1f%% 最大下落%.1f%%" % ((bm.iloc[-1] / bm.iloc[0]) ** (1 / yrs) * 100 - 100, (bm / bm.cummax() - 1).min() * 100))
