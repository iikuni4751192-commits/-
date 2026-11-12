"""少額資金・100株単位・信用取引（金利あり・追証あり）で、期待値重視の手法（候補B・順位合計・126日）がどこまで通用するかを検証"""
import sys, numpy as np, pandas as pd
from port_lib import *
S = X[SETS["B 手順3まで(決算なし・業界ぐるみ)"].reindex(X.index).fillna(False)].copy()
S["rk"] = S.groupby("date")["配当利回り%"].rank(pct=True) + S.groupby("date")["ボラ(60日,年率%)"].rank(pct=True) + S.groupby("date")["25日乖離%"].rank(pct=True, ascending=False)
S["lot"] = [100 * O[int(e), int(j)] for e, j in zip(S.e, S.j)]
# 126営業日が暦で何日か
cal = np.array([(idx[min(int(e) + 126, len(idx) - 1)] - idx[int(e)]).days for e in S.e]); print("126営業日の暦日数: 平均%.0f 最小%d 最大%d 182日超の割合%.0f%%" % (cal.mean(), cal.min(), cal.max(), (cal > 182).mean() * 100))
cal120 = np.array([(idx[min(int(e) + 120, len(idx) - 1)] - idx[int(e)]).days for e in S.e]); print("120営業日の暦日数: 平均%.0f 最大%d 182日超の割合%.0f%%" % (cal120.mean(), cal120.max(), (cal120 > 182).mean() * 100))
print("\n100株の購入代金（候補B、買う日の始値×100、現在の売買単位換算）の分布")
for lab, m in [("全期間", S.date >= "2000"), ("2018年以降", S.date >= "2018")]:
    q = S[m].lot; print(f"  {lab}: 中央値{q.median()/1e4:.1f}万円 25%点{q.quantile(.25)/1e4:.1f}万 75%点{q.quantile(.75)/1e4:.1f}万 ／ " + " ".join(f"{t}万円以下{(q<=t*1e4).mean()*100:.0f}%" for t in (10, 20, 30, 50, 100)))
S["r126"] = [run(int(e), int(j), hold=126)[0] for e, j in zip(S.e, S.j)]
print("\n100株の購入代金別の126日後の平均リターン（候補B）")
for lab, a, b in [("10万円以下", 0, 10e4), ("10〜20万", 10e4, 20e4), ("20〜50万", 20e4, 50e4), ("50〜100万", 50e4, 100e4), ("100万超", 100e4, 1e12)]:
    d = S[(S.lot > a) & (S.lot <= b)]; print(f"  {lab:10s} 件数{len(d):5d} 平均{d.r126.mean():+6.1f}% 中央{d.r126.median():+6.1f}% 勝率{(d.r126>0).mean()*100:5.1f}% 下位5%{d.r126.quantile(.05):+6.1f} 前半{d[d.date<'2018'].r126.mean():+6.1f} 後半{d[d.date>='2018'].r126.mean():+6.1f}")
RATE = 0.028
def sim(trades, cap, L, N=10, rate=RATE, mm=0.20):
    ent = {}
    for tr in trades: ent.setdefault(tr["e"], []).append(tr)
    t0 = min(ent); t1 = min(max(tr["e"] + len(tr["v"]) for tr in trades) + 1, len(idx) - 1)
    cash = cap; pos = []; eqs = np.full(len(idx), np.nan); prev = cap; liq = 0; bought = 0; skip_cost = 0; skip_full = 0; expo = []
    for t in range(t0, t1 + 1):
        pv_prev = sum(a * v[min(t - 1 - e, len(v) - 1)] for a, e, v in pos)
        if L > 1 and pv_prev > 0: cash -= rate / 250 * pv_prev
        if t in ent:
            for tr in sorted(ent[t], key=lambda z: -z["prio"]):
                pvn = sum(a * v[min(t - 1 - e, len(v) - 1)] for a, e, v in pos); bp = L * prev - pvn
                tgt = L * prev / N * (0.5 if tr["nw"] == 1 else 1.0); c0 = tr["lot"]
                n = int(tgt // c0)
                if n < 1 and c0 <= L * prev / 3 * (0.5 if tr["nw"] == 1 else 1.0) * 1.0: n = 1
                if n < 1: skip_cost += 1; continue
                n = int(min(n, bp // c0, (cash // c0) if L == 1 else n))
                if n < 1: skip_full += 1; continue
                cash -= n * c0; pos.append([n * c0, tr["e"], tr["v"]]); bought += 1
        tot = 0.0; keep = []
        for a, e, v in pos:
            k = t - e; val = a * v[min(k, len(v) - 1)]
            if k >= len(v) - 1: cash += val
            else: keep.append([a, e, v]); tot += val
        pos = keep; eq = cash + tot
        if L > 1 and tot > 0 and eq / tot < mm:   # 追証＝全建玉を引け値で手仕舞い
            cash += tot; tot = 0; pos = []; eq = cash; liq += 1
        eqs[t] = eq; prev = max(eq, 1e-9); expo.append(tot / prev if prev > 0 else 0)
        if eq <= 0: eqs[t:] = 0; break
    s = pd.Series(eqs, index=idx).dropna(); return s, bought, skip_cost, skip_full, liq, float(np.mean(expo))
def summarize(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25; fin = s.iloc[-1] / s.iloc[0]
    a = s[s.index < "2018-01-01"]; b = s[s.index >= "2018-01-01"]
    ya = lambda z: ((z.iloc[-1] / z.iloc[0]) ** (365.25 / max((z.index[-1] - z.index[0]).days, 1)) - 1) * 100 if z.iloc[0] > 0 and z.iloc[-1] > 0 else -100
    return dict(最終倍率=fin, 年率=(fin ** (1 / yrs) - 1) * 100 if fin > 0 else -100, 最大下落=(s / s.cummax() - 1).min() * 100, 前半=ya(a), 後半=ya(b))
EXITS = {"固定126日(一般信用・現物向け)": dict(hold=126), "損切り-25% 126日(一般信用・現物向け)": dict(hold=126, stop=.25), "固定120日(制度信用の6か月期限に収まる)": dict(hold=120)}
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300); rows = []
for en, spec in EXITS.items():
    base = [(r, run(int(r.e), int(r.j), with_path=True, **spec)) for _, r in S.iterrows()]
    trades = [dict(e=int(r.e), v=o[2], prio=float(r["rk"]), nw=r.nw, lot=r["lot"]) for r, o in base if o is not None]
    for cap in (30e4, 50e4, 100e4, 200e4, 500e4, 1000e4):
        for L, lab in [(1, "現物"), (2, "信用2倍"), (3, "信用3倍")]:
            if "制度" in en and L == 1: continue
            if "一般" in en and L == 1 and False: continue
            s, bought, sc, sf, liq, expo = sim(trades, cap, L); m = summarize(s); m.update(手仕舞い=en, 資金=f"{int(cap/1e4)}万円", 取引=lab, 買った数=bought, 買えなかった=sc, 枠不足=sf, 追証=liq, 平均の建玉比率=expo); rows.append(m)
R = pd.DataFrame(rows); R.to_pickle(SP + "lots_sim.pkl")
for en, g in R.groupby("手仕舞い", sort=False):
    print("=" * 170); print(en)
    print(g.set_index(["資金", "取引"])[["最終倍率", "年率", "最大下落", "前半", "後半", "買った数", "買えなかった", "枠不足", "追証", "平均の建玉比率"]].round(2).to_string())
