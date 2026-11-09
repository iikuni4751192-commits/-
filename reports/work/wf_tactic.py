"""RSI2<10戦術・RSI14<30戦術を ①ウォークフォワード ②移動期間での優位性 で検証（wf_main.py と同じ方法）
使い方: python wf_tactic.py rsi2 | rsi14"""
import sys, itertools, os
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/-")
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
COST = 0.001
which = sys.argv[1]
X = pd.read_pickle(SP + ("rsi2_extra_events.pkl" if which == "rsi2" else "RSI14lt30_events.pkl"))
for c in X.columns:
    if X[c].dtype == object and c not in ("t", "sector"): X[c] = pd.to_numeric(X[c], errors="coerce")
X["sec1y"] = X["業種比1年%(直近1か月除く)"]; X["r1y"] = X["1年騰落%(直近1か月除く)"]
# 同じ週の全銘柄（基準）：キャッシュ
bf = SP + "wf_base_by_date.pkl"
base = pd.read_pickle(bf) if os.path.exists(bf) else {}
need = [pd.Timestamp(d) for d in sorted(X.date.unique()) if pd.Timestamp(d) not in base]
if need:
    from weekpick.universe import load_universe
    from weekpick import data as WD
    from weekpick.factors import eligible
    u = load_universe("topix500"); p = WD.load_panel(list(u.ticker)); idx = p.close.index; mask = eligible(p)
    A = p.adjf.ffill(); O = (p.open * A); Ca = (p.close * A).ffill(limit=5); V = p.volume; pos = {d: i for i, d in enumerate(idx)}
    for d in need:
        e = pos[d] + 1
        if e + 62 >= len(idx): continue
        ok = mask.loc[d] & (V.iloc[e] > 0) & O.iloc[e].notna()
        r = (Ca.iloc[e + 62] / O.iloc[e] * (1 - COST) / (1 + COST) - 1)[ok].dropna() * 100
        base[d] = (r.mean(), (r > 0).mean() * 100)
    pd.to_pickle(base, bf)
X["b_mean"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[0]); X["b_win"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[1])
X = X.dropna(subset=["b_mean", "r3m"]).copy(); X["edge"] = X.r3m - X.b_mean; X["wedge"] = X.up * 100 - X.b_win; X["year"] = X.date.dt.year
X = X.reset_index(drop=True)
earn = {"決算の除外なし": np.ones(len(X), bool), "直前4週を除外": (X["直近4週に決算"] == 0).values, "直前4週・直後2週を除外": ((X["直近4週に決算"] == 0) & (X["2週以内に決算"] == 0)).values}
ind = {"業界ぐるみ条件なし": np.ones(len(X), bool)}
for th in (20, 30, 40, 50): ind[f"業種の{th}%以上 or 業種比-5%以内"] = ((X["業種内のシグナル銘柄割合%"] >= th) | (X["業種比20日%"] > -5)).values
div = {f"配当{v}%以上" if v else "配当条件なし": (X["配当利回り%"] >= v).values if v else np.ones(len(X), bool) for v in (0, 2.0, 2.45, 3.0, 3.5, 4.0)}
hot = {"上がりすぎ除外なし": np.ones(len(X), bool)}
for v in (40, 60, 100): hot[f"業種比1年+{v}%超を除外"] = ~((X.sec1y > v) | (X.r1y > 100)).values
RULES = {" / ".join([a, b, c, h]): ma & mb & mc & mh for (a, ma), (b, mb), (c, mc), (h, mh) in itertools.product(earn.items(), ind.items(), div.items(), hot.items())}
if which == "rsi2":
    FINAL = "直前4週を除外 / 業界ぐるみ条件なし / 配当2.45%以上 / 上がりすぎ除外なし"; TITLE = "RSI2<10 戦術"
else:
    FINAL = "直前4週・直後2週を除外 / 業種の30%以上 or 業種比-5%以内 / 配当2.45%以上 / 上がりすぎ除外なし"; TITLE = "RSI14<30 戦術"
ALL = "決算の除外なし / 業界ぐるみ条件なし / 配当条件なし / 上がりすぎ除外なし"
def blk(d, c="edge"):
    w = d.groupby("date")[c].mean(); b = w.groupby((w.index - pd.Timestamp("2000-01-07")).days // 91).mean()
    return b.mean() / (b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 2 else np.nan
def stat(d):
    return f"件数{len(d):6d} 勝率{d.up.mean()*100:5.1f}% 基準の勝率{d.b_win.mean():5.1f}% 勝率の差{d.wedge.mean():+5.1f}pt 平均{d.r3m.mean():+5.1f}% 平均の差{d.edge.mean():+5.1f}pt t{blk(d):+.1f}"
print("=" * 110); print(f"{TITLE}  ① ウォークフォワード（{len(RULES)}通りから毎年選び直す。学習は前年までに結果が確定した事例、最低300件・40週）")
chosen, outs = [], []
for Y in range(2013, 2027):
    cut = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=100); tr = (X.date < cut).values; te = (X.year == Y).values
    if te.sum() == 0: continue
    best, bv = None, -1e9
    for nm, m in RULES.items():
        mm = tr & m
        if mm.sum() < 300: continue
        d = X[mm]
        if d.date.nunique() < 40: continue
        v = d.wedge.mean()
        if v > bv: best, bv = nm, v
    chosen.append((Y, best, bv)); outs.append(X[te & RULES[best]])
for Y, nm, v in chosen: print(f"   {Y}: {nm}")
OOS = (X.year >= 2013).values
W = pd.concat(outs)
print("\n【2013年以降の実際の成績】")
print("   毎年選び直したルール     ", stat(W))
print("   今の最終ルール           ", stat(X[OOS & RULES[FINAL]]))
print("   シグナルのすべて         ", stat(X[OOS & RULES[ALL]]))
ins = X[RULES[FINAL]]
print(f"   最終ルールの全期間での勝率の差 {ins.wedge.mean():+.1f}pt → ウォークフォワード {W.wedge.mean():+.1f}pt（残った割合 {W.wedge.mean()/ins.wedge.mean()*100:.0f}%）")
print("\n【各条件が毎年採用されたか（2013→2026）】")
for lab, key in [("直前4週・直後2週の除外", "直前4週・直後2週"), ("直前4週だけの除外", "/ 直前4週を除外"), ("業界ぐるみ", "業種の"), ("配当の条件", "配当条件なし"), ("上がりすぎの除外", "除外なし")]:
    if key == "/ 直前4週を除外": s = "".join("○" if nm.startswith("直前4週を除外") else "・" for _, nm, _ in chosen)
    elif key in ("配当条件なし", "除外なし"): s = "".join("・" if key in nm.split(" / ")[2 if key == "配当条件なし" else 3] else "○" for _, nm, _ in chosen)
    else: s = "".join("○" if key in nm else "・" for _, nm, _ in chosen)
    print(f"   {lab:16s} {s}")
print("   選ばれた配当の基準:", " ".join(nm.split(" / ")[2].replace("配当", "").replace("以上", "") for _, nm, _ in chosen))
print("\n" + "=" * 110); print(f"{TITLE}  ② 移動期間での優位性（直近3年間。基準＝同じ週の全銘柄）")
rows = []
for Y in range(2010, 2027):
    lo, hi = pd.Timestamp(f"{Y-2}-01-01"), pd.Timestamp(f"{Y}-12-31"); r = {"3年間の終わり": Y}
    for lab, nm in [("最終ルール", FINAL), ("全シグナル", ALL)]:
        d = X[RULES[nm] & (X.date >= lo).values & (X.date <= hi).values]
        r[f"{lab} 件数"] = len(d); r[f"{lab} 勝率"] = d.up.mean() * 100; r[f"{lab} 勝率の差"] = d.wedge.mean(); r[f"{lab} 平均の差"] = d.edge.mean(); r[f"{lab} t"] = blk(d)
    rows.append(r)
pd.set_option("display.width", 250); print(pd.DataFrame(rows).set_index("3年間の終わり").round(1).to_string())
print("\n【期間別：条件を1つずつ外した場合】")
F = RULES[FINAL]; noe4 = earn["直前4週を除外"]; noe6 = earn["直前4週・直後2週を除外"]; ind30 = ind["業種の30%以上 or 業種比-5%以内"]; dv = X["配当利回り%"].values
V = {"最終ルール": F}
if which == "rsi2":
    V.update({"配当条件を外す": noe4, "決算の除外を外す": dv >= 2.45, "直後2週も除外する": noe6 & (dv >= 2.45), "業界ぐるみを加える": F & ind30, "配当3.5%以上に上げる": noe4 & (dv >= 3.5), "配当4%以上だけ": noe4 & (dv >= 4)})
else:
    V.update({"業界ぐるみを外す": noe6 & (dv >= 2.45), "配当条件を外す": noe6 & ind30, "決算の除外を外す": ind30 & (dv >= 2.45), "配当3.5%以上に上げる": noe6 & ind30 & (dv >= 3.5), "配当4%以上だけ": noe6 & ind30 & (dv >= 4)})
V["全シグナル"] = RULES[ALL]
rows = []
for nm, m in V.items():
    r = {"ルール": nm}
    for lab, a, b in [("2008-12", 2008, 2012), ("2013-17", 2013, 2017), ("2018-22", 2018, 2022), ("2023-26", 2023, 2026)]:
        d = X[m & X.year.between(a, b).values]; r[f"{lab} 件数"] = len(d); r[f"{lab} 勝率"] = d.up.mean() * 100; r[f"{lab} 勝率の差"] = d.wedge.mean(); r[f"{lab} 平均の差"] = d.edge.mean()
    rows.append(r)
print(pd.DataFrame(rows).set_index("ルール").round(1).to_string())
print("\n【直近（2023〜2026年7月）の最終ルール：独立した局面ごと（13週以上空いたら別の局面）】")
D = X[F & (X.year >= 2023).values]
if len(D):
    d = pd.Series(sorted(D.date.unique())); ep = (d.diff().dt.days > 91).cumsum(); D = D.assign(ep=D.date.map(dict(zip(d, ep))))
    g = D.groupby("ep").agg(開始=("date", "min"), 終了=("date", "max"), 件数=("up", "size"), 勝率=("up", "mean"), 基準の勝率=("b_win", "mean"), 勝率の差=("wedge", "mean"), 平均の差=("edge", "mean"))
    g["勝率"] *= 100; print(g.round(1).to_string(), f"\n   局面数 {len(g)}")
print("\n【直近の相場環境の目安：米10年金利の20日変化（最終ルール）】")
rt = X["米10年金利20日変化"].values
for lab, a, b in [("2008-17", 2008, 2017), ("2018-22", 2018, 2022), ("2023-26", 2023, 2026)]:
    per = X.year.between(a, b).values
    q = [f"{nm} {X[F & per & m].up.mean()*100:5.1f}%({(F & per & m).sum()})" for nm, m in [("低下", rt <= -0.1), ("横ばい", (rt > -0.1) & (rt <= 0.05)), ("上昇", rt > 0.05)]]
    print(f"   {lab}: " + " ／ ".join(q))
