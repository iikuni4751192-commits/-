"""主戦略（25日乖離≤-10%）を ①ウォークフォワード ②移動期間での優位性の監視 で検証"""
import sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/-")
from weekpick.universe import load_universe
from weekpick import data as WD
from weekpick.factors import eligible
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
COST = 0.001
X = pd.read_pickle(SP + "runup_events.pkl")
X["sec1y"] = X["業種比 1年%"]; X["r1y"] = X["1年騰落%(直近1か月除く)"]
# --- 同じ週の全銘柄（基準）の3か月リターン：翌営業日寄り→63営業日目の終値 ---
u = load_universe("topix500"); p = WD.load_panel(list(u.ticker)); idx = p.close.index; mask = eligible(p)
A = p.adjf.ffill(); O = (p.open * A); Ca = (p.close * A).ffill(limit=5); V = p.volume
pos = {d: i for i, d in enumerate(idx)}
base = {}
for d in sorted(X.date.unique()):
    e = pos[pd.Timestamp(d)] + 1
    if e + 62 >= len(idx): continue
    ok = mask.loc[d] & (V.iloc[e] > 0) & O.iloc[e].notna()
    r = (Ca.iloc[e + 62] / O.iloc[e] * (1 - COST) / (1 + COST) - 1)[ok].dropna() * 100
    base[pd.Timestamp(d)] = (r.mean(), (r > 0).mean() * 100)
X["b_mean"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[0]); X["b_win"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[1])
X = X.dropna(subset=["b_mean"]).copy(); X["edge"] = X.r3m - X.b_mean; X["wedge"] = X.up * 100 - X.b_win; X["year"] = X.date.dt.year
X.to_pickle(SP + "wf_main_events.pkl")
# --- ルールの部品 ---
earn = {"決算の除外なし": np.ones(len(X), bool), "直前4週を除外": (X["直近4週に決算"] == 0).values, "直前4週・直後2週を除外": ((X["直近4週に決算"] == 0) & (X["2週以内に決算"] == 0)).values}
ind = {"業界ぐるみ条件なし": np.ones(len(X), bool)}
for th in (20, 30, 40, 50): ind[f"業種の{th}%以上 or 業種比-5%以内"] = ((X["業種内のシグナル銘柄割合%"] >= th) | (X["業種比20日%"] > -5)).values
div = {f"配当{v}%以上" if v else "配当条件なし": (X["配当利回り%"] >= v).values if v else np.ones(len(X), bool) for v in (0, 2.0, 2.6, 3.0, 3.5)}
hot = {"上がりすぎ除外なし": np.ones(len(X), bool)}
for v in (40, 60, 100): hot[f"業種比1年+{v}%超を除外"] = ~((X.sec1y > v) | (X.r1y > 100)).values
RULES = {}
for (a, ma), (b, mb), (c, mc), (h, mh) in itertools.product(earn.items(), ind.items(), div.items(), hot.items()):
    RULES[" / ".join([a, b, c, h])] = ma & mb & mc & mh
FINAL = "直前4週・直後2週を除外 / 業種の30%以上 or 業種比-5%以内 / 配当2.6%以上 / 業種比1年+60%超を除外"
ALL = "決算の除外なし / 業界ぐるみ条件なし / 配当条件なし / 上がりすぎ除外なし"
NOE_IND = "直前4週・直後2週を除外 / 業種の30%以上 or 業種比-5%以内 / 配当条件なし / 上がりすぎ除外なし"
def blk(d, c="edge"):
    w = d.groupby("date")[c].mean(); b = w.groupby((w.index - pd.Timestamp("2000-01-07")).days // 91).mean()
    return b.mean() / (b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 2 else np.nan
def stat(d):
    if len(d) == 0: return dict(件数=0)
    return dict(件数=len(d), 勝率=d.up.mean() * 100, 基準の勝率=d.b_win.mean(), 勝率の差=d.wedge.mean(), 平均=d.r3m.mean(), 基準との差=d.edge.mean(), t=blk(d))
# --- ① ウォークフォワード ---
crit_opts = ["勝率の差", "基準との差"]
out = {k: [] for k in crit_opts}; chosen = {k: [] for k in crit_opts}
for Y in range(2013, 2027):
    cut = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=100)  # 3か月の結果が確定した事例だけで学習
    tr = (X.date < cut).values; te = (X.year == Y).values
    if te.sum() == 0: continue
    for crit in crit_opts:
        best, bv = None, -1e9
        for nm, m in RULES.items():
            d = X[tr & m]
            if len(d) < 300 or d.date.nunique() < 40: continue
            v = d.wedge.mean() if crit == "勝率の差" else d.edge.mean()
            if v > bv: best, bv = nm, v
        chosen[crit].append((Y, best))
        out[crit].append(X[te & RULES[best]].assign(rule=best))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
fmt = lambda s: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()}
OOS = X.year >= 2013
print("=" * 110); print("① ウォークフォワード（毎年、その年より前に結果が確定したデータだけで300通りのルールから最良を選び、その年に適用）")
print("   検証する年：2013〜2026年（2026年は7月までのシグナル）\n")
for crit in crit_opts:
    print(f"【選び方：学習期間で『{crit}』が最大のルール】")
    for Y, nm in chosen[crit]: print(f"   {Y}: {nm}")
    W = pd.concat(out[crit]); print("   → 2013年以降の実際の成績:", fmt(stat(W)), "\n")
print("【比較：2013年以降の同じ期間】")
for lab, nm in [("最終ルール（全期間を見て決めた今の戦略）", FINAL), ("決算なし＋業界ぐるみだけ", NOE_IND), ("25日乖離≤-10%のすべて", ALL)]:
    print(f"   {lab:28s}", fmt(stat(X[OOS & RULES[nm]])))
print("\n【学習期間（その時点まで）での見かけの成績 と 翌年の実際の成績】（選び方：勝率の差）")
rows = []
for (Y, nm), te_d in zip(chosen["勝率の差"], out["勝率の差"]):
    cut = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=100); trd = X[(X.date < cut).values & RULES[nm]]
    fin = X[(X.year == Y).values & RULES[FINAL]]; al = X[(X.year == Y).values & RULES[ALL]]
    rows.append({"年": Y, "学習での勝率の差": trd.wedge.mean(), "翌年の勝率の差": te_d.wedge.mean(), "翌年の件数": len(te_d),
                 "翌年の勝率": te_d.up.mean() * 100, "最終ルールの勝率": fin.up.mean() * 100 if len(fin) else np.nan, "最終ルールの件数": len(fin),
                 "全シグナルの勝率": al.up.mean() * 100, "基準(全銘柄)の勝率": al.b_win.mean()})
R = pd.DataFrame(rows).set_index("年"); print(R.round(1).to_string())
ins = X[RULES[FINAL]]; print(f"\n   最終ルールの全期間（学習データそのもの）での勝率の差 {ins.wedge.mean():+.1f}pt ／ 2013年以降に限ると {X[OOS & RULES[FINAL]].wedge.mean():+.1f}pt")
W = pd.concat(out["勝率の差"]); print(f"   ウォークフォワードでの勝率の差 {W.wedge.mean():+.1f}pt → 残った割合 {W.wedge.mean() / ins.wedge.mean() * 100:.0f}%")
# 部品ごと：ウォークフォワードで毎年採用されたか
print("\n【各条件が毎年のウォークフォワードで採用されたか】（選び方：勝率の差）")
for part in ["直前4週・直後2週を除外", "業種の", "配当", "上がりすぎ除外なし"]:
    print(f"   {part}: ", "".join("○" if part in nm else "・" for _, nm in chosen["勝率の差"]), "（2013→2026）")
# --- ② 移動期間での優位性 ---
print("\n" + "=" * 110); print("② 移動期間での優位性（直近3年間のシグナル。基準＝同じ週の全銘柄。t は13週ブロック）")
rows = []
for Y in range(2010, 2027):
    lo, hi = pd.Timestamp(f"{Y-2}-01-01"), pd.Timestamp(f"{Y}-12-31")
    r = {"3年間の終わり": Y}
    for lab, nm in [("最終ルール", FINAL), ("決算なし＋業界ぐるみ", NOE_IND), ("全シグナル", ALL)]:
        d = X[RULES[nm] & (X.date >= lo).values & (X.date <= hi).values]
        r[f"{lab} 件数"] = len(d); r[f"{lab} 勝率の差"] = d.wedge.mean(); r[f"{lab} 平均の差"] = d.edge.mean(); r[f"{lab} t"] = blk(d)
    rows.append(r)
print(pd.DataFrame(rows).set_index("3年間の終わり").round(1).to_string())
print("\n【1年ごと】")
rows = []
for Y in range(2008, 2027):
    r = {"年": Y}
    for lab, nm in [("最終ルール", FINAL), ("全シグナル", ALL)]:
        d = X[RULES[nm] & (X.year == Y).values]
        r[f"{lab} 件数"] = len(d); r[f"{lab} 勝率"] = d.up.mean() * 100 if len(d) else np.nan; r[f"{lab} 勝率の差"] = d.wedge.mean() if len(d) else np.nan; r[f"{lab} 平均の差"] = d.edge.mean() if len(d) else np.nan
        r["局面(週)数"] = d.date.nunique() if lab == "最終ルール" else r.get("局面(週)数")
    rows.append(r)
print(pd.DataFrame(rows).set_index("年").round(1).to_string())
