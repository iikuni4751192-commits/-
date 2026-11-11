"""買いのタイミング：翌営業日の寄り付き以外の買い方で、平均リターン（期待値）がどう変わるか（手仕舞いは同じ日＝寄り付きから63営業日目の終値）"""
import sys, numpy as np, pandas as pd
from profit_lab import *
X, SETS = events(); X = X[(X.e + 189 + 2 < len(idx))].copy(); X["nw"] = [NW[pos[d]] for d in X.date]; X = X[X.nw <= 1].copy()
pd.set_option("display.width", 260)
def entry_returns(S):
    out = {k: [] for k in ["E0 月曜の寄り付き(基準)", "E1 月曜の引け", "E2 火曜の寄り付き", "E3 1週間待って翌月曜の寄り付き", "E4 前週末終値-2%の指値(5日)", "E5 前週末終値-4%の指値(5日)",
                          "E6 前週末終値-6%の指値(10日)", "E7 前日高値を超えた日の引け(10日以内)", "E8 半分は月曜、半分は-6%の指値(15日・無ければ15日目)"]}
    gap = []
    for e, j in zip(S.e.values, S.j.values):
        o = O[e:e + 63, j]; h = H[e:e + 63, j]; l = L[e:e + 63, j]; c = C[e:e + 63, j]; ref = C[e - 1, j]; xp = c[62]
        f = lambda pe: net(xp, pe) * 100
        out["E0 月曜の寄り付き(基準)"].append(f(o[0])); out["E1 月曜の引け"].append(f(c[0])); out["E2 火曜の寄り付き"].append(f(o[1])); out["E3 1週間待って翌月曜の寄り付き"].append(f(o[5]))
        def lim(x_, days):
            lv = ref * (1 - x_); hit = np.flatnonzero(l[:days] <= lv)
            if not len(hit): return np.nan
            k = hit[0]; return f(min(o[k], lv) if k > 0 else min(o[0], lv))
        out["E4 前週末終値-2%の指値(5日)"].append(lim(.02, 5)); out["E5 前週末終値-4%の指値(5日)"].append(lim(.04, 5)); out["E6 前週末終値-6%の指値(10日)"].append(lim(.06, 10))
        cf = np.flatnonzero(c[1:10] > h[:9]); out["E7 前日高値を超えた日の引け(10日以内)"].append(f(c[1 + cf[0]]) if len(cf) else np.nan)
        lv = ref * .94; hit = np.flatnonzero(l[:15] <= lv); p2 = min(o[hit[0]], lv) if len(hit) and hit[0] > 0 else (min(o[0], lv) if len(hit) else o[15])
        out["E8 半分は月曜、半分は-6%の指値(15日・無ければ15日目)"].append((f(o[0]) + f(p2)) / 2); gap.append(o[0] / ref - 1)
    return pd.DataFrame(out), np.array(gap) * 100
for sname, m in SETS.items():
    S = X[m.reindex(X.index).fillna(False)].copy()
    if len(S) < 500: continue
    R, gap = entry_returns(S); S["gap"] = gap
    print("=" * 170); print(sname, f"（{len(S)}件）")
    rows = []
    for k in R.columns:
        r = R[k]; ok = r.notna(); rr = r.fillna(0)
        d = pd.DataFrame({"d": S.date.values, "x": rr.values - R.iloc[:, 0].values}); w = d.groupby("d").x.mean(); b = w.groupby((w.index - pd.Timestamp("2000-01-07")).days // 91).mean()
        rows.append({"買い方": k, "約定率%": ok.mean() * 100, "約定した分の平均%": r[ok].mean(), "機会あたりの平均%(未約定は0)": rr.mean(), "約定した分の中央値": r[ok].median(), "勝率%": (r[ok] > 0).mean() * 100, "下位5%": r[ok].quantile(.05),
                     "前半(機会あたり)": rr[S.date.values < np.datetime64("2018-01-01")].mean(), "後半": rr[S.date.values >= np.datetime64("2018-01-01")].mean(), "基準との差のt": b.mean() / (b.std(ddof=1) / np.sqrt(len(b))) if k[:2] != "E0" else np.nan})
    print(pd.DataFrame(rows).set_index("買い方").round(2).to_string())
    print("  [月曜の寄り付きの前週末終値からのギャップ別：寄り付きから63日目までの平均%（件数）]")
    for lab, a, b in [("-4%以下に窓開け下落", -99, -4), ("-4〜-2%", -4, -2), ("-2〜0%", -2, 0), ("0〜+2%", 0, 2), ("+2%以上に窓開け上昇", 2, 99)]:
        mm = (S.gap >= a) & (S.gap < b); print(f"    {lab:22s} 平均{R.iloc[:,0][mm.values].mean():+6.1f}%  勝率{(R.iloc[:,0][mm.values]>0).mean()*100:5.1f}%  ({int(mm.sum())})")
