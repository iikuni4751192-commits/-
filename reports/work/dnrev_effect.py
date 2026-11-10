"""25日乖離≤-10%のシグナル：直前の決算で下方修正があったか（株探の見出し、2024年11月〜）で3か月後の成績を比べる"""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
E = pd.read_pickle(SP + "kw/ev_clean.pkl")[["code", "base", "day", "up", "dn", "t"]]
X = pd.read_pickle(SP + "wf_main_events.pkl"); X["code"] = X.t.str[:4]
start = E.day.min() + pd.Timedelta(days=100)
X = X[X.date >= start].copy()
def last_ev(r):
    e = E[(E.code == r.code) & (E.day <= r.date)]
    if not len(e): return pd.Series({"直前決算": "なし", "日数": np.nan, "見出し": ""})
    z = e.sort_values("day").iloc[-1]; k = "下方修正" if z.dn else ("上方修正" if z.up else "修正なし")
    return pd.Series({"直前決算": k, "日数": (r.date - z.day).days, "見出し": z.t})
X = pd.concat([X, X.apply(last_ev, axis=1)], axis=1)
noe = (X["直近4週に決算"] == 0) & (X["2週以内に決算"] == 0); ind = (X["業種内のシグナル銘柄割合%"] >= 30) | (X["業種比20日%"] > -5)
def ep(d):
    s = pd.Series(sorted(d.date.unique())); return int((s.diff().dt.days > 91).sum() + 1) if len(s) else 0
def st(d):
    if len(d) < 10: return f"（{len(d)}件）"
    return (f"{d.up.mean()*100:5.1f}%（{len(d):4d}件・{d.code.nunique():3d}銘柄・{d.date.nunique():2d}週） 平均{d.r3m.mean():+5.1f}% 中央{d.r3m.median():+5.1f}% "
            f"-10%以下{(d.r3m<=-10).mean()*100:3.0f}% 基準との差 勝率{d.wedge.mean():+5.1f}pt 平均{d.edge.mean():+5.1f}pt")
print("対象期間：", X.date.min().date(), "〜", X.date.max().date(), " シグナル", len(X), "件")
for nm, m in [("25日乖離≤-10%のすべて", pd.Series(True, index=X.index)), ("決算の前後でない（直前4週・直後2週なし）", noe), ("決算の前後でない＋業界ぐるみ", noe & ind)]:
    D = X[m]; print("=" * 130); print(nm)
    for k in ["下方修正", "修正なし", "上方修正"]: print(f"   直前の決算が{k:5s}", st(D[D.直前決算 == k]))
    for lab, a, b in [("直前の決算から4週以内", 0, 28), ("4〜13週", 29, 91), ("13週超", 92, 999)]:
        dd = D[D.日数.between(a, b)]
        if len(dd) >= 10: print(f"   {lab}：下方修正", st(dd[dd.直前決算 == "下方修正"]), "\n" + " " * (len(lab) + 4) + "  それ以外", st(dd[dd.直前決算 != "下方修正"]))
D = X[noe & (X.直前決算 == "下方修正")]
print("\n【決算の前後でない×直前の決算が下方修正：事例】")
print(D.groupby(["code"]).agg(件数=("up", "size"), 最初=("date", "min"), 勝率=("up", "mean"), 平均=("r3m", "mean"), 見出し=("見出し", "first")).round(2).to_string())
