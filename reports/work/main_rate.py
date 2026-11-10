"""主戦略（25日乖離≤-10%）：米国10年金利の20日変化が成績に与える影響"""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
X = pd.read_pickle(SP + "wf_main_events.pkl")
X["rt"] = pd.to_numeric(X["米10年金利20日変化"], errors="coerce")
noe = (X["直近4週に決算"] == 0) & (X["2週以内に決算"] == 0); ind = (X["業種内のシグナル銘柄割合%"] >= 30) | (X["業種比20日%"] > -5)
hot = (X.sec1y > 60) | (X.r1y > 100); dv = X["配当利回り%"]
SETS = {"最終ルール（決算なし・業界ぐるみ・配当2.6%以上・上がりすぎ除外）": noe & ind & (dv >= 2.6) & ~hot,
        "決算なし＋業界ぐるみ（手順3まで）": noe & ind, "25日乖離≤-10%のすべて": pd.Series(True, index=X.index)}
def ep(d):
    s = pd.Series(sorted(d.date.unique())); return int((s.diff().dt.days > 91).sum() + 1) if len(s) else 0
def st(d):
    if len(d) < 30: return f"（{len(d)}件）"
    a, b = d[d.date < "2018"], d[d.date >= "2018"]; c = d[d.date >= "2023"]
    f = lambda q: f"{q.up.mean()*100:5.1f}%" if len(q) >= 20 else "   - "
    return (f"{d.up.mean()*100:5.1f}%（{len(d):5d}件・{ep(d):3d}局面） 平均{d.r3m.mean():+5.1f}% -10%以下{(d.r3m<=-10).mean()*100:3.0f}% "
            f"基準との差 勝率{d.wedge.mean():+5.1f}pt 平均{d.edge.mean():+5.1f}pt ｜前半{f(a)} 後半{f(b)} 2023年〜{f(c)}")
BK = [("大きく低下（≤-0.27）", lambda r: r <= -0.27), ("低下（-0.27〜-0.13）", lambda r: (r > -0.27) & (r <= -0.13)),
      ("横ばい（-0.13〜+0.06）", lambda r: (r > -0.13) & (r <= 0.06)), ("上昇（+0.06〜+0.20）", lambda r: (r > 0.06) & (r <= 0.20)),
      ("大きく上昇（>+0.20）", lambda r: r > 0.20)]
# 警戒サイン
x = pd.read_pickle(SP + "crisis_ext_1107.pkl").ffill()
for nm, m in SETS.items():
    D = X[m].dropna(subset=["rt"])
    print("=" * 140); print(nm); print("   全体                  ", st(D))
    for lab, f in BK: print(f"   {lab:22s}", st(D[f(D.rt)]))
    print("   ― 2区分 ―")
    print(f"   {'低下（≤-0.13）':22s}", st(D[D.rt <= -0.13])); print(f"   {'上昇（>+0.06）':22s}", st(D[D.rt > 0.06]))
    print(f"   {'上昇（>-0.03、RSI14戦術の基準）':22s}", st(D[D.rt > -0.03]))
