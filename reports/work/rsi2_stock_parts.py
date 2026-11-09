"""指定銘柄：RSI2<10 が出た週を、戦術に組み込まれた各条件（1つずつ）で分けた3か月後の勝率（必須条件を満たさない事例も含む）"""
import sys, numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
codes = sys.argv[1:]
X = pd.read_pickle(SP + "rsi2_extra_events.pkl")
for c in X.columns:
    if X[c].dtype == object and c not in ("t", "sector"): X[c] = pd.to_numeric(X[c], errors="coerce")
conds = {
    "RSI2<10 のすべて": lambda D: pd.Series(True, index=D.index),
    "配当2.45%以上": lambda D: D["配当利回り%"] >= 2.45,
    "直近4週に決算なし": lambda D: D["直近4週に決算"] == 0,
    "米金利が20日で0.13以上低下": lambda D: D["米10年金利20日変化"] <= -0.13,
    "VIX 22超": lambda D: D.VIX > 22,
    "25日乖離≤-10%も同時": lambda D: D["25日乖離%"] <= -10,
    "業界ぐるみ（業種内50%以上）": lambda D: D["業種内のシグナル銘柄割合%"] >= 50,
    "市場全体の急落（全体36%以上）": lambda D: D["全体のシグナル銘柄割合%"] >= 36,
    "警戒：米金利が0.06超上昇": lambda D: D["米10年金利20日変化"] > 0.06,
}
now = {"4114": [1,1,1,1,1,0,1,0,0], "4631": [1,1,1,1,1,0,1,0,0], "9364": [1,1,1,1,1,0,1,0,0], "5021": [1,1,1,1,1,1,1,0,0],
       "1893": [1,1,1,1,1,0,0,0,0], "6481": [1,1,1,1,1,1,1,0,0], "4183": [1,1,1,1,1,1,1,0,0]}
def cell(D):
    if len(D) == 0: return "-"
    d = pd.Series(sorted(D.date.unique())); ep = int((d.diff().dt.days > 91).sum() + 1)
    return f"{D.up.mean()*100:3.0f}%({len(D)}件/{ep}局面)"
rows = {}
for c in codes:
    D = X[X.t == c + ".T"]
    col = {}
    for k, (lab, f) in enumerate(conds.items()):
        col[("○" if now[c][k] else "　") + lab] = cell(D[f(D).fillna(False).astype(bool)])
    # 今と同じ条件がすべてそろった事例（必須条件の決算は除く）
    m = pd.Series(True, index=D.index)
    for k, (lab, f) in enumerate(conds.items()):
        if now[c][k] and lab not in ("RSI2<10 のすべて",): m &= f(D).fillna(False).astype(bool)
    col["今と同じ条件がすべてそろった"] = cell(D[m])
    rows[c] = col
T = pd.DataFrame(rows)
pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 40)
print(T.to_string())
# 全銘柄での同じ条件（参考）
print("\n参考：全銘柄（RSI2<10）で同じ条件")
for lab, f in conds.items():
    m = f(X).fillna(False).astype(bool); print(f"  {lab:24s} {X[m].up.mean()*100:5.1f}%（{m.sum()}件）")
