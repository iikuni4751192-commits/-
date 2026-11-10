"""3つの戦術：直前4週・直後2週の決算が3か月後の成績に与える影響（決算以外の条件はそろえる）"""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
base = pd.read_pickle(SP + "wf_base_by_date.pkl")
def load(fn):
    X = pd.read_pickle(SP + fn)
    for c in X.columns:
        if X[c].dtype == object and c not in ("t", "sector"): X[c] = pd.to_numeric(X[c], errors="coerce")
    if "b_win" not in X:
        X["b_win"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[1]); X["b_mean"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[0])
        X["wedge"] = X.up * 100 - X.b_win; X["edge"] = X.r3m - X.b_mean
    return X.dropna(subset=["r3m"])
def st(d):
    if len(d) < 20: return f"（{len(d)}件）"
    a, b, c = d[d.date < "2018"], d[d.date >= "2018"], d[d.date >= "2023"]
    f = lambda q: f"{q.up.mean()*100:5.1f}%" if len(q) >= 20 else "   - "
    return (f"{d.up.mean()*100:5.1f}%（{len(d):5d}件） 平均{d.r3m.mean():+5.1f}% -10%以下{(d.r3m<=-10).mean()*100:3.0f}% "
            f"基準との差 勝率{d.wedge.mean():+5.1f}pt 平均{d.edge.mean():+5.1f}pt ｜前半{f(a)} 後半{f(b)} 2023年〜{f(c)}")
for name, fn, cond in [
    ("主戦略（25日乖離≤-10%）：業界ぐるみ・配当2.6%以上・上がりすぎ除外をそろえる", "wf_main_events.pkl",
     lambda X: ((X["業種内のシグナル銘柄割合%"] >= 30) | (X["業種比20日%"] > -5)) & (X["配当利回り%"] >= 2.6) & ~((X.sec1y > 60) | (X.r1y > 100))),
    ("RSI14<30 戦術：業界ぐるみ・配当2.45%以上をそろえる", "RSI14lt30_events.pkl",
     lambda X: ((X["業種内のシグナル銘柄割合%"] >= 30) | (X["業種比20日%"] > -5)) & (X["配当利回り%"] >= 2.45)),
    ("RSI2<10 戦術：配当2.45%以上をそろえる", "rsi2_extra_events.pkl", lambda X: X["配当利回り%"] >= 2.45)]:
    X = load(fn); B = X[cond(X).fillna(False)]
    pe = B["直近4週に決算"] == 1; fe = B["2週以内に決算"] == 1
    print("=" * 150); print(name)
    for lab, m in [("決算なし（前後とも）", ~pe & ~fe), ("直前4週に決算あり", pe & ~fe), ("直後2週に決算あり", ~pe & fe), ("前後とも決算あり", pe & fe)]:
        print(f"   {lab:16s}", st(B[m]))
