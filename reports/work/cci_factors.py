"""CCI20<-100：3か月後の上昇確率を上げる／下げる条件（前半・後半、同じ週の全銘柄との差つき）"""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
X = pd.read_pickle(SP + "CCI20lt-100_events.pkl"); base = pd.read_pickle(SP + "wf_base_by_date.pkl")
X["b_win"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[1]); X["b_mean"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[0])
X["wedge"] = X.up * 100 - X.b_win; X["edge"] = X.r3m - X.b_mean
x = pd.read_pickle(SP + "crisis_ext.pkl").ffill()
gs = (x["^GSPC"] / x["^GSPC"].rolling(200).mean() - 1) * 100; cu = x["HG=F"].pct_change(20) * 100
X["spx20"] = gs.shift(20).reindex(X.date, method="ffill").values; X["cu"] = cu.reindex(X.date, method="ffill").values
X["mk200"] = X["市場200日乖離%"]
def st(d):
    a, b, c = d[d.date < "2018"], d[d.date >= "2018"], d[d.date >= "2023"]
    return (f"{d.up.mean()*100:5.1f}%（{len(d):6d}件） 平均{d.r3m.mean():+5.1f}% -10%以下{(d.r3m<=-10).mean()*100:3.0f}% 差 勝率{d.wedge.mean():+5.1f}pt 平均{d.edge.mean():+4.1f}pt"
            f"｜前半{a.up.mean()*100:5.1f}% 後半{b.up.mean()*100:5.1f}% 2023年〜{c.up.mean()*100:5.1f}%")
dv = X["配当利回り%"]; rt = X["米10年金利20日変化"]; n4 = X["直近4週に決算"] == 0; n2 = X["2週以内に決算"] == 0
C = [("すべて", pd.Series(True, index=X.index)),
     ("配当 1.45%未満", dv < 1.45), ("配当 2.45〜3%", (dv >= 2.45) & (dv < 3)), ("配当 3〜4%", (dv >= 3) & (dv < 4)), ("配当 4%以上", dv >= 4),
     ("直後2週に決算あり", ~n2), ("直前4週に決算あり", ~n4), ("決算なし（前後とも）", n4 & n2),
     ("米金利20日 ≤-0.13", rt <= -0.13), ("米金利20日 >+0.06", rt > 0.06),
     ("VIX 22超", X.VIX > 22), ("VIX 16未満", X.VIX < 16),
     ("25日乖離≤-10%も同時", X["25日乖離%"] <= -10), ("全体の43%以上にシグナル", X["全体のシグナル銘柄割合%"] >= 43.3),
     ("市場が200日線を5%超下回る", X.mk200 < -5), ("市場が200日線を5%超上回る", X.mk200 > 5),
     ("S&P500が20日前に200日線-5%未満（弱気相場）", X.spx20 < -5), ("銅20日 ≤-8%", X.cu <= -8),
     ("円高（ドル円20日 ≤-1.6%）", X["ドル円20日%"] <= -1.62), ("円安（ドル円20日 >+0.6%）", X["ドル円20日%"] > 0.61),
     ("★ 配当3%以上＋決算なし", (dv >= 3) & n4 & n2), ("★ 配当4%以上＋決算なし", (dv >= 4) & n4 & n2),
     ("★ 配当4%以上＋決算なし＋米金利低下", (dv >= 4) & n4 & n2 & (rt <= -0.13)),
     ("▼ 配当1.45%未満＋直後2週に決算", (dv < 1.45) & ~n2), ("▼ 配当1.45%未満＋米金利上昇", (dv < 1.45) & (rt > 0.06)),
     ("▼ 弱気相場＋銅急落", (X.spx20 < -5) & (X.cu <= -8))]
for lab, m in C: print(f"{lab:26s}", st(X[m.fillna(False)]))
