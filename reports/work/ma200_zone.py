"""市場が200日線を5〜10%下回る局面（ma200_verify.py で両戦術とも低かった区間）の頑健性確認"""
import numpy as np, pandas as pd
exec(open("ma200_verify.py").read().split("for name, B in")[0])
a1 = ((spx.shift(20) < -5) | (mkt.shift(20) < -5)); a2 = ew.rolling(252).apply(lambda v: (len(v) - 1 - np.argmax(v)) / 5, raw=True) > 40; a3 = x["HG=F"].pct_change(20) * 100 <= -8
nw = (a1.astype(int) + a2.astype(int) + a3.astype(int))
for name, B in [("RSI2<10", prep(B2)), ("RSI14<30", prep(B14))]:
    B["nw"] = nw.reindex(B.date).values; B["mchg20"] = (mkt - mkt.shift(20)).reindex(B.date).values
    Z = B[(B.mkt > -10) & (B.mkt <= -5)]
    print("=" * 80); print(name, "市場 -10〜-5% の区間")
    print("  全体", s(Z)); print("  警戒サイン0〜1個", s(Z[Z.nw <= 1])); print("  警戒サイン2個以上", s(Z[Z.nw >= 2]))
    print("  20日前より市場の位置が下がっている（下げの途中）", s(Z[Z.mchg20 < 0])); print("  上がっている（戻りの途中）", s(Z[Z.mchg20 >= 0]))
    # 独立した局面（13週以上空いたら別の局面）
    d = pd.Series(sorted(Z.date.unique())); ep = (d.diff().dt.days > 91).cumsum()
    E = Z.assign(ep=Z.date.map(dict(zip(d, ep))))
    g = E.groupby("ep").agg(開始=("date", "min"), 終了=("date", "max"), 件数=("up", "size"), 勝率=("up", "mean"), 平均=("r3", "mean"))
    g["勝率"] = (g.勝率 * 100).round(0); g["平均"] = g.平均.round(1)
    print(f"  独立した局面 {len(g)}個、局面ごとの勝率が50%未満の局面 {(g.勝率 < 50).sum()}個"); print(g.to_string())
    # 比較：他の区間の局面数
    for lab, m in [("市場 -10%未満", B.mkt <= -10), ("市場 -5〜0%", (B.mkt > -5) & (B.mkt <= 0)), ("市場 0%超", B.mkt > 0)]:
        q = B[m]; dd = pd.Series(sorted(q.date.unique())); n_ep = (dd.diff().dt.days > 91).sum() + 1
        print(f"  参考 {lab}: {s(q)}  局面{n_ep}個")
