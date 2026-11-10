"""cci_combo.py で残った候補を局面単位で確かめる（事例は同じ週に集中するため）＋現在の状態"""
import numpy as np, pandas as pd
src = open("cci_combo.py").read().split("C = {k: np.nan_to_num")[0]; exec(src)
C = {k: np.nan_to_num(np.asarray(v, dtype=float)).astype(bool) for k, v in C.items()}
CAND = [("基準（必須条件のみ）", []), ("参考：円高(60日) × 米金利60日低下", ["円高(60日≤-1.5%)", "米金利60日低下(≤-0.16)"]),
        ("金が60日で+8%超", ["金が60日で+8%超"]), ("金+8% × 原油-15%", ["金が60日で+8%超", "原油が60日で-15%以下"]),
        ("金+8% × 配当5%以上", ["金が60日で+8%超", "配当5%以上"]), ("金+8% × 200日線より上の銘柄が5割未満", ["金が60日で+8%超", "200日線より上の銘柄が5割未満"]),
        ("金+8% × 25日乖離≤-10% × 直前4週に決算なし", ["金が60日で+8%超", "25日乖離≤-10%", "直前4週に決算なし"]),
        ("円高(20日) × 銅が20日で+3%超", ["円高(20日≤-1.6%)", "銅が20日で+3%超"]), ("円高(20日) × 銅+3% × 直前4週に決算なし", ["円高(20日≤-1.6%)", "銅が20日で+3%超", "直前4週に決算なし"]),
        ("ドル指数-2% × 日本株20日-3.7%以下", ["ドル指数が60日で-2%以下", "日本株が20日で-3.7%以下"]), ("ドル指数-2% × 配当4%以上", ["ドル指数が60日で-2%以下", "配当4%以上"]),
        ("VIX22超で低下 × 日本株20日-3.7%以下", ["VIX22超で5日前より低下", "日本株が20日で-3.7%以下"]),
        ("全体の43%以上にシグナル × 大型株優位", ["全体の43%以上にシグナル", "大型株優位(日経÷均等 20日>+0.4%)"]), ("銅が20日で+3%超 × 配当5%以上", ["銅が20日で+3%超", "配当5%以上"])]
rows = []
for lab, ks in CAND:
    m = np.ones(len(B), bool)
    for k in ks: m &= C[k]
    D = B[m]; d = pd.Series(sorted(D.date.unique())); ep = (d.diff().dt.days > 91).cumsum(); D = D.assign(ep=D.date.map(dict(zip(d, ep))))
    E = D.groupby("ep").agg(w=("up", "mean"), r=("r3m", "mean"), n=("up", "size"), s=("date", "min")); Wk = D.groupby("date").agg(w=("up", "mean"), bw=("b_win", "first"))
    worst = E.sort_values("r").head(2)
    rows.append({"組み合わせ": lab, "件数": len(D), "週": len(Wk), "局面": len(E), "事例の勝率": D.up.mean() * 100, "週ごとの平均勝率": Wk.w.mean() * 100, "同じ週の全銘柄": Wk.bw.mean(),
                 "勝ち越した局面": f"{int((E.w > 0.5).sum())}/{len(E)}", "局面の平均リターンの中央値": E.r.median(), "最悪の局面": " / ".join(f"{s.date()}:{r:+.0f}%({n})" for s, r, n in zip(worst.s, worst.r, worst.n)),
                 "2008年を除く": D[D.date.dt.year != 2008].up.mean() * 100, "2020年を除く": D[D.date.dt.year != 2020].up.mean() * 100})
pd.set_option("display.width", 400); pd.set_option("display.max_colwidth", 60)
print(pd.DataFrame(rows).round(1).to_string(index=False))
# 現在
t = x.index[-1]; f = lambda k, n: (x[k].iloc[-1] / x[k].dropna().iloc[-1 - n] - 1) * 100
print(f"\n現在（{t.date()}）: 金60日 {x['GC=F'].pct_change(60).iloc[-1]*100:+.1f}%  原油60日 {x['CL=F'].pct_change(60).iloc[-1]*100:+.1f}%  銅20日 {x['HG=F'].pct_change(20).iloc[-1]*100:+.1f}%  ドル指数60日 {x['DX-Y.NYB'].pct_change(60).iloc[-1]*100:+.1f}%"
      f"  ドル円20日 {x['JPY=X'].pct_change(20).iloc[-1]*100:+.1f}% 60日 {x['JPY=X'].pct_change(60).iloc[-1]*100:+.1f}%  VIX {x['^VIX'].iloc[-1]:.1f}（5日前比 {x['^VIX'].diff(5).iloc[-1]:+.1f}）  米金利60日 {x['^TNX'].diff(60).iloc[-1]:+.2f}")
