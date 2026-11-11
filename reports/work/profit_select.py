"""銘柄選び：どの要因が『勝率』ではなく『平均リターン（期待値）』を分けるか（同じ週の中の順位相関）"""
import sys, numpy as np, pandas as pd
from profit_lab import *
X, SETS = events(); X = X[(X.e + 189 + 2 < len(idx))].copy(); X["nw"] = [NW[pos[d]] for d in X.date]; X = X[X.nw <= 1].copy()
S = X[SETS["C 手順2まで(決算なし)"].reindex(X.index).fillna(False)].copy()
for h in (63, 126):
    S[f"r{h}"] = [run(e, j, hold=h)[0] for e, j in zip(S.e.values, S.j.values)]
ST = [c for c in S.attrs.get("stock", [])] if False else None
base_cols = ["配当利回り%", "25日乖離%", "200日乖離%", "RSI14", "5日騰落%", "20日騰落%", "60日騰落%", "1年騰落%(直近1か月除く)", "52週高値比%", "最大1日下落%(20日)", "ボラ(60日,年率%)", "出来高比(5日/60日)", "規模(売買代金,log)",
            "業種20日騰落%", "業種比20日%", "業種内のシグナル銘柄割合%", "業種比1年%(直近1か月除く)", "決算反応%(発表前日→2日後)", "連続週(何週目)", "シグナル数(上位12)", "sec1y"]
new_cols = ["配当利回りの対3年中央値比", "配当利回りの業種中央値比", "20日高値から何日前か", "20日中の下落日数", "出来高の枯れ(5日/60日)", "値動きの大きさ(20日÷250日)", "5日レンジ内の位置(0=安値,1=高値)", "5日安値からの反発%", "下落日の対市場の強さ(250日平均%)", "市場との相関(60日)", "次の権利落ちまでの営業日", "次の決算までの日数"]
feats = [c for c in base_cols + new_cols if c in S.columns]
for f in feats: S[f] = pd.to_numeric(S[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
def blk(w): b = w.groupby((w.index - pd.Timestamp("2000-01-07")).days // 91).mean(); return b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))
rows = []
for f in feats:
    d = S.dropna(subset=[f, "r126"]); r = {"要因": f, "件数": len(d)}
    for tgt in ("r63", "r126"):
        ic = pd.Series({w: g[f].rank().corr(g[tgt].rank()) for w, g in d.groupby("date") if len(g) >= 8 and g[f].nunique() > 2}, dtype=float).dropna(); ic.index = pd.to_datetime(ic.index)
        r[f"{tgt}:順位相関"] = ic.mean(); r[f"{tgt}:t"] = blk(ic); r[f"{tgt}:前半t"] = blk(ic[ic.index < "2018"]); r[f"{tgt}:後半t"] = blk(ic[ic.index >= "2018"])
    q = d.groupby("date")[f].rank(pct=True); n = d.groupby("date")[f].transform("size"); m = n >= 8
    r["同じ週の上位1/3の126日平均"] = d[m & (q > 2/3)].r126.mean(); r["下位1/3"] = d[m & (q <= 1/3)].r126.mean(); r["差"] = r["同じ週の上位1/3の126日平均"] - r["下位1/3"]
    rows.append(r)
R = pd.DataFrame(rows).sort_values("r126:t", key=abs, ascending=False)
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200)
print(f"手順2まで・警戒サイン0〜1個の事例 {len(S)}件（63日平均{S.r63.mean():.1f}%、126日平均{S.r126.mean():.1f}%）。目安：|t|>3.2、前半・後半で同じ向き"); print(R.round(2).to_string(index=False))
# 配当の段階別の平均リターン
print("\n[配当利回り別の平均リターン（%）]")
for lab, a, b in [("2.6%未満", 0, 2.6), ("2.6〜3.5%", 2.6, 3.5), ("3.5〜4.5%", 3.5, 4.5), ("4.5〜6%", 4.5, 6), ("6%以上", 6, 99)]:
    d = S[(S["配当利回り%"] >= a) & (S["配当利回り%"] < b)]
    print(f"  {lab:10s} 件数{len(d):5d} 63日 平均{d.r63.mean():+6.1f}% 中央{d.r63.median():+6.1f}% 勝率{(d.r63>0).mean()*100:4.1f}% 下位5%{d.r63.quantile(.05):+6.1f} ｜126日 平均{d.r126.mean():+6.1f}% 中央{d.r126.median():+6.1f}% 勝率{(d.r126>0).mean()*100:4.1f}% ｜前半126日{d[d.date<'2018'].r126.mean():+6.1f}% 後半{d[d.date>='2018'].r126.mean():+6.1f}%")
print("\n[同じ週の中で上位k銘柄だけ買った場合（126日、平均リターン%・件数）]")
for f, asc in [("配当利回り%", False), ("25日乖離%", True), ("ボラ(60日,年率%)", False), ("業種内のシグナル銘柄割合%", False), ("52週高値比%", True), ("規模(売買代金,log)", True)]:
    d = S.dropna(subset=[f, "r126"]).copy(); d["rk"] = d.groupby("date")[f].rank(ascending=asc, method="first")
    print(f"  {f:22s} " + " ".join(f"上位{k}:{d[d.rk<=k].r126.mean():+5.1f}%({(d.rk<=k).sum()})" for k in (1, 2, 3, 5, 10)) + f" ／全体{d.r126.mean():+5.1f}%")
S.to_pickle(SP + "profit_select_events.pkl")
