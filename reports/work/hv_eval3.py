import numpy as np, pandas as pd, pickle
exec(open("hv_eval.py").read().split("pd.set_option")[0])
cond = pickle.load(open(SP + "hv_cond.pkl", "rb")); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 90)
dts = np.array(sorted(D.date.unique()))
def episodes(m):
    d = np.array(sorted(D.date[m].unique())); 
    if len(d) == 0: return 0
    n = 1; last = d[0]
    for x_ in d[1:]:
        if (x_ - last) / np.timedelta64(1, "D") > 91: n += 1
        last = x_
    return n
Q = lambda k: cond[k]
DY = Q("配当利回り3%以上"); PULL = Q("25日乖離% 低(下位1/3)"); CALM = Q("1日最大下落%(20日) 低(下位1/3)"); NOEARN = Q("決算の前後なし(直近4週・次2週とも無し)"); LOWCORR = Q("市場との相関(60日) 低(下位1/3)")
VIX = Q("【市場】VIX 高(上位1/3)"); CRASH = Q("【市場】急落銘柄の割合% 高(上位1/3)"); MD = Q("【市場】日本株の250日高値からの下落% 低(下位1/3)")
MARK = VIX | CRASH
tests = {"高ボラ全体(基準)": np.ones(len(D), bool), "配当3%以上": DY, "売られ(25日乖離下位1/3)": PULL, "直近の最大下落が小さい": CALM, "決算前後なし": NOEARN,
 "市場:VIX高 or 急落銘柄割合高": MARK, "市場:VIX低 かつ 急落割合低(平穏)": ~VIX & ~CRASH,
 "配当3%以上＋売られ": DY & PULL, "配当3%以上＋売られ＋決算前後なし": DY & PULL & NOEARN, "配当3%以上＋売られ＋最大下落小": DY & PULL & CALM,
 "配当3%以上＋市場(VIX高 or 急落割合高)": DY & MARK, "配当3%以上＋売られ＋市場(VIX高 or 急落割合高)": DY & PULL & MARK, "配当3%以上＋売られ＋決算前後なし＋市場(VIX高 or 急落割合高)": DY & PULL & NOEARN & MARK,
 "売られ＋市場(VIX高 or 急落割合高)": PULL & MARK, "売られ＋最大下落小＋市場(VIX高 or 急落割合高)": PULL & CALM & MARK, "配当3%以上＋売られ＋平穏相場": DY & PULL & ~VIX & ~CRASH}
rows = []
for k, m in tests.items():
    for h in (63, 126):
        r = D[f"r{h}"].to_numpy(); ok = m & ~np.isnan(r)
        if ok.sum() < 100: rows.append(dict(条件=k, 期間=h, n=int(ok.sum()))); continue
        a = ok & first; b = ok & ~first
        rows.append(dict(条件=k, 期間=h, n=int(ok.sum()), 週数=D.date[ok].nunique(), 独立局面=episodes(ok), 平均=r[ok].mean(), 勝率=(r[ok] > 0).mean() * 100, 中央値=np.median(r[ok]), 前半平均=r[a].mean() if a.sum() > 50 else np.nan, 前半勝率=(r[a] > 0).mean() * 100 if a.sum() > 50 else np.nan, 後半平均=r[b].mean(), 後半勝率=(r[b] > 0).mean() * 100, 下位5pct=np.percentile(r[ok], 5)))
R = pd.DataFrame(rows)
for h in (63, 126): print(f"\n=== {h}日 ==="); print(R[R.期間 == h].drop(columns="期間").round(1).to_string(index=False))
