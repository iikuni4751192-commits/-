"""新しい要因が、将来（検証に使っていない年）の勝率の予想に役立つかを、毎年学習し直す方法（ウォークフォワード）で確かめる
・特徴量セット A:配当のみ  B:既知の要因  C:既知＋新しい要因
・評価は CCI戦術の必須条件（配当3%以上・直後2週に決算なし）の事例で、年ごとの予測精度と、予測上位の勝率"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
X = pd.read_pickle(SP + "free_cci.pkl"); ST, MK, CAL = X.attrs["stock"], X.attrs["market"], X.attrs["cal"]
x = pd.concat([pd.read_pickle(SP + "crisis_ext.pkl"), pd.read_pickle(SP + "ext2.pkl")], axis=1).ffill(); x = x.loc[:, ~x.columns.duplicated()]
at = lambda s: s.reindex(pd.to_datetime(X.date), method="ffill").values
X["金60"] = at(x["GC=F"].pct_change(60) * 100); X["ドル指数60"] = at(x["DX-Y.NYB"].pct_change(60) * 100); X["原油60"] = at(x["CL=F"].pct_change(60) * 100); X["銅20"] = at(x["HG=F"].pct_change(20) * 100)
KNOWN_STOCK = ["配当利回り%", "25日乖離%", "200日乖離%", "RSI14", "5日騰落%", "20日騰落%", "60日騰落%", "1年騰落%(直近1か月除く)", "52週高値比%", "最大1日下落%(20日)", "ボラ(60日,年率%)", "出来高比(5日/60日)", "規模(売買代金,log)",
               "業種20日騰落%", "業種比20日%", "業種内のシグナル銘柄割合%", "業種比1年%(直近1か月除く)", "直近4週に決算", "決算反応%(発表前日→2日後)"]
KNOWN_MK = ["市場20日騰落%", "市場200日乖離%", "全体のシグナル銘柄割合%", "VIX", "米10年金利20日変化", "ドル円20日%", "米信用(HYG/IEF)20日%", "ドル円の60日騰落%", "米10年金利の60日変化", "金60", "ドル指数60", "原油60", "銅20"]
SETS = {"A 配当のみ": ["配当利回り%"], "B 既知の要因": KNOWN_STOCK + KNOWN_MK, "C 既知＋新しい要因": KNOWN_STOCK + KNOWN_MK + ST + MK + CAL}
for f in sum(SETS.values(), []): X[f] = pd.to_numeric(X[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
X["date"] = pd.to_datetime(X.date); y = X.up.astype(int).values
evalm = ((X["配当利回り%"] >= 3) & (X["2週以内に決算"] == 0)).values
rows = []; pred = {k: np.full(len(X), np.nan) for k in SETS}
for Y in range(2013, 2027):
    cut = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=100); tr = (X.date < cut).values; te = (X.date.dt.year == Y).values
    if te.sum() == 0: continue
    for nm, fs in SETS.items():
        mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120, min_samples_leaf=300, l2_regularization=5.0, random_state=0)
        use = [f for f in fs if X.loc[tr, f].nunique() >= 2]
        mdl.fit(X.loc[tr, use].values, y[tr]); pred[nm][te] = mdl.predict_proba(X.loc[te, use].values)[:, 1]
X["年"] = X.date.dt.year; res = []
for nm in SETS:
    X["p"] = pred[nm]
    for Y in range(2013, 2027):
        d = X[evalm & (X.年 == Y)]
        if len(d) < 200 or d.up.nunique() < 2: continue
        thr = d.p.quantile(0.7); wk = d.groupby("date"); rk = wk.p.rank(pct=True); n = wk.p.transform("size"); m = n >= 6
        res.append(dict(セット=nm, 年=Y, 件数=len(d), AUC=roc_auc_score(d.up, d.p), 基準の勝率=d.up.mean() * 100, 予測上位3割の勝率=d[d.p >= thr].up.mean() * 100, 予測下位3割の勝率=d[d.p <= d.p.quantile(0.3)].up.mean() * 100,
                        同じ週で上位1_3=d[m & (rk > 2/3)].up.mean() * 100, 同じ週で下位1_3=d[m & (rk <= 1/3)].up.mean() * 100))
R = pd.DataFrame(res); pd.set_option("display.width", 300)
print("【年ごとの AUC（0.5=当て推量）】"); print(R.pivot(index="年", columns="セット", values="AUC").round(3).to_string())
print("\n【予測上位3割の勝率（基準＝その年の評価対象の勝率）】"); P = R.pivot(index="年", columns="セット", values="予測上位3割の勝率"); P["基準"] = R.groupby("年").基準の勝率.first(); print(P.round(1).to_string())
g = R.groupby("セット").agg(AUC平均=("AUC", "mean"), 上位3割の勝率=("予測上位3割の勝率", "mean"), 下位3割の勝率=("予測下位3割の勝率", "mean"), 同じ週の上位1_3=("同じ週で上位1_3", "mean"), 同じ週の下位1_3=("同じ週で下位1_3", "mean"), 基準=("基準の勝率", "mean"))
print("\n【全体（年の平均）】"); print(g.round(3).to_string())
print("AUCがAを上回った年:", {nm: int((R[R.セット == nm].set_index("年").AUC > R[R.セット == "A 配当のみ"].set_index("年").AUC).sum()) for nm in SETS if nm != "A 配当のみ"}, "／", R.年.nunique(), "年")
print("CがBを上回った年(AUC):", int((R[R.セット.str.startswith("C")].set_index("年").AUC > R[R.セット.str.startswith("B")].set_index("年").AUC).sum()), "／", R.年.nunique())
# 重要度（Cの最終年モデルの順列重要度は重いので、単純にCの予測と各特徴の順位相関）
X["pC"] = pred["C 既知＋新しい要因"]; d = X[evalm & X.pC.notna()]
cor = {f: d[[f, "pC"]].corr(method="spearman").iloc[0, 1] for f in ST + MK + CAL + KNOWN_MK + KNOWN_STOCK if f in d}
print("\n【Cの予測と最も連動した要因（順位相関）】"); print(pd.Series(cor).sort_values(key=abs, ascending=False).head(14).round(2).to_string())
