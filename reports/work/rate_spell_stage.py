"""金利低下／上昇シグナルの『序盤』と『終盤』で買った場合の3か月成績（RSI2・RSI14戦術の必須条件の事例）
序盤／終盤の区別：
  (a) その時点で分かる：連続何週目か、20日変化が前週より強まったか弱まったか
  (b) 後から分かる（参考）：その連続期間の最後の週か（将来の情報を使うので実際には使えない）"""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
x = pd.read_pickle(SP + "crisis_ext.pkl")["^TNX"].dropna(); ch = x.diff(20)
wk = ch.resample("W-FRI").last().dropna(); wk = wk[wk.index >= "2007-10-01"]
base = pd.read_pickle(SP + "wf_base_by_date.pkl")
def spells(flag):
    run = (flag != flag.shift()).cumsum(); n = flag.groupby(run).cumcount() + 1; L = flag.groupby(run).transform("size")
    return n.where(flag), L.where(flag)
def load(fn):
    X = pd.read_pickle(SP + fn)
    for c in X.columns:
        if X[c].dtype == object and c not in ("t", "sector"): X[c] = pd.to_numeric(X[c], errors="coerce")
    X["b_win"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[1]); X["b_mean"] = X.date.map(lambda d: base.get(pd.Timestamp(d), (np.nan, np.nan))[0])
    X = X.dropna(subset=["b_win", "r3m"]); X["wedge"] = X.up * 100 - X.b_win; X["edge"] = X.r3m - X.b_mean
    return X
def blk(d):
    w = d.groupby("date").edge.mean(); b = w.groupby((w.index - pd.Timestamp("2000-01-07")).days // 91).mean()
    return b.mean() / (b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 2 else np.nan
def s(d):
    if len(d) < 30: return f"（{len(d)}件：少なすぎ）"
    a = d[d.date < "2018"]; b = d[d.date >= "2018"]
    f = lambda q: f"{q.up.mean()*100:4.1f}%" if len(q) >= 20 else "  -  "
    return (f"{d.up.mean()*100:5.1f}%（{len(d):5d}件、{d.date.nunique():3d}週） 平均{d.r3m.mean():+5.1f}% 基準との差：勝率{d.wedge.mean():+5.1f}pt 平均{d.edge.mean():+5.1f}pt"
            f"  前半{f(a)} 後半{f(b)}")
for name, fn, dn_th, up_th in [("RSI2<10 戦術（配当2.45%以上・直前4週に決算なし）", "rsi2_extra_events.pkl", -0.13, 0.06),
                               ("RSI14<30 戦術（必須条件）", "RSI14lt30_events.pkl", -0.27, -0.03)]:
    X = load(fn); dv = X["配当利回り%"]; n4 = X["直近4週に決算"] == 0; n6 = n4 & (X["2週以内に決算"] == 0)
    ind = (X["業種内のシグナル銘柄割合%"] >= 30) | (X["業種比20日%"] > -5)
    B = X[n4 & (dv >= 2.45)].copy() if "RSI2" in name else X[n6 & ind & (dv >= 2.45)].copy()
    dflag = wk <= dn_th; uflag = wk > up_th
    dn_n, dn_L = spells(dflag); up_n, up_L = spells(uflag)
    prev = wk.shift(1)
    B["chg"] = wk.reindex(B.date).values; B["prev"] = prev.reindex(B.date).values
    B["dn_n"] = dn_n.reindex(B.date).values; B["dn_L"] = dn_L.reindex(B.date).values
    B["up_n"] = up_n.reindex(B.date).values; B["up_L"] = up_L.reindex(B.date).values
    print("=" * 120); print(name, f" 低下＝20日変化≤{dn_th}、上昇＝20日変化>{up_th}"); print("   全体          ", s(B))
    for kind, nn, LL, th in [("低下", "dn_n", "dn_L", dn_th), ("上昇", "up_n", "up_L", up_th)]:
        D = B[B[nn].notna()]
        print(f"\n [{kind}シグナルが出ている週]                 ", s(D))
        print(f"   (a) 連続何週目（その時点で分かる）")
        for lab, m in [("1週目", D[nn] == 1), ("2週目", D[nn] == 2), ("3〜4週目", D[nn].between(3, 4)), ("5週目以降", D[nn] >= 5)]:
            print(f"      {lab:8s}", s(D[m]))
        if kind == "低下":
            acc = D.chg < D.prev     # 前週より下げ幅が拡大
        else:
            acc = D.chg > D.prev     # 前週より上げ幅が拡大
        print(f"   (a) 20日変化の勢い（その時点で分かる）")
        print(f"      {kind}が強まっている", s(D[acc])); print(f"      {kind}が弱まり始めた", s(D[~acc]))
        print(f"   (b) 後から見た位置（参考：将来の情報を使う）")
        last = D[nn] == D[LL]; first = D[nn] == 1
        print(f"      1週だけで終わった", s(D[first & last])); print(f"      続いた期間の最初の週", s(D[first & ~last]))
        print(f"      途中の週          ", s(D[~first & ~last])); print(f"      続いた期間の最後の週", s(D[~first & last]))
    # 直後：シグナルが終わった直後の週
    N = B[B.dn_n.isna() & pd.Series(dflag.shift(1).reindex(B.date).values, index=B.index).fillna(False).astype(bool)]
    print(f"\n [低下シグナルが終わった翌週]              ", s(N))
    N2 = B[B.up_n.isna() & pd.Series(uflag.shift(1).reindex(B.date).values, index=B.index).fillna(False).astype(bool)]
    print(f" [上昇シグナルが終わった翌週]              ", s(N2))
    Z = B[B.dn_n.isna() & B.up_n.isna()]; print(f" [どちらでもない週]                        ", s(Z))
