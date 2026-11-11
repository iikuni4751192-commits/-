"""新しい要因のうち、2つの戦術で同じ向きに効いたものを、前半（〜2017）で閾値を決め、後半（2018〜）で確かめる"""
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user--/1954a664-8bdc-5ba8-8209-ae58b010ff2a/scratchpad/"
x = pd.concat([pd.read_pickle(SP + "crisis_ext.pkl"), pd.read_pickle(SP + "ext2.pkl")], axis=1).ffill(); x = x.loc[:, ~x.columns.duplicated()]
SETS = {}
X = pd.read_pickle(SP + "free_main.pkl"); SETS["主戦略"] = X[(X["直近4週に決算"] == 0) & (X["2週以内に決算"] == 0)].copy()
Y = pd.read_pickle(SP + "free_cci.pkl"); SETS["CCI戦術"] = Y[(Y["配当利回り%"] >= 3) & (Y["2週以内に決算"] == 0)].copy()
NEW = [("台湾株−日本株の20日差%", "高", "台湾が日本より強い"), ("韓国株−日本株の20日差%", "高", "韓国が日本より強い"), ("^MOVEの20日変化", "高", "米債券の変動率が上昇"),
       ("米30年−10年金利差の20日変化", "高", "米長期金利が短期より上昇"), ("決算発表の集中度(直近14日の発表銘柄の割合%)", "低", "決算シーズンの外")]
at = lambda D, s: s.reindex(pd.to_datetime(D.date), method="ffill").values
for nm, D in SETS.items():
    D["date"] = pd.to_datetime(D.date); D["金+8"] = at(D, x["GC=F"].pct_change(60) * 100) > 8; D["ドル安"] = at(D, x["DX-Y.NYB"].pct_change(60) * 100) <= -2
    print("=" * 130); print(nm, len(D), "件")
    early = D[D.date < "2018"]; flags = []
    for f, side, lab in NEW:
        D[f] = pd.to_numeric(D[f], errors="coerce"); wk = early.groupby("date")[f].first().dropna(); thr = wk.quantile(2/3) if side == "高" else wk.quantile(1/3)
        fl = (D[f] >= thr) if side == "高" else (D[f] <= thr); D["F_" + lab] = fl.fillna(False); flags.append("F_" + lab)
        for per, m in [("前半(閾値を決めた期間)", D.date < "2018"), ("後半(検証)", D.date >= "2018"), ("2023年〜", D.date >= "2023")]:
            a, b = D[m & fl], D[m & ~fl & D[f].notna()]
            wa = a.groupby("date").up.mean().mean() * 100; wb = b.groupby("date").up.mean().mean() * 100
            print(f"  {lab:18s}{per:18s} 当てはまる {a.up.mean()*100:5.1f}%（{len(a):5d}件・{a.date.nunique():3d}週・週平均{wa:5.1f}%・差{a.wedge.mean():+4.1f}pt）／当てはまらない {b.up.mean()*100:5.1f}%（{len(b):5d}件・週平均{wb:5.1f}%・差{b.wedge.mean():+4.1f}pt）")
    D["新しい追い風の数"] = D[flags].sum(axis=1)
    print("  --- 追い風の数（上の5つのうち）---")
    for per, m in [("前半", D.date < "2018"), ("後半", D.date >= "2018")]:
        for k in [0, 1, 2, 3]:
            a = D[m & ((D["新しい追い風の数"] == k) if k < 3 else (D["新しい追い風の数"] >= 3))]
            if len(a) >= 50: print(f"  {per} 追い風{k}{'以上' if k==3 else ''}個: {a.up.mean()*100:5.1f}%（{len(a):5d}件・{a.date.nunique():3d}週・週平均{a.groupby('date').up.mean().mean()*100:5.1f}%・差{a.wedge.mean():+4.1f}pt・-10%以下{(a.r3m<=-10).mean()*100:3.0f}%）")
    print("  --- 既知の追い風（金+8%・ドル安）との関係（後半のみ）---")
    L = D[D.date >= "2018"]
    for lab, m in [("既知の追い風なし", ~L.金+8 if False else ~(L["金+8"] | L["ドル安"])), ("既知の追い風あり", L["金+8"] | L["ドル安"])]:
        for k, kk in [("新しい追い風0〜1個", L["新しい追い風の数"] <= 1), ("新しい追い風2個以上", L["新しい追い風の数"] >= 2)]:
            a = L[m & kk]; 
            if len(a) >= 50: print(f"  {lab} × {k}: {a.up.mean()*100:5.1f}%（{len(a):5d}件・{a.date.nunique():3d}週・週平均{a.groupby('date').up.mean().mean()*100:5.1f}%・差{a.wedge.mean():+4.1f}pt）")
    c = D.groupby("date")[flags + ["金+8", "ドル安"]].first().astype(float).corr().round(2); print("  --- 追い風どうしの相関（週単位）---"); print(c.to_string())
