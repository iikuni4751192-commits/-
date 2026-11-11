import sys, numpy as np, pandas as pd
from profit_lab import *
X, SETS = events()
Hmax = 189
X = X[(X.e + Hmax + 2 < len(idx))].copy()
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
RULES = {
 "固定 21日": dict(hold=21), "固定 42日": dict(hold=42), "固定 63日(基準)": dict(hold=63), "固定 84日": dict(hold=84), "固定 126日": dict(hold=126), "固定 189日": dict(hold=189),
 "損切り-10% 63日": dict(hold=63, stop=.10), "損切り-15% 63日": dict(hold=63, stop=.15), "損切り-20% 63日": dict(hold=63, stop=.20), "損切り-15% 126日": dict(hold=126, stop=.15), "損切り-25% 126日": dict(hold=126, stop=.25),
 "利確+20% 63日": dict(hold=63, tp=.20), "利確+30% 63日": dict(hold=63, tp=.30), "利確+30% 126日": dict(hold=126, tp=.30), "利確+50% 126日": dict(hold=126, tp=.50),
 "追従損切り15% 126日": dict(hold=126, trail=.15), "追従損切り20% 126日": dict(hold=126, trail=.20), "追従損切り25% 126日": dict(hold=126, trail=.25),
 "追従15%(+15%超で有効) 126日": dict(hold=126, trail=.15, trail_act=.15), "追従10%(+20%超で有効) 126日": dict(hold=126, trail=.10, trail_act=.20), "追従12%(+30%超で有効) 189日": dict(hold=189, trail=.12, trail_act=.30),
 "25日線を回復で売り 63日": dict(hold=63, sig=0.0), "25日線を回復で売り 126日": dict(hold=126, sig=0.0), "25日線+3%で売り 126日": dict(hold=126, sig=.03), "25日線+6%で売り 126日": dict(hold=126, sig=.06), "25日線+10%で売り 189日": dict(hold=189, sig=.10),
 "21日で-5%なら見切り 126日": dict(hold=126, tstop=(21, .05)), "42日で-8%なら見切り 126日": dict(hold=126, tstop=(42, .08)), "63日で-10%なら見切り 126日": dict(hold=126, tstop=(63, .10)),
 "警戒サイン2個以上で売り 63日": dict(hold=63, crisis=True), "警戒サイン2個以上で売り 126日": dict(hold=126, crisis=True),
 "損切り-20%＋警戒サインで売り 126日": dict(hold=126, stop=.20, crisis=True), "損切り-25%＋追従15%(+15%超)＋警戒 126日": dict(hold=126, stop=.25, trail=.15, trail_act=.15, crisis=True),
}
res = {}
for sname, m in SETS.items():
    S = X[m.reindex(X.index).fillna(False)]
    print("=" * 200); print(sname, f"（共通の標本 {len(S)}件、2008〜2025年末まで）")
    rows = []; base = None
    for rn, spec in RULES.items():
        out = [run(e, j, **spec) for e, j in zip(S.e.values, S.j.values)]; ok = [i for i, o in enumerate(out) if o is not None]
        r = np.array([out[i][0] for i in ok]); d = np.array([out[i][1] for i in ok]); dt = S.date.values[ok]
        st = stats(r, d, dt); st["ルール"] = rn; st["r"] = r
        if rn == "固定 63日(基準)": base = r.copy(); bdt = dt
        rows.append(st)
    for st in rows: st["差のt"] = paired_t(st["r"], base, bdt) if len(st["r"]) == len(base) else np.nan
    R = pd.DataFrame(rows).drop(columns="r").set_index("ルール"); print(R.round(2).to_string()); res[sname] = R
pd.to_pickle(res, SP + "profit_exit_res.pkl")
