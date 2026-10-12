"""現行設定（買い持ち＋信用上乗せ）の改善余地を調べる診断。 python analysis/diagnose_core_satellite.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest225 import data as D  # noqa: E402
from backtest225.core_satellite import run_core_satellite  # noqa: E402
from backtest225.rulebook import RulebookParams, compute_features, load_breadth, load_events  # noqa: E402
from backtest225.rulebook_cli import _nt_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
base = yaml.safe_load(open(ROOT / "strategies/rulebook_1321.yaml", encoding="utf-8"))["params"]
p = RulebookParams(**base)
df, _ = D.load_prices()
nt, ev, br = _nt_ratio(), load_events(), load_breadth(ROOT / "data/cache/breadth_prime25.csv")
div = D.to_ex_dates(D.load_dividends_csv(D.BUILTIN_DIVIDENDS["1321.T"]), df.index)
lots = D.lot_size_series(df.index, "1321.T", "auto")
res, c = run_core_satellite(df, p, lots=lots, dividends=div, start="2010-01-04", nt=nt, breadth=br, events=ev)

rows = []
for t in res.satellite.trades:
    if t.exit_reason.startswith("期間終了"):
        continue
    i0, i1 = df.index.get_loc(t.entry_date), df.index.get_loc(t.exit_date)
    seg, fwd = df.iloc[i0:i1 + 1], df["close"].iloc[i1:i1 + 21]
    rows.append({"setup": t.setup, "year": t.entry_date.year, "R": t.r_multiple, "pnl": t.pnl, "reason": t.exit_reason,
                 "mfe": (seg["high"].max() - t.entry_price) / t.risk_per_unit,
                 "giveback": (seg["high"].max() - t.exits[-1][2]) / t.risk_per_unit,
                 "post20": (fwd.iloc[-1] / fwd.iloc[0] - 1) * 100 if len(fwd) > 1 else np.nan})
T = pd.DataFrame(rows)
print("取引", len(T), "信用損益", round(T.pnl.sum()), "金利", round(c["total_interest"]), "上位5取引", round(T.pnl.nlargest(5).sum()))
print(T.groupby("setup").agg(n=("R", "size"), avgR=("R", "mean"), pnl=("pnl", "sum")).round(2))
print(T.groupby("reason").agg(n=("R", "size"), avgR=("R", "mean"), pnl=("pnl", "sum"), mfe=("mfe", "mean"),
                              giveback=("giveback", "mean"), post20=("post20", "mean")).round(2))
print("年別", T.groupby("year").pnl.sum().round(0).to_dict())
print("見送り", {k: v for k, v in c.items() if k.startswith("skipped") or k in ("unfilled", "signals")})

# Setup B のシグナル日：翌日指値（水準ちょうど）が約定したか別の20日先騰落
f = compute_features(df, p, nt, br)
cl, m = df["close"], df.index >= "2010-01-04"
B = f.env.isin(["上昇", "レンジ・移行"]) & (f.nt >= f.nt_ma) & (cl > f.cloud_top) & (cl > f.hh_prev) & m & ~(f.breadth >= 120).fillna(False)
nxt_open = df["open"].shift(-1)
fwd20 = (cl.shift(-20) / nxt_open - 1) * 100
t = pd.DataFrame({"約定": (df["low"].shift(-1) <= f.hh_prev)[B], "fwd20": fwd20[B]}).dropna()
print(t.groupby("約定").fwd20.agg(["size", "mean"]).round(2), "全営業日平均 %.2f%%" % fwd20[m].mean())
