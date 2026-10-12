"""ルール表シミュレーションの損失要因を取引単位で分解する診断スクリプト。

python analysis/diagnose_rulebook.py [--start 2024-09-17] [--breadth CSV]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest225 import data as D  # noqa: E402
from backtest225.rulebook import RulebookParams, load_breadth, load_events, run_rulebook  # noqa: E402
from backtest225.rulebook_cli import _nt_ratio  # noqa: E402


def setup_run(start, breadth_path, **kw):
    df, _ = D.load_prices()
    nt = _nt_ratio()
    ev = load_events()
    div = D.to_ex_dates(D.load_dividends_csv(D.BUILTIN_DIVIDENDS["1321.T"]), df.index)
    lots = D.lot_size_series(df.index, "1321.T", "auto")
    br = load_breadth(breadth_path) if breadth_path else None
    p = RulebookParams(use_breadth=br is not None, **kw)
    res, counts = run_rulebook(df, p, nt=nt, breadth=br, events=ev, dividends=div, lots=lots, start=start)
    return df, res, counts


def trade_table(df, res):
    rows = []
    idx = df.index
    for t in res.trades:
        if t.exit_reason.startswith("期間終了"):
            continue
        i0, i1 = idx.get_loc(t.entry_date), idx.get_loc(t.exit_date)
        r = t.risk_per_unit
        seg = df.iloc[i0:i1 + 1]
        mfe = (seg["high"].max() - t.entry_price) / r
        mae = (seg["low"].min() - t.entry_price) / r
        after = df["close"].iloc[i1:i1 + 21]
        post20 = (after.iloc[-1] / after.iloc[0] - 1) * 100 if len(after) > 1 else np.nan
        # 損切り後20日以内に建値+1Rへ届いたか（損切りが早すぎた指標）
        fwd_hi = df["high"].iloc[i1 + 1:i1 + 21].max() if i1 + 1 < len(df) else np.nan
        recovered = fwd_hi >= t.entry_price + r if pd.notna(fwd_hi) else False
        # エントリー時点の水準（ブレイク後どれだけ上で買ったか）
        rows.append({
            "setup": t.setup, "entry": t.entry_date.date(), "exit": t.exit_date.date(),
            "days": i1 - i0, "R": t.r_multiple, "pnl": t.pnl, "reason": t.exit_reason,
            "stop_atr": r / D_atr.loc[t.entry_date] if t.entry_date in D_atr.index else np.nan,
            "MFE_R": mfe, "MAE_R": mae, "post20_%": post20, "recovered_1R": recovered,
            "partial": any(x[3] == "部分利確" for x in t.exits),
            "event_half": any(x[3].startswith("イベント前") for x in t.exits),
            "exit_px_vs_stop_R": (t.exits[-1][2] - (t.entry_price - r)) / r,
        })
    return pd.DataFrame(rows)


def exposure_attribution(df, res):
    eq = res.equity
    ret = df["close"].reindex(eq.index).pct_change().fillna(0)
    held = (eq["units"].shift(1).fillna(0) > 0)  # 前日終値時点で保有＝当日の値動きを受ける
    return {
        "保有日数": int(held.sum()), "非保有日数": int((~held).sum()),
        "保有日の指数累積%": (np.prod(1 + ret[held]) - 1) * 100,
        "非保有日の指数累積%": (np.prod(1 + ret[~held]) - 1) * 100,
        "保有時の平均建玉比率%": eq.loc[eq["units"] > 0, "exposure"].mean() * 100,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-09-17")
    ap.add_argument("--breadth")
    a = ap.parse_args()
    df, res, counts = setup_run(a.start, a.breadth)
    from backtest225 import indicators as ind
    D_atr = ind.atr(df["high"], df["low"], df["close"], 14)
    tt = trade_table(df, res)
    pd.set_option("display.width", 220)
    print(tt.round(2).to_string())
    print("\n[セットアップ×決済理由]")
    print(tt.groupby(["setup", "reason"]).agg(n=("R", "size"), avgR=("R", "mean"), pnl=("pnl", "sum")).round(2))
    print("\n[MFE/MAE] 平均MFE %.2fR / 平均MAE %.2fR / +1R到達率 %.0f%% / +2R到達率 %.0f%%" % (
        tt.MFE_R.mean(), tt.MAE_R.mean(), (tt.MFE_R >= 1).mean() * 100, (tt.MFE_R >= 2).mean() * 100))
    st = tt[tt.reason == "初期損切り"]
    print("[初期損切り] %d件 平均R %.2f / 損切り後20日内に建値+1R到達 %d件 / 決済価格の損切り価格からのずれ平均 %.2fR" % (
        len(st), st.R.mean(), st.recovered_1R.sum(), st.exit_px_vs_stop_R.mean()))
    ma = tt[tt.reason == "MA25割れ"]
    print("[MA25割れ決済] %d件 平均保有 %.1f日 平均R %.2f / 決済後20日の騰落 平均 %.2f%%" % (
        len(ma), ma.days.mean(), ma.R.mean(), ma["post20_%"].mean()))
    print("[損切り幅] 平均 %.2fATR（中央値 %.2f）" % (tt.stop_atr.mean(), tt.stop_atr.median()))
    print("[イベント前半減] %d件 そのうち最終R>0: %d" % (tt.event_half.sum(), (tt[tt.event_half].R > 0).sum()))
    print("[保有/非保有の寄与]", {k: round(v, 2) for k, v in exposure_attribution(df, res).items()})
    print("[見送り]", counts)
