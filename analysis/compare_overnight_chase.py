"""寄り前に分かる夜間の値動きで、Setup B の発注方法を変える案の検証（買い持ち＋信用上乗せの現行設定）。
python analysis/compare_overnight_chase.py

基準: B は「ブレイク水準ちょうど」の指値
案1: 夜間のCME日経先物（円建て）が前日の日経平均終値より +0.5% 超 → B を寄付成行で買う（それ以外は指値）
案2: 夜間の SOX が +0.5% 超 → 寄付成行
案3: 夜間の NASDAQ が +0.5% 超 → 寄付成行
参考: 夜間の情報を使わず、B は常に寄付成行
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest225 import data as D  # noqa: E402
from backtest225.core_satellite import run_core_satellite  # noqa: E402
from backtest225.metrics import compute  # noqa: E402
from backtest225.rulebook import RulebookParams, load_breadth, load_events, summarize  # noqa: E402
from backtest225.rulebook_cli import _nt_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PERIODS = [("開発 2010-01〜2024-08", "2010-01-04", "2024-08-31"), ("W 2024-09〜2026-10", "2024-09-17", None),
           ("全期間", "2010-01-04", None)]


def avail_before(series, index):
    s = series.copy()
    s.index = s.index + pd.Timedelta(days=1)
    return s.reindex(index, method="ffill")


def main():
    base = yaml.safe_load(open(ROOT / "strategies/rulebook_1321.yaml", encoding="utf-8"))["params"]
    df, _ = D.load_prices()
    idx = df.index
    nt, ev, br = _nt_ratio(), load_events(), load_breadth(ROOT / "data/cache/breadth_prime25.csv")
    div = D.to_ex_dates(D.load_dividends_csv(D.BUILTIN_DIVIDENDS["1321.T"]), idx)
    lots = D.lot_size_series(idx, "1321.T", "auto")
    n225_prev = D.fetch_yahoo("^N225")["close"].reindex(idx).shift(1)  # 発注日の前営業日（シグナル日）の終値

    def overnight(ticker, ref=None):
        s = D.fetch_yahoo(ticker)["close"]
        now = avail_before(s, idx)  # 発注日の寄り前に分かる最新終値
        if ref is not None:
            return (now / ref - 1) * 100
        prev = avail_before(s, idx.to_series().shift(1).bfill())  # シグナル日の大引け時点で分かっていた終値
        return (now.to_numpy() / prev.to_numpy() - 1) * 100

    chase = {
        "基準（B は水準ちょうどの指値）": None,
        "案1 夜間CME日経先物 >+0.5% なら寄付成行": overnight("NIY=F", n225_prev) > 0.5,
        "案2 夜間SOX >+0.5% なら寄付成行": pd.Series(overnight("^SOX") > 0.5, index=idx),
        "案3 夜間NASDAQ >+0.5% なら寄付成行": pd.Series(overnight("^IXIC") > 0.5, index=idx),
        "参考 B は常に寄付成行": pd.Series(True, index=idx),
    }
    for name, ch in chase.items():
        print(name)
        for label, start, end in PERIODS:
            d = df if end is None else df[idx <= end]
            res, c = run_core_satellite(d, RulebookParams(**base), lots=lots, dividends=div, start=start,
                                        nt=nt, breadth=br, events=ev, b_chase=ch)
            m = compute(res)
            s, b = m["strategy"], m["benchmark"]
            t = summarize(res.satellite)["trades"]
            tb = t[t.setup == "B"]
            print(f"  {label}: 最終資産 {s['final_equity']:>12,.0f}円 年率 {s['cagr']*100:5.2f}% (買い持ち {b['cagr']*100:5.2f}%) 最大DD {s['max_drawdown']*100:5.1f}% "
                  f"| 信用 {s['n_trades']}回 PF {s['profit_factor']:.2f} B {len(tb)}回 平均R {tb.r.mean():.2f} 損益 {tb.pnl.sum():>10,.0f}円 "
                  f"成行 {c.get('chased', 0)}回 指値不成立 {c['unfilled']}")


if __name__ == "__main__":
    main()
