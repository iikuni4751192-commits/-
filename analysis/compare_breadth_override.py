"""騰落レシオ120%以上（E5）で新規を見送るかを、海外指標・日経先物で判断し直す案の検証。
python analysis/compare_breadth_override.py

騰落レシオのデータは 2024-09-17 以降のみのため、検証できるのはこの期間だけ（開発期間では E5 自体が働かない）。
判断に使う値はすべてシグナル日（日本の大引け）時点で確定しているもの（米国・CMEは米国日付が前日以前の終値）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest225 import data as D  # noqa: E402
from backtest225.core_satellite import run_core_satellite  # noqa: E402
from backtest225.metrics import compute  # noqa: E402
from backtest225.rulebook import RulebookParams, load_breadth, load_events, summarize  # noqa: E402
from backtest225.rulebook_cli import _nt_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
START = "2024-09-17"


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

    def trend_ok(ticker):  # 200日線の上 かつ 20日騰落がプラス
        s = D.fetch_yahoo(ticker)["close"]
        return avail_before((s > s.rolling(200).mean()) & (s > s.shift(20)), idx).astype(bool)

    spx, ndx, sox = trend_ok("^GSPC"), trend_ok("^IXIC"), trend_ok("^SOX")
    niy = avail_before(D.fetch_yahoo("NIY=F")["close"], idx)
    n225 = D.fetch_yahoo("^N225")["close"].reindex(idx)
    niy_up = (niy / n225.shift(1) - 1) > 0  # シグナル日の前夜のCME先物が前日日経平均終値より上
    variants = {
        "基準（騰落レシオ120%以上は常に見送り）": None,
        "見送りをやめる（常にエントリー）": pd.Series(True, index=idx),
        "S&P500が上昇トレンドなら見送らない": spx,
        "NASDAQが上昇トレンドなら見送らない": ndx,
        "SOXが上昇トレンドなら見送らない": sox,
        "米3指数すべて上昇トレンドなら見送らない": spx & ndx & sox,
        "前夜のCME日経先物が上昇なら見送らない": niy_up,
    }
    for name, ov in variants.items():
        res, c = run_core_satellite(df, RulebookParams(**base), lots=lots, dividends=div, start=START,
                                    nt=nt, breadth=br, events=ev, hot_override=ov)
        m = compute(res)
        s, b = m["strategy"], m["benchmark"]
        t = summarize(res.satellite)["trades"]
        print(f"{name:<26} 最終資産 {s['final_equity']:>11,.0f}円 年率 {s['cagr']*100:6.2f}% 最大DD {s['max_drawdown']*100:5.1f}% "
              f"信用 {s['n_trades']:2d}回 勝率 {s['win_rate']*100:3.0f}% PF {s['profit_factor']:.2f} 信用損益 {t.pnl.sum():>9,.0f}円 過熱で見送り {c['skipped_breadth']}回")
    print(f"（買い持ち 年率 {b['cagr']*100:.2f}%）\n")

    # 過熱で見送ったシグナルの一覧と各指標の状態
    res, c = run_core_satellite(df, RulebookParams(**base), lots=lots, dividends=div, start=START, nt=nt, breadth=br, events=ev)
    rows = []
    for sk in c.get("skips", []):
        if sk["reason"] != "騰落レシオ過熱":
            continue
        k = idx.get_loc(sk["date"])
        win = df.iloc[k + 1:k + 21]
        base_px = df["open"].iloc[k + 1]
        rows.append({"シグナル日": sk["date"].date(), "種類": sk["setup"], "騰落レシオ": round(br.get(sk["date"], np.nan), 1),
                     "S&P": "○" if spx.iloc[k] else "×", "NASDAQ": "○" if ndx.iloc[k] else "×", "SOX": "○" if sox.iloc[k] else "×",
                     "前夜先物": "↑" if niy_up.iloc[k] else "↓",
                     "20日内最高値まで%": round((win["high"].max() / base_px - 1) * 100, 2),
                     "20日後%": round((win["close"].iloc[-1] / base_px - 1) * 100, 2) if len(win) == 20 else np.nan,
                     "20日内に損切り価格割れ": bool((win["close"] <= sk["stop"]).any())})
    pd.set_option("display.width", 200, "display.unicode.east_asian_width", True)
    print("[騰落レシオ過熱で見送ったシグナル（延べ）]")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
