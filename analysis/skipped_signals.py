"""見送ったシグナルと、その後の値動き。 python analysis/skipped_signals.py [--top 12]

現行設定（strategies/rulebook_1321.yaml：買い持ち＋信用上乗せ）で全期間を実行し、見送ったシグナルを記録する。
連続する日に同じ理由で見送った同じセットアップは1件にまとめ、最初の日を表示する。
その後の値動き＝シグナル翌日の始値を基準に、20営業日以内の最高値までの上昇率と、20営業日後の終値の騰落率。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest225 import data as D, indicators as ind  # noqa: E402
from backtest225.core_satellite import run_core_satellite  # noqa: E402
from backtest225.rulebook import RulebookParams, compute_features, load_breadth, load_events  # noqa: E402
from backtest225.rulebook_cli import _nt_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(top):
    base = yaml.safe_load(open(ROOT / "strategies/rulebook_1321.yaml", encoding="utf-8"))["params"]
    p = RulebookParams(**base)
    df, _ = D.load_prices()
    nt, ev, br = _nt_ratio(), load_events(), load_breadth(ROOT / "data/cache/breadth_prime25.csv")
    div = D.to_ex_dates(D.load_dividends_csv(D.BUILTIN_DIVIDENDS["1321.T"]), df.index)
    lots = D.lot_size_series(df.index, "1321.T", "auto")
    res, c = run_core_satellite(df, p, lots=lots, dividends=div, start="2010-01-04", nt=nt, breadth=br, events=ev)
    f = compute_features(df, p, nt, br)
    idx = df.index
    entries = {t.entry_date for t in res.satellite.trades}
    held = res.equity["units"] > 0

    rows = []
    for s in c.get("skips", []):
        k = idx.get_loc(s["date"])
        sig = k if s["signal_day"] else k - 1          # シグナル日（終値判定日）
        if sig + 1 >= len(idx):
            continue
        base_px = df["open"].iloc[sig + 1]
        win = df.iloc[sig + 1: sig + 21]
        hi = win["high"].max()
        r = s["limit"] - s["stop"]
        rows.append({
            "シグナル日": idx[sig].date(), "セットアップ": s["setup"], "見送り理由": s["reason"],
            "終値": round(df["close"].iloc[sig], 1), "指値": round(s["limit"], 1), "損切り": round(s["stop"], 1),
            "ATR": round(f["atr"].iloc[sig], 2), "環境": f["env"].iloc[sig],
            "騰落レシオ": round(f["breadth"].iloc[sig], 1) if pd.notna(f["breadth"].iloc[sig]) else None,
            "20日内最高値まで%": round((hi / base_px - 1) * 100, 2),
            "20日後%": round((win["close"].iloc[-1] / base_px - 1) * 100, 2) if len(win) == 20 else np.nan,
            "最高値までR": round((hi - s["limit"]) / r, 2) if r > 0 else np.nan,
            "20日内に損切り価格割れ": bool((win["close"] <= s["stop"]).any()),
            "同日に別セットアップで建玉": idx[sig + 1] in entries,
            "_k": sig,
        })
    t = pd.DataFrame(rows).sort_values("_k")
    # 連続日の重複をまとめる（同じセットアップ・同じ理由の種類で3営業日以内に続くもの）
    t["理由種別"] = t["見送り理由"].str.replace(r"損切り幅.*", "損切り幅外", regex=True)
    keep, last = [], {}
    for i, row in t.iterrows():
        key = (row["セットアップ"], row["理由種別"])
        if key in last and row["_k"] - last[key] <= 3:
            last[key] = row["_k"]
            continue
        last[key] = row["_k"]
        keep.append(i)
    ep = t.loc[keep].drop(columns=["_k"])
    ep = ep[~ep["同日に別セットアップで建玉"]]

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.unicode.east_asian_width", True)
    print(f"見送り（延べ）{len(t)}件 → 連続分をまとめて {len(ep)}件\n")
    print("[理由別] 20日内最高値までの上昇率と20日後騰落率（翌日始値基準）")
    print(ep.groupby("理由種別").agg(件数=("20日後%", "size"), 最高値まで平均=("20日内最高値まで%", "mean"),
                                    二十日後平均=("20日後%", "mean"), 上昇割合=("20日後%", lambda s: (s > 0).mean() * 100),
                                    損切り割れ割合=("20日内に損切り価格割れ", lambda s: s.mean() * 100)).round(2))
    allfwd = (df["close"].shift(-20) / df["open"].shift(-1) - 1)[idx >= "2010-01-04"] * 100
    print(f"（参考）全営業日の20日後騰落 平均 {allfwd.mean():.2f}%\n")
    cols = ["シグナル日", "セットアップ", "見送り理由", "終値", "指値", "損切り", "環境", "騰落レシオ",
            "20日内最高値まで%", "20日後%", "最高値までR", "20日内に損切り価格割れ"]
    print(f"[20日内の上昇が大きかった見送り 上位{top}件]")
    print(ep.sort_values("20日内最高値まで%", ascending=False)[cols].head(top).to_string(index=False))
    ep.to_csv(ROOT / "reports/skipped_signals.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    main(ap.parse_args().top)
