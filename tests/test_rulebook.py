import numpy as np
import pandas as pd

from backtest225.rulebook import RulebookParams, event_block_days, ichimoku_cloud, load_breadth, run_rulebook


def test_event_block_days():
    idx = pd.bdate_range("2026-09-01", "2026-09-30")
    ev = pd.DataFrame({"date": pd.to_datetime(["2026-09-11", "2026-09-16", "2026-09-18"]),
                       "type": ["CPI", "FOMC", "BOJ"]})
    b = event_block_days(idx, ev)
    marked = b[b != ""]
    assert list(marked.index.strftime("%m-%d")) == ["09-10", "09-11", "09-16", "09-17"]
    assert marked["2026-09-10"] == "SQ前日"  # 第2金曜 9/11 の前営業日


def test_cloud_is_shifted_forward():
    idx = pd.bdate_range("2024-01-01", periods=120)
    df = pd.DataFrame({"high": np.arange(120.0) + 1, "low": np.arange(120.0)}, index=idx)
    kijun, top, bot = ichimoku_cloud(df, RulebookParams())
    span_b_raw = (df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2
    assert bot.iloc[100] == span_b_raw.iloc[75]  # 25本前に計算した値が当日の雲


def test_load_breadth_japanese(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("日付,騰落レシオ\n2024/01/04,105.2%\n2024/01/05,98.1\n", encoding="cp932")
    s = load_breadth(p)
    assert s.tolist() == [105.2, 98.1]


def test_rulebook_runs_and_respects_lots():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=700)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, len(idx))))
    o = c * (1 + rng.normal(0, 0.003, len(idx)))
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.005, "low": np.minimum(o, c) * 0.995,
                       "close": c, "volume": 1}, index=idx)
    lots = pd.Series(100, index=idx)
    res, counts = run_rulebook(df, RulebookParams(use_events=False, use_breadth=False), lots=lots)
    assert len(res.equity) == len(df)
    for t in res.trades:
        assert t.initial_units % 100 == 0
        assert t.risk_per_unit > 0
