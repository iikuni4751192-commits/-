import numpy as np
import pandas as pd
import pytest

from backtest225 import data as D
from backtest225.engine import ExecutionConfig, RiskConfig, run_backtest
from backtest225.expr import ExprError, evaluate
from backtest225.strategy import entries_exits_to_target, rule_signals


def make_df(closes, opens=None, highs=None, lows=None, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    o = pd.Series(opens if opens is not None else closes, index=idx, dtype=float)
    h = pd.Series(highs if highs is not None else np.maximum(o, c), index=idx, dtype=float)
    l = pd.Series(lows if lows is not None else np.minimum(o, c), index=idx, dtype=float)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1000})


def test_signal_executes_next_open_not_same_day():
    df = make_df([100, 100, 110, 120], opens=[100, 100, 105, 115])
    target = pd.Series([np.nan, 1.0, np.nan, np.nan], index=df.index)  # 2日目終値でシグナル
    res = run_backtest(df, target, capital=10_000)
    t = res.trades[0]
    assert t.entry_date == df.index[2]
    assert t.entry_price == 105  # 翌日の始値
    assert res.equity["units"].iloc[1] == 0  # シグナル当日はまだ持っていない


def test_lot_rounding_and_cash_limit():
    df = make_df([100, 100, 100])
    target = pd.Series([1.0, np.nan, np.nan], index=df.index)
    lots = pd.Series(10, index=df.index)
    res = run_backtest(df, target, capital=1_555, lots=lots)
    assert res.equity["units"].iloc[-1] == 10  # 15口分の資金でも10口単位なので10口


def test_commission_and_slippage():
    df = make_df([100, 100, 100])
    target = pd.Series([1.0, 0.0, np.nan], index=df.index)
    ex = ExecutionConfig(commission_rate=0.01, slippage_rate=0.01)
    res = run_backtest(df, target, capital=100_000, execution=ex)
    assert res.equity["equity"].iloc[-1] < 100_000
    t = res.trades[0]
    assert t.entry_price == pytest.approx(101)
    assert t.exit_price == pytest.approx(99)


def test_stop_loss_gap_fills_at_open():
    df = make_df([100, 100, 100, 80], opens=[100, 100, 100, 85], lows=[100, 100, 99, 79])
    target = pd.Series([1.0, np.nan, np.nan, np.nan], index=df.index)
    res = run_backtest(df, target, capital=10_000, risk=RiskConfig(stop_loss=0.05))
    t = res.trades[0]
    assert t.exit_reason == "損切り"
    assert t.exit_price == 85  # 逆指値95を窓で飛び越えたので始値


def test_stop_loss_intraday_fills_at_stop():
    df = make_df([100, 100, 98], opens=[100, 100, 99], lows=[100, 100, 90])
    target = pd.Series([1.0, np.nan, np.nan], index=df.index)
    res = run_backtest(df, target, capital=10_000, risk=RiskConfig(stop_loss=0.05))
    assert res.trades[0].exit_price == pytest.approx(95)


def test_short_profit_when_price_falls():
    df = make_df([100, 100, 90, 80])
    target = pd.Series([-1.0, np.nan, np.nan, 0.0], index=df.index)
    res = run_backtest(df, target, capital=10_000)
    assert res.equity["equity"].iloc[-1] > 10_000


def test_dividend_credited_to_holder():
    df = make_df([100, 100, 100, 100])
    target = pd.Series([1.0, np.nan, np.nan, np.nan], index=df.index)
    div = pd.Series([2.0], index=[df.index[2]])
    res = run_backtest(df, target, capital=10_000, dividends=div)
    assert res.total_dividends == pytest.approx(100 * 2.0)


def test_expression_rules():
    df = make_df([1, 2, 3, 2, 1, 2, 3])
    s = evaluate("close > ref(close, 1) and not (close >= 3)", df)
    assert s.tolist() == [False, True, False, False, False, True, False]
    t = rule_signals(df, {"entry": "cross_up(close, 1.5)", "exit": "close < 1.5"})
    assert t.iloc[1] == 1.0 and t.iloc[4] == 0.0


def test_expression_rejects_unsafe():
    df = make_df([1, 2, 3])
    for bad in ["__import__('os')", "close.__class__", "open('x')", "[1,2]"]:
        with pytest.raises(ExprError):
            evaluate(bad, df)


def test_exit_wins_when_both_true():
    idx = pd.bdate_range("2024-01-01", periods=2)
    t = entries_exits_to_target(pd.Series([True, True], index=idx), pd.Series([False, True], index=idx))
    assert t.tolist() == [1.0, 0.0]


def test_split_adjustment_detects_unadjusted_csv():
    df = make_df([70000, 71000, 712, 720], start="2026-10-01")
    adj, notes = D.adjust_splits(df, D.KNOWN_SPLITS["1321.T"])
    assert adj["close"].iloc[0] == pytest.approx(700)
    assert notes
    adj2, notes2 = D.adjust_splits(adj, D.KNOWN_SPLITS["1321.T"])
    assert not notes2  # 調整済みには何もしない


def test_lot_schedule_1321():
    idx = pd.DatetimeIndex(["2026-10-02", "2026-10-05"])
    assert D.lot_size_series(idx, "1321.T", "auto").tolist() == [100, 10]


def test_ex_dates_t2_and_t3():
    idx = pd.bdate_range("2013-07-01", "2013-07-12").append(pd.bdate_range("2026-07-01", "2026-07-10"))
    div = pd.Series([1.87, 8.12], index=pd.to_datetime(["2013-07-08", "2026-07-08"]))
    ex = D.to_ex_dates(div, idx)
    # 2013年(T+3): 権利付最終日 7/3 → 権利落ち 7/4 / 2026年(T+2): 7/6 → 7/7
    assert list(ex.index) == [pd.Timestamp("2013-07-04"), pd.Timestamp("2026-07-07")]


def test_clean_fills_zero_open():
    df = make_df([100, 101, 102], opens=[100, 0, 102])
    out, notes = D.clean(df)
    assert out["open"].iloc[1] == 100 and notes
