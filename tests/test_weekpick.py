import numpy as np
import pandas as pd

from weekpick.backtest import Setup, prepare, run, stats, weeks
from weekpick.data import build_panel
from weekpick.factors import score


def make_frames(n_days=300, tickers=("A", "B", "C"), seed=0):
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    out = {}
    for i, t in enumerate(tickers):
        c = 1000 * np.cumprod(1 + rng.normal(0, 0.01, n_days)) * (i + 1)
        out[t] = pd.DataFrame({"open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "close": c,
                               "volume": 1e6, "adj_close": c}, index=idx)
    return out


def test_weeks_use_previous_week_last_day_as_signal():
    idx = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-09", "2024-01-11"])  # 1/8(月)祝日
    w = weeks(idx)
    assert len(w) == 1
    assert w[0].signal == pd.Timestamp("2024-01-05")
    assert w[0].entry == pd.Timestamp("2024-01-09")  # 月曜休場なら火曜の始値
    assert w[0].exit == pd.Timestamp("2024-01-11")   # 金曜休場なら木曜の終値


def test_week_return_is_monday_open_to_friday_close_with_cost():
    frames = make_frames()
    p = build_panel(frames)
    s = Setup(weights={"reversal": 1}, min_turnover=0, min_history=10, cost=0.001)
    prep = prepare(p, s)
    w = prep.weeks[5]
    exp = frames["A"].loc[w.exit, "close"] / frames["A"].loc[w.entry, "open"]
    assert np.isclose(prep.returns.loc[w.entry, "A"], exp * 0.999 / 1.001 - 1)


def test_dividend_included_via_adj_close():
    frames = make_frames()
    f = frames["A"]
    # 終値は配当分下がるが、配当込み終値（adj_close）はその前を割り引いて連続
    f["adj_close"] = f["close"]
    ex = f.index[103]
    f.loc[f.index < ex, "adj_close"] = f.loc[f.index < ex, "close"] * 0.98
    p = build_panel(frames)
    prep = prepare(p, Setup(weights={"reversal": 1}, min_turnover=0, min_history=10, cost=0))
    w = next(w for w in prep.weeks if w.entry < ex <= w.exit)
    exp = f.loc[w.exit, "close"] / (f.loc[w.entry, "open"] * 0.98) - 1
    assert np.isclose(prep.returns.loc[w.entry, "A"], exp)


def test_score_uses_only_signal_day_information():
    frames = make_frames()
    p = build_panel(frames)
    s = Setup(weights={"reversal": 1}, min_turnover=0, min_history=10)
    a = prepare(p, s)
    # 未来の値を書き換えてもスコアは変わらない
    frames2 = {k: v.copy() for k, v in frames.items()}
    cut = a.weeks[10].entry
    frames2["A"].loc[frames2["A"].index >= cut, ["open", "high", "low", "close", "adj_close"]] *= 3
    b = prepare(build_panel(frames2), s)
    d = a.weeks[10].entry
    pd.testing.assert_series_equal(a.scores.loc[d], b.scores.loc[d])


def test_score_rank_and_weights():
    idx = pd.DatetimeIndex(["2024-01-05"])
    f = {"x": pd.DataFrame([[1.0, 2.0, 3.0]], index=idx, columns=list("ABC"))}
    m = pd.DataFrame(True, index=idx, columns=list("ABC"))
    assert score(f, {"x": 1}, m).loc[idx[0]].idxmax() == "C"
    assert score(f, {"x": -1}, m).loc[idx[0]].idxmax() == "A"


def test_run_picks_top_n_and_capital_lots():
    frames = make_frames(tickers=("A", "B", "C", "D"))
    p = build_panel(frames)
    prep = prepare(p, Setup(weights={"reversal": 1}, min_turnover=0, min_history=10, cost=0))
    w = run(prep, 2)
    d = w.index[3]
    top2 = prep.scores.loc[d].nlargest(2).index
    assert np.isclose(w.loc[d, "ret"], prep.returns.loc[d, top2].mean())
    assert set(w.loc[d, "picks"].split()) == set(top2)
    # 資金 30万円・2銘柄 → 1銘柄15万円まで。100株で15万円を超える銘柄（B,C,Dは株価2000円超）は対象外
    wc = run(prep, 2, capital=300_000)
    assert all(t == "A" for t in " ".join(wc["picks"]).split())
    assert (wc["invested"] <= 0.5 + 1e-9).all()
    st = stats(w)
    assert st["weeks"] == len(w) and -1 < st["max_dd"] <= 0


def test_dividend_factor_ignores_broken_adj_close():
    from weekpick.data import dividend_factor

    idx = pd.bdate_range("2024-01-01", periods=6)
    close = pd.Series(100.0, index=idx)
    adj = pd.Series([97, 97, 100, 1e30, -5, 100], index=idx, dtype=float)  # 3日目に3%配当、4〜5日目は異常
    f, bad = dividend_factor(adj, close)
    assert bad == 1
    assert np.isclose(f.iloc[0], 0.97) and np.isclose(f.iloc[-1], 1.0)
    assert (f.diff().dropna() >= -1e-12).all()


def test_drop_outliers_removes_impossible_jump():
    from weekpick.data import drop_outliers

    idx = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame({c: [100, 101, 5.5e10, 100, 102] for c in ["open", "high", "low", "close"]}, index=idx, dtype=float)
    df["volume"] = 1.0
    out, n = drop_outliers(df)
    assert n == 1 and out["close"].isna().sum() == 1 and out["close"].iloc[3] == 100


def test_detect_unadjusted_split():
    from weekpick.data import adjust_unadjusted_splits

    idx = pd.bdate_range("2024-01-01", periods=20)
    c = np.r_[np.full(10, 11000.0), np.full(10, 3700.0)]
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1000.0}, index=idx)
    out, found = adjust_unadjusted_splits(df)
    assert len(found) == 1 and "1:3" in found[0]
    assert np.isclose(out["close"].iloc[0], 11000 / 3) and out["volume"].iloc[0] == 3000
    same, none = adjust_unadjusted_splits(out)
    assert none == []


def _timing_panel():
    frames = make_frames(n_days=400, tickers=("A", "B", "C", "D"), seed=3)
    # A: 週末に急落させて RSI(2)<10 を作る
    f = frames["A"]
    for i in range(300, 305):
        f.iloc[i, :5] = f.iloc[i - 1]["close"] * 0.97 * np.array([1.0, 1.0, 0.99, 1.0, 1.0])
        f.iloc[i, 3] = f.iloc[i - 1]["close"] * 0.97
        f.iloc[i, 5] = f.iloc[i, 3]
    return build_panel(frames)


def test_timing_flags_and_history():
    from weekpick import timing as T

    p = _timing_panel()
    sectors = pd.Series({"A": "x", "B": "x", "C": "y", "D": "y"})
    ind = T.indicators(p, sectors)
    C = p.close
    d = C.index[310]
    assert ind["flag_dev25"].at[d, "A"] == (C.at[d, "A"] / C["A"].rolling(25).mean().at[d] - 1 <= -0.10)
    mkt = (C / C.shift(5) - 1).mean(axis=1)
    assert ind["flag_mkt_drop"].at[d, "B"] == (mkt.at[d] <= -0.03)
    h = T.history(p, ind, ["A"], start="2024-01-01")
    assert (h["RSI2"] < T.RSI2_THRESHOLD).all() and len(h) >= 1
    row = h.iloc[0]
    assert row["参考点"] == T.score(row.to_dict())
    last = float(p.close["A"].iloc[-1])
    for entry in (last, float(p.low["A"].iloc[-10:].min()) * 0.9):  # 通常 / 直近安値を割って寄り付いた場合
        lv = T.trade_levels(p, "A", entry=entry)
        assert lv["A_損切り"] < entry and np.isclose(lv["A_利確"] - entry, 2 * (entry - lv["A_損切り"]))
        assert np.isclose(entry - lv["B_損切り"], 2 * lv["ATR14"])


def test_timing_williams_r_signal_and_peer_share():
    from weekpick import timing as T

    p = _timing_panel()
    sectors = pd.Series({"A": "x", "B": "x", "C": "x", "D": "y"})
    ind = T.indicators(p, sectors)
    H, L, C = p.high, p.low, p.close
    d = C.index[304]
    exp = -100 * (H["A"].rolling(14).max().at[d] - C.at[d, "A"]) / (H["A"].rolling(14).max().at[d] - L["A"].rolling(14).min().at[d])
    assert np.isclose(ind["wr14"].at[d, "A"], exp)
    # 同業種の他銘柄（B, C）のうち %R<-80 の割合（A 自身は数えない）
    on = ind["wr14"].loc[d, ["B", "C"]] < -80
    assert np.isclose(ind["peer_wr_share_pct"].at[d, "A"], on.mean() * 100)
    assert np.isnan(ind["peer_wr_share_pct"].at[d, "D"])  # 同業種に他銘柄がいない
    h = T.history(p, ind, ["A"], start="2024-01-01", signal="wr14", conditions=T.WR_CONDITIONS)
    assert len(h) >= 1 and (h["%R14"] < -80).all()
    assert set(c.key for c in T.WR_CONDITIONS) <= set(h.columns)
    eff = T.condition_effect(p, ind, "wr14", T.WR_CONDITIONS, start="2024-01-01")
    assert list(eff["条件"]) == [c.label for c in T.WR_CONDITIONS]
    assert ((eff["あり_回数"] >= 0)).all()


def test_sar_flip_detects_reversal():
    from weekpick.timing import _sar_bull_flip

    down = np.linspace(100, 80, 15)
    up = np.linspace(80, 100, 15)
    c = np.r_[down, up]
    flip = _sar_bull_flip(c + 0.5, c - 0.5)
    assert flip[15:].any() and not flip[:15].any()
