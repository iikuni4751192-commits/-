"""「NF日経225（1321）テクニカル・スイング 投資ルール表」（Excel）の専用シミュレーター。

汎用エンジン（engine.py）では表せない、ルール表固有の仕組みを1日ずつ再現する:
  - セットアップ A（押し目）/ B（ブレイク）/ C（リバウンド）と優先順位 A>B>C
  - 翌営業日の指値（寄り前発注・当日限り）、構造ラインの損切り、1〜3ATRの損切り幅チェック
  - 1トレード許容リスク1%の口数計算、サイズ係数（MA200下・Setup C・連敗）
  - +1Rで建値へ、+2Rで50%利確、ATRトレーリング/MA25割れ、時間ストップ、Setup C の目標
  - 重要イベント（SQ・日銀・FOMC・米CPI）前の新規見送りと建玉半減
  - 1日・月間の損失上限
ルール表で定義が曖昧な点の解釈は RULE_NOTES にまとめ、レポートに明記する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import indicators as ind
from .engine import Result, Trade

RULE_NOTES = [
    "価格・指標はすべて1321の日足（分割調整後）で計算。判定は終値、発注は翌営業日の寄り前（当日限り）",
    "指値は「寄りが指値以下なら寄値、安値が指値以下なら指値」で約定。逆指値は寄りで飛び越えたら寄値",
    "同じ日に損切りと利確の両方の価格に届いた場合は損切りを優先（保守的）",
    "雲＝一目均衡表の先行スパンA/Bを25本先にずらしたもの（当日を1本目とする26日先行）",
    "スイング高値＝直近20日の最高値、スイング安値＝その高値の日より前の20日間の最安値",
    "Setup A の指値＝シグナル日の終値（押し目ゾーン内で終えた日の終値以下で買う）。損切り＝フィボ61.8%の1ティック下（建値以上になる場合はスイング安値の1ティック下）",
    "Setup B の指値＝直近20日高値＋0.5ATR。損切り＝ブレイク水準−1ATR、または終値が雲上限を割れた翌日寄り",
    "Setup C は C1 が成立した日またはその翌日に、陽線引け／長い下ヒゲ（下ヒゲ≧実体×2 かつ 値幅の50%以上）で確認。指値＝確認日の終値、損切り＝直近5日安値の1ティック下",
    "損切り幅が1〜3ATRに収まらないシグナルは全セットアップで見送り（1_指標設定 C23・C24）",
    "トレーリング（保有後最高値−3ATR）・MA25割れは部分利確の前後を問わず建玉全体に適用",
    "時間ストップ：A/B は保有15営業日までに一度も+1Rに届かなければ翌寄りで決済、C は5営業日で翌寄りで決済",
    "+1R到達で翌日から損切りを建値へ。+2R到達で50%を指値利確（C も対象）",
    "サイズ係数：MA200下・Setup C・3連敗以上のいずれかで0.5（重なっても0.5が下限）。勝てば連敗リセット",
    "1日の損失上限2%（終値ベースの評価損益込み）に達したら翌営業日の新規停止。月間−6%で当月の新規停止",
    "重要イベントの『前営業日』：日銀＝会合最終日の前営業日、FOMC・米CPI＝米国の発表日（日本時間では翌日未明/夜）以前の最後の営業日、SQ＝SQ日（第2金曜、休業日なら前営業日）の前営業日。この日は新規約定しない",
    "X10：イベント前営業日の寄りで、損切りがまだ建値未満なら建玉の半分を決済（建値以上なら維持）",
    "朝7時評点（G1）は過去データがないため常に『通常エントリー可』として扱う。X9（任意の追加利確）は未反映",
    "G2：建玉は1つまで（ナンピン・分割エントリーなし）。NT倍率は ^N225 ÷ 1305.T（TOPIX連動ETF）で代用",
]


@dataclass
class RulebookParams:
    # 1_指標設定
    ma_short: int = 5
    ma_mid: int = 25
    ma_long: int = 75
    ma_ultra: int = 200
    tenkan: int = 9
    kijun: int = 26
    senkou_b: int = 52
    bb_n: int = 20
    bb_k: float = 2.0
    atr_n: int = 14
    breadth_hot: float = 120
    breadth_cold: float = 70
    nt_ma: int = 20
    fib_shallow: float = 0.382
    fib_deep: float = 0.618
    stop_atr: float = 2.0
    trail_atr: float = 3.0
    stop_min_atr: float = 1.0
    stop_max_atr: float = 3.0
    partial_r: float = 2.0
    partial_ratio: float = 0.5
    time_stop_ab: int = 15
    time_stop_c: int = 5
    near_atr: float = 0.5
    b_limit_atr: float = 0.5  # Setup B の指値＝ブレイク水準＋この倍率×ATR（3_エントリー B3。原因5の検証）
    b_stop_atr: float = 1.0  # Setup B の損切り＝ブレイク水準−この倍率×ATR（3_エントリー B4）
    breakout_n: int = 20
    swing_n: int = 20
    # 5_資金管理
    capital: float = 2_400_000
    risk_per_trade: float = 0.01
    daily_loss_limit: float = 0.02
    monthly_loss_limit: float = 0.06
    losing_streak: int = 3
    streak_factor: float = 0.5
    position_cap: float = 1.0
    # 執行・コスト
    tick: float = 0.1
    commission_rate: float = 0.0
    slippage_rate: float = 0.0005  # 成行・逆指値の約定のみ（指値は滑らない）
    use_events: bool = True
    event_entry_block: bool = True   # E8 イベント前営業日は新規見送り（原因4の検証で切替）
    event_halving: bool = True       # X10 イベント前営業日に建玉半減
    breadth_hot_filter: bool = True  # E5 騰落レシオ過熱で新規見送り
    # 信用取引（買い持ち＋信用で上乗せ の検証用）。0 なら現物（金利なし）
    margin_rate: float = 0.0          # 買方金利（年率）。楽天証券 制度・一般無期限とも 2.80%（2026年9月 信用取引ルール）
    margin_requirement: float = 0.30  # 新規建に必要な委託保証金率（楽天証券 レバレッジ3.3倍＝30%）
    maintenance_ratio: float = 0.20   # 最低維持率（これを下回ると追証。楽天証券 20%）
    use_breadth: bool = True
    # 損切りの執行方式（原因2の検証）: intraday=ザラ場の逆指値（ルール表X1・X2）/ close=終値が損切り価格以下なら翌寄りで決済
    stop_mode: str = "intraday"
    disaster_atr: float = 0.0
    # Setup A の MA25割れ決済（4_決済損切り X6）の扱い（原因3の検証）
    #   ma25: 終値がMA25割れ（ルール表どおり） / ma25_atr: 終値が MA25−a_exit_atr×ATR 割れ / swing_low: MA25割れを使わず、終値が直近スイング安値割れ
    a_exit_mode: str = "ma25"
    a_exit_atr: float = 1.0  # stop_mode=close のとき、損切り価格−この倍率×ATR に置く緊急逆指値（0=置かない）


@dataclass
class RBTrade(Trade):
    setup: str = ""
    risk_per_unit: float = 0.0
    initial_units: int = 0
    exits: list = field(default_factory=list)

    @property
    def r_multiple(self) -> float:
        risk = self.risk_per_unit * self.initial_units
        return self.pnl / risk if risk else np.nan


def ichimoku_cloud(df: pd.DataFrame, p: RulebookParams) -> tuple[pd.Series, pd.Series, pd.Series]:
    hl = lambda n: (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2
    tenkan, kijun = hl(p.tenkan), hl(p.kijun)
    span_a = ((tenkan + kijun) / 2).shift(p.kijun - 1)
    span_b = hl(p.senkou_b).shift(p.kijun - 1)
    return kijun, np.maximum(span_a, span_b), np.minimum(span_a, span_b)


def compute_features(df: pd.DataFrame, p: RulebookParams, nt: pd.Series | None, breadth: pd.Series | None) -> pd.DataFrame:
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    f["ma5"], f["ma25"], f["ma75"], f["ma200"] = (ind.sma(c, n) for n in (p.ma_short, p.ma_mid, p.ma_long, p.ma_ultra))
    f["ma25_5ago"] = f["ma25"].shift(5)
    f["kijun"], f["cloud_top"], f["cloud_bot"] = ichimoku_cloud(df, p)
    f["bb_mid"] = ind.sma(c, p.bb_n)
    f["bb_lo"] = ind.bb_lower(c, p.bb_n, p.bb_k)
    f["bb_hi"] = ind.bb_upper(c, p.bb_n, p.bb_k)
    f["atr"] = ind.atr(df["high"], df["low"], c, p.atr_n)
    f["hh_prev"] = df["high"].shift(1).rolling(p.breakout_n).max()  # 直近N日高値（前日まで）
    f["low5"] = df["low"].rolling(5).min()
    # スイング: 直近N日の最高値と、その日より前のN日間の最安値
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    n = p.swing_n
    sh = np.full(len(df), np.nan)
    sl = np.full(len(df), np.nan)
    for i in range(2 * n, len(df)):
        w = hi[i - n + 1:i + 1]
        j = i - n + 1 + int(np.argmax(w))
        sh[i] = hi[j]
        sl[i] = lo[j - n:j].min()
    f["swing_hi"], f["swing_lo"] = sh, sl
    rng = f["swing_hi"] - f["swing_lo"]
    f["fib382"] = f["swing_hi"] - p.fib_shallow * rng
    f["fib618"] = f["swing_hi"] - p.fib_deep * rng
    f["nt"] = nt.reindex(df.index).ffill() if nt is not None else np.nan
    f["nt_ma"] = f["nt"].rolling(p.nt_ma).mean() if nt is not None else np.nan
    f["breadth"] = breadth.reindex(df.index) if breadth is not None else np.nan

    up = (c > f["ma25"]) & (f["ma25"] > f["ma75"]) & (c > f["cloud_top"]) & (f["ma25"] > f["ma25_5ago"])
    down = (c < f["ma25"]) & (f["ma25"] < f["ma75"]) & (c < f["cloud_bot"])
    f["env"] = np.where(up, "上昇", np.where(down, "下降", "レンジ・移行"))
    return f


def event_block_days(index: pd.DatetimeIndex, events: pd.DataFrame | None) -> pd.Series:
    """各営業日が『重要イベントの前営業日』かどうか（理由の文字列。該当なしは空文字）。"""
    out = pd.Series("", index=index)
    pos = {d: i for i, d in enumerate(index)}

    def last_on_or_before(d):
        k = index.searchsorted(d, side="right") - 1
        return k if k >= 0 else None

    def mark(k, why):
        if k is not None and 0 <= k < len(index):
            out.iloc[k] = (out.iloc[k] + "・" + why).strip("・")

    # SQ: 毎月第2金曜（休業日なら前営業日）
    months = pd.period_range(index[0], index[-1], freq="M")
    for m in months:
        first = m.start_time
        fri = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 7)
        k = last_on_or_before(fri)
        if k is not None and index[k].month == fri.month:
            mark(k - 1, "SQ前日")
    if events is not None:
        for d, typ in zip(events["date"], events["type"]):
            if typ == "BOJ":
                k = index.searchsorted(d, side="left") - 1  # 会合最終日の前営業日
                mark(k, "日銀前")
            elif typ in ("FOMC", "CPI"):
                mark(last_on_or_before(d), f"{typ}前")
    return out


def load_events(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else D.DATA_DIR / "events.csv"
    ev = pd.read_csv(path, comment="#")
    ev["date"] = pd.to_datetime(ev["date"])
    return ev


def load_breadth(path: str | Path) -> pd.Series:
    """騰落レシオCSV（列: 日付, 騰落レシオ(%)）。列名は date/日付 と breadth/騰落レシオ/ratio を受け付ける。"""
    raw = None
    for enc in ("utf-8-sig", "cp932"):
        try:
            raw = pd.read_csv(path, encoding=enc, comment="#")
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raise ValueError(f"騰落レシオCSVを読めません: {path}")
    cols = {c: str(c).strip().lower() for c in raw.columns}
    raw = raw.rename(columns=cols)
    if {"series", "value", "date"} <= set(raw.columns):  # touraku_scraper.py の chart_series_raw.csv 形式
        raw = raw[raw["series"] == "25日間"][["date", "value"]]
    dcol = next((c for c in raw.columns if c in ("date", "日付", "年月日")), raw.columns[0])
    vcol = next((c for c in raw.columns if c != dcol and any(k in c for k in ("騰落", "breadth", "ratio", "レシオ"))), raw.columns[1])
    s = pd.Series(pd.to_numeric(raw[vcol].astype(str).str.replace("%", "").str.replace(",", ""), errors="coerce").values,
                  index=pd.to_datetime(raw[dcol]))
    return s.dropna().sort_index()


class _Pos:
    def __init__(self, setup, units, entry, stop, i):
        self.setup, self.units, self.entry, self.stop, self.i = setup, units, entry, stop, i
        self.r = entry - stop
        self.reached1r = False
        self.partial_done = False
        self.halved_for_event = False
        self.peak = entry
        self.pending_exit = ""
        self.atr = np.nan


def run_rulebook(df: pd.DataFrame, p: RulebookParams, nt: pd.Series | None = None,
                 breadth: pd.Series | None = None, events: pd.DataFrame | None = None,
                 dividends: pd.Series | None = None, lots: pd.Series | None = None,
                 start: str | None = None, extra_equity: pd.Series | None = None,
                 collateral: pd.Series | None = None, b_chase: pd.Series | None = None,
                 hot_override: pd.Series | None = None, notional_cap: pd.Series | None = None) -> tuple[Result, dict]:
    """extra_equity: 口座のうちこのシミュレーション以外の資産の増減（買い持ち部分の評価額−初期資金）。
    口数計算・損失上限は「口座全体の資産」で行う。collateral: 委託保証金に算入できる代用評価額（掛目適用後）。"""
    f = compute_features(df, p, nt, breadth if p.use_breadth else None)
    # hot_override: シグナル日ごとの True/False。True の日は騰落レシオ過熱（E5）でも新規を見送らない
    f["hot_ok"] = hot_override.reindex(df.index).fillna(False).astype(bool) if hot_override is not None else False
    block = event_block_days(df.index, events) if p.use_events else pd.Series("", index=df.index)
    if start:
        keep = df.index >= pd.Timestamp(start)
        df, f, block = df[keep], f[keep], block[keep]
    lots = (lots if lots is not None else pd.Series(1, index=df.index)).reindex(df.index).ffill().bfill().astype(int)
    div = (dividends.groupby(dividends.index).sum().reindex(df.index).fillna(0.0)
           if dividends is not None and len(dividends) else pd.Series(0.0, index=df.index))

    ext = extra_equity.reindex(df.index).ffill().fillna(0.0).to_numpy() if extra_equity is not None else np.zeros(len(df))
    coll = collateral.reindex(df.index).ffill().fillna(0.0).to_numpy() if collateral is not None else None
    # b_chase: 発注日ごとの True/False。True の日は Setup B の新規を指値でなく寄付成行で買う（寄り前に分かる情報で決める）
    chase = b_chase.reindex(df.index).fillna(False).astype(bool).to_numpy() if b_chase is not None else np.zeros(len(df), bool)
    total_interest = 0.0
    # notional_cap: 日ごとの新規建玉金額の上限（円）。コアとの合計上限・信用建余力から外側で計算して渡す
    ncap = notional_cap.reindex(df.index).ffill().fillna(np.inf).to_numpy() if notional_cap is not None else None
    min_ratio, call_days, cap_limited = np.inf, 0, 0
    O, H, L, C = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    dates = df.index
    n = len(df)
    cash = p.capital
    pos: _Pos | None = None
    trade: RBTrade | None = None
    trades: list[RBTrade] = []
    order = None  # 翌日の指値注文
    streak = 0
    block_until = -1
    month_block = None
    month_start_eq = p.capital
    prev_eq = p.capital + (ext[0] if len(ext) else 0.0)
    total_cost = 0.0
    total_div = 0.0
    counts = {"signals": {"A": 0, "B": 0, "C": 0}, "skipped_stop_width": 0, "skipped_event": 0,
              "skipped_loss_limit": 0, "skipped_breadth": 0, "unfilled": 0, "skipped_size0": 0}
    rows = []

    def sell(units, price, i, why, market=True):
        nonlocal cash, pos, trade, total_cost, streak
        px = price * (1 - p.slippage_rate) if market else price
        comm = units * px * p.commission_rate
        total_cost += comm + (units * (price - px))
        cash += units * px - comm
        gross = (px - pos.entry) * units
        trade.pnl += gross - comm
        trade.cost += comm
        trade.exits.append((dates[i], units, px, why))
        pos.units -= units
        if pos.units == 0:
            tot = sum(u for _, u, _, _ in trade.exits)
            trade.exit_price = sum(u * x for _, u, x, _ in trade.exits) / tot
            trade.exit_date = dates[i]
            trade.exit_reason = why
            trades.append(trade)
            streak = 0 if trade.pnl > 0 else streak + 1
            pos, trade = None, None

    for i in range(n):
        d = dates[i]
        if d.to_period("M") != (dates[i - 1].to_period("M") if i else None):
            month_start_eq = prev_eq
            month_block = None
        # 分配金（権利落ち日に保有していれば）
        if pos and div.iloc[i]:
            amt = pos.units * div.iloc[i]
            cash += amt
            total_div += amt
            trade.pnl += amt
        # 信用買いの金利（暦日ベース。土日祝も日割りで発生）
        if pos and p.margin_rate and i > 0:
            days = (d - dates[i - 1]).days
            intr = pos.units * pos.entry * p.margin_rate * days / 365
            cash -= intr
            total_interest += intr
            trade.pnl -= intr

        # ① 寄り：前日終値で決まった決済、イベント前の建玉半減
        if pos and pos.pending_exit:
            sell(pos.units, O[i], i, pos.pending_exit)
        if pos and p.event_halving and block.iloc[i] and not pos.halved_for_event and pos.stop < pos.entry:
            half = int(pos.units * 0.5 // lots.iloc[i] * lots.iloc[i])
            if half > 0:
                pos.halved_for_event = True
                sell(half, O[i], i, f"イベント前半減({block.iloc[i]})")

        # ② 指値の約定判定
        held_from_before = pos is not None
        if order and pos is None:
            if p.event_entry_block and block.iloc[i]:
                counts["skipped_event"] += 1
                _log_skip(counts, d, order, "イベント前")
            elif i <= block_until or month_block == d.to_period("M"):
                counts["skipped_loss_limit"] += 1
                _log_skip(counts, d, order, "損失上限")
            else:
                lim = order["limit"]
                if order["setup"] == "B" and chase[i]:
                    fill = O[i] * (1 + p.slippage_rate)  # 寄付成行
                    counts["chased"] = counts.get("chased", 0) + 1
                else:
                    fill = O[i] if O[i] <= lim else (lim if L[i] <= lim else None)
                if fill is None:
                    counts["unfilled"] += 1
                    _log_skip(counts, d, order, "指値不成立")
                else:
                    equity = cash + ext[i]
                    risk_yen = equity * p.risk_per_trade * order["factor"]
                    lot = lots.iloc[i]
                    per = fill - order["stop"]
                    units = int(risk_yen / per // lot * lot) if per > 0 else 0
                    cap = int(equity * p.position_cap / fill // lot * lot)
                    units = min(units, cap)
                    if ncap is not None:
                        lim_units = int(max(ncap[i], 0) / fill // lot * lot)
                        if lim_units < units:
                            cap_limited += 1
                            units = lim_units
                    if coll is not None:  # 委託保証金（代用評価額＋信用の実現損益）の範囲で建てる
                        guarantee = coll[i] + (cash - p.capital)
                        mcap = int(max(guarantee, 0) / (p.margin_requirement * fill) // lot * lot)
                        if mcap < units:
                            cap_limited += 1
                            units = mcap
                    if units <= 0:
                        counts["skipped_size0"] += 1
                        _log_skip(counts, d, order, "口数0")
                    else:
                        comm = units * fill * p.commission_rate
                        cash -= units * fill + comm
                        if p.margin_rate:  # 両端入れのため建てた日の1日分
                            intr = units * fill * p.margin_rate / 365
                            cash -= intr
                            total_interest += intr
                        total_cost += comm
                        pos = _Pos(order["setup"], units, fill, order["stop"], i)
                        pos.peak = H[i]
                        pos.atr = f["atr"].iloc[i - 1]
                        trade = RBTrade(direction="long", entry_date=d, entry_price=fill, units=units,
                                        setup=order["setup"], risk_per_unit=fill - order["stop"],
                                        initial_units=units, max_units=units, invested=units * fill)
                        trade.cost += comm
                        trade.pnl -= comm
        order = None

        # ③ ザラ場：損切り・部分利確・Setup C 目標
        if pos:
            if p.stop_mode == "intraday":
                level, why = pos.stop, ("建値撤退" if pos.stop >= pos.entry else "初期損切り")
            elif p.disaster_atr > 0:
                level, why = pos.stop - p.disaster_atr * pos.atr, "緊急損切り"
            else:
                level = None
            if level is not None and L[i] <= level:
                gap = O[i] <= level  # 寄りで損切り価格を下回った（新規約定日も含む）
                sell(pos.units, O[i] if gap else level, i, why)
        if pos:
            if pos.setup == "C":
                tgt = min(f["bb_mid"].iloc[i - 1], f["ma25"].iloc[i - 1])  # 前日終値時点の水準（先に届く方）
                if tgt > pos.entry and H[i] >= tgt:
                    sell(pos.units, max(O[i], tgt) if held_from_before else tgt, i, "目標到達", market=False)
            if pos and not pos.partial_done:
                tp = pos.entry + p.partial_r * pos.r
                if H[i] >= tp:
                    k = int(pos.units * p.partial_ratio // lots.iloc[i] * lots.iloc[i])
                    pos.partial_done = True
                    if k > 0:
                        sell(k, max(O[i], tp) if held_from_before else tp, i, "部分利確", market=False)
        if pos:
            if H[i] >= pos.entry + pos.r:
                pos.reached1r = True
            pos.peak = max(pos.peak, H[i])

        # ④ 終値判定 → 翌寄りの決済予約
        if pos:
            held = i - pos.i
            atr = f["atr"].iloc[i]
            if p.stop_mode == "close" and C[i] <= pos.stop:
                pos.pending_exit = "建値撤退" if pos.stop >= pos.entry else "初期損切り"
            elif pos.setup in ("A", "B"):
                pos.pending_exit = _ab_exit(pos, p, f, i, C[i], atr, held)
            elif held >= p.time_stop_c:
                pos.pending_exit = "時間ストップ"
            if pos.reached1r and pos.stop < pos.entry:
                pos.stop = pos.entry  # X4（翌日から有効）

        # ⑤ 評価・損失上限
        eq = cash + (pos.units * C[i] if pos else 0.0)
        if pos and coll is not None:
            realized = cash + pos.units * pos.entry - p.capital
            unreal = pos.units * (C[i] - pos.entry)
            ratio = (coll[i] + realized + min(unreal, 0.0)) / (pos.units * pos.entry)
            min_ratio = min(min_ratio, ratio)
            call_days += ratio < p.maintenance_ratio
        eq_total = eq + ext[i]
        if eq_total - prev_eq <= -p.daily_loss_limit * prev_eq:
            block_until = i + 1
        if eq_total - month_start_eq <= -p.monthly_loss_limit * month_start_eq:
            month_block = d.to_period("M")
        prev_eq = eq_total
        rows.append((d, eq, cash, pos.units if pos else 0, C[i], (pos.units * C[i] / eq) if pos else 0.0))

        # ⑥ シグナル判定（建玉なしのときのみ）→ 翌日の指値
        if pos is None and i < n - 1:
            order = _signal(i, df, f, p, streak, counts)

    if trade is not None:
        trade.pnl += (C[-1] - pos.entry) * pos.units
        trade.exit_date, trade.exit_price = dates[-1], C[-1]
        trade.exit_reason = "期間終了（未決済・評価）"
        trades.append(trade)

    eq = pd.DataFrame(rows, columns=["date", "equity", "cash", "units", "close", "exposure"]).set_index("date")
    from .engine import ExecutionConfig, buy_and_hold
    bench = buy_and_hold(df, p.capital, lots.to_numpy(), ExecutionConfig(commission_rate=p.commission_rate), div)
    res = Result(equity=eq, trades=trades, benchmark=bench, initial_capital=p.capital,
                 total_cost=total_cost, total_dividends=total_div)
    res.price = df
    counts.update({"total_interest": total_interest, "margin_min_ratio": (min_ratio if np.isfinite(min_ratio) else np.nan),
                   "margin_call_days": int(call_days), "margin_cap_limited": cap_limited})
    return res, counts


def _ab_exit(pos, p: RulebookParams, f, i, close, atr, held) -> str:
    """Setup A/B の終値ベースの決済判定（X6・B4・X8）。該当なしは空文字。"""
    if close < pos.peak - p.trail_atr * atr:
        return "トレーリング"
    ma25 = f["ma25"].iloc[i]
    if pos.setup == "B" or p.a_exit_mode == "ma25":
        if close < ma25:
            return "MA25割れ"
    elif p.a_exit_mode == "ma25_atr":
        if close < ma25 - p.a_exit_atr * atr:
            return "MA25−ATR割れ"
    elif p.a_exit_mode == "swing_low":
        if close < f["swing_lo"].iloc[i]:
            return "スイング安値割れ"
    else:
        raise ValueError(f"a_exit_mode が不正です: {p.a_exit_mode}")
    if pos.setup == "B" and close < f["cloud_top"].iloc[i]:
        return "雲上限割れ"
    if held >= p.time_stop_ab and not pos.reached1r:
        return "時間ストップ"
    return ""


def _log_skip(counts: dict, date, order: dict, reason: str, signal_day: bool = False):
    """見送ったシグナルの記録（分析用）。date はシグナル日（signal_day=True）または発注予定日。"""
    counts.setdefault("skips", []).append({"date": date, "signal_day": signal_day, "setup": order["setup"],
                                           "limit": float(order["limit"]), "stop": float(order["stop"]), "reason": reason})


def _signal(i, df, f, p: RulebookParams, streak: int, counts: dict):
    r = f.iloc[i]
    o, h, l, c = (df[k].iloc[i] for k in ("open", "high", "low", "close"))
    if any(pd.isna(r[k]) for k in ("ma200", "cloud_top", "atr", "swing_hi", "kijun", "hh_prev")):
        return None
    atr = r["atr"]
    breadth = r["breadth"]
    hot = pd.notna(breadth) and breadth >= p.breadth_hot
    cold = pd.notna(breadth) and breadth <= p.breadth_cold
    factor = 1.0
    if c < r["ma200"]:
        factor *= 0.5
    if streak >= p.losing_streak:
        factor *= p.streak_factor

    cands = []
    # Setup A
    if r["env"] == "上昇":
        in_zone = (abs(c - r["ma25"]) <= p.near_atr * atr or abs(c - r["kijun"]) <= p.near_atr * atr
                   or r["fib618"] <= c <= r["fib382"])
        if in_zone:
            stop = r["fib618"] - p.tick
            if stop >= c:
                stop = r["swing_lo"] - p.tick
            cands.append(("A", c, stop))
    # Setup B
    nt_ok = pd.isna(r["nt"]) or pd.isna(r["nt_ma"]) or r["nt"] >= r["nt_ma"]
    if r["env"] in ("上昇", "レンジ・移行") and nt_ok and c > r["cloud_top"] and c > r["hh_prev"]:
        cands.append(("B", r["hh_prev"] + p.b_limit_atr * atr, r["hh_prev"] - p.b_stop_atr * atr))
    # Setup C（C1 が当日または前日に成立 + 当日のローソク足で確認）
    if pd.notna(breadth):
        prev = f.iloc[i - 1]
        c1_today = cold and c <= r["bb_lo"]
        c1_prev = (pd.notna(prev["breadth"]) and prev["breadth"] <= p.breadth_cold
                   and df["close"].iloc[i - 1] <= prev["bb_lo"])
        body = abs(c - o)
        lower = min(o, c) - l
        confirm = c > o or (lower >= 2 * body and lower >= 0.5 * (h - l) and h > l)
        if (c1_today or c1_prev) and confirm:
            cands.append(("C", c, r["low5"] - p.tick))

    for setup, limit, stop in cands:  # A > B > C
        counts["signals"][setup] += 1
        if setup in ("A", "B") and hot and p.breadth_hot_filter and not r["hot_ok"]:
            counts["skipped_breadth"] += 1
            _log_skip(counts, df.index[i], {"setup": setup, "limit": limit, "stop": stop}, "騰落レシオ過熱", signal_day=True)
            continue
        width = (limit - stop) / atr
        if not (p.stop_min_atr - 1e-9 <= width <= p.stop_max_atr + 1e-9):  # 境界ちょうどは許容（浮動小数の誤差対策）
            counts["skipped_stop_width"] += 1
            _log_skip(counts, df.index[i], {"setup": setup, "limit": limit, "stop": stop},
                      f"損切り幅{width:.2f}ATR（2〜3ATR外）", signal_day=True)
            continue
        fac = factor * (0.5 if setup == "C" else 1.0)
        return {"setup": setup, "limit": float(limit), "stop": float(stop), "factor": max(min(fac, 1.0), 0.5)}
    return None


def summarize(res: Result) -> dict:
    """9_検証集計 に対応する集計（R倍数・セットアップ別・決済理由別）。"""
    closed = [t for t in res.trades if not t.exit_reason.startswith("期間終了")]
    df = pd.DataFrame([{
        "setup": t.setup, "pnl": t.pnl, "r": t.r_multiple, "reason": t.exit_reason,
        "days": np.busday_count(t.entry_date.date(), t.exit_date.date()),
        "entry": t.entry_date, "partial": any(x[3] == "部分利確" for x in t.exits),
    } for t in closed])
    if df.empty:
        return {"trades": df}
    by_setup = df.groupby("setup").agg(件数=("pnl", "size"), 勝率=("pnl", lambda s: (s > 0).mean()),
                                       平均R=("r", "mean"), 純損益=("pnl", "sum"))
    by_reason = df.groupby("reason").agg(件数=("pnl", "size"), 平均R=("r", "mean"), 純損益=("pnl", "sum"))
    return {"trades": df, "by_setup": by_setup, "by_reason": by_reason,
            "avg_r": df["r"].mean(), "median_r": df["r"].median(), "partial_rate": df["partial"].mean()}
