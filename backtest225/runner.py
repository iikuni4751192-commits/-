"""設定ファイル（YAML）を読み、データ取得→シグナル生成→バックテストまでを実行する。"""
from __future__ import annotations

import ast
import itertools
import operator
from dataclasses import fields
from pathlib import Path

import pandas as pd
import yaml

from . import data as D
from .engine import ExecutionConfig, PositionConfig, Result, RiskConfig, run_backtest
from .metrics import compute
from .strategy import build_signals


def load_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg, path.resolve().parent


def _dc(cls, section: dict | None):
    section = section or {}
    allowed = {f.name for f in fields(cls)}
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"{cls.__name__} に未知の設定があります: {sorted(unknown)}（使える項目: {sorted(allowed)}）")
    return cls(**section)


def prepare_data(cfg: dict, base_dir: Path, overrides: dict | None = None):
    o = overrides or {}
    dcfg = dict(cfg.get("data") or {})
    period = dict(cfg.get("period") or {})
    if o.get("csv"):
        dcfg["source"], dcfg["csv"] = "csv", o["csv"]
    source = dcfg.get("source", "yahoo")
    ticker = dcfg.get("ticker", D.DEFAULT_TICKER)
    csv_path = dcfg.get("csv")
    if csv_path and source == "csv" and not Path(csv_path).is_absolute() and not o.get("csv"):
        csv_path = str(base_dir / csv_path)
    df, notes = D.load_prices(
        source=source, ticker=ticker, csv_path=csv_path,
        start=None, end=o.get("end") or period.get("end"),
        refresh=bool(o.get("refresh")),
    )
    start = o.get("start") or period.get("start")

    div_src = dcfg.get("dividends", "none")
    dividends = None
    if div_src == "builtin":
        if ticker not in D.BUILTIN_DIVIDENDS:
            raise ValueError(f"{ticker} の同梱分配金データはありません（yahoo かCSVを指定）")
        dividends = D.load_dividends_csv(D.BUILTIN_DIVIDENDS[ticker])
        notes.append("分配金: 同梱データ（運用会社公表値 nextfunds.jp より作成）")
    elif div_src == "yahoo":
        dividends = D.fetch_yahoo_dividends(ticker, df)
        notes.append(f"分配金: Yahoo Finance（{len(dividends)}件。非公式データ、口数基準を自動補正）")
    elif div_src and div_src != "none":
        p = Path(div_src)
        dividends = D.load_dividends_csv(p if p.is_absolute() else base_dir / p)
        notes.append(f"分配金: CSV {div_src}（{len(dividends)}件）")
    else:
        notes.append("分配金: 含めない（価格リターンのみ）")
    if dividends is not None and len(dividends):
        dividends = D.to_ex_dates(dividends, df.index)
        notes.append("分配金は決算日から算出した権利落ち日に受け取ったものとして計算（実際の入金は数か月後）")
    return df, start, ticker, dividends, notes


def backtest_from_config(cfg: dict, base_dir: Path, overrides: dict | None = None,
                         params_override: dict | None = None, prepared=None) -> tuple[Result, dict]:
    o = overrides or {}
    df, start, ticker, dividends, notes = prepared or prepare_data(cfg, base_dir, o)
    # 指標はウォームアップのため全期間で計算し、検証開始日以降だけ売買する
    target = build_signals(cfg, df, base_dir, params_override)
    if start:
        mask = df.index >= pd.Timestamp(start)
        df, target = df[mask], target[mask]
        if dividends is not None:
            dividends = dividends[dividends.index >= pd.Timestamp(start)]
    if len(df) < 2:
        raise ValueError("検証期間のデータが2日未満です")
    lots = D.lot_size_series(df.index, ticker, cfg.get("lot_size", "auto"))
    capital = float(o.get("capital") or cfg.get("capital", 1_000_000))
    res = run_backtest(
        df, target, capital=capital, lots=lots,
        execution=_dc(ExecutionConfig, cfg.get("execution")),
        risk=_dc(RiskConfig, cfg.get("risk")),
        position=_dc(PositionConfig, cfg.get("position")),
        dividends=dividends,
    )
    res.notes = list(notes)
    if lots.iloc[0] * df["open"].iloc[0] > capital:
        res.notes.append(
            f"初期資金 {capital:,.0f}円 では最小売買単位（{lots.iloc[0]}口×{df['open'].iloc[0]:,.1f}円）を買えません"
        )
    res.price = df  # レポート用
    return res, compute(res)


# ---- パラメータ最適化 ----

_SCALAR_OPS = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
}


def _scalar_eval(expr: str, names: dict) -> bool:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            return names[n.id]
        if isinstance(n, ast.BinOp):
            return _SCALAR_OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return not ev(n.operand)
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, comp in zip(n.ops, n.comparators):
                right = ev(comp)
                if not _SCALAR_OPS[type(op)](left, right):
                    return False
                left = right
            return True
        raise ValueError(f"constraint に使えない要素: {ast.dump(n)[:60]}")
    return bool(ev(ast.parse(expr, mode="eval")))


def optimize(cfg: dict, base_dir: Path, overrides: dict | None = None,
             split_date: str | None = None) -> tuple[pd.DataFrame, dict | None]:
    """grid の全組合せを検証。split_date を指定すると、それより前（学習期間）で最良の
    パラメータを、それ以降（検証期間）でも試した結果を返す（過剰最適化のチェック用）。"""
    ocfg = cfg.get("optimize") or {}
    grid = ocfg.get("grid")
    if not grid:
        raise ValueError("optimize.grid（パラメータの候補）が設定されていません")
    metric = ocfg.get("metric", "sharpe")
    constraint = ocfg.get("constraint")
    o = dict(overrides or {})
    end_is = None
    if split_date:
        end_is = (pd.Timestamp(split_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    prepared = prepare_data(cfg, base_dir, o)
    if end_is:
        df, start, ticker, div, notes = prepared
        prepared_is = (df[df.index <= pd.Timestamp(end_is)], start, ticker, div, notes)
    else:
        prepared_is = prepared

    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        if constraint and not _scalar_eval(constraint, params):
            continue
        _, m = backtest_from_config(cfg, base_dir, o, params, prepared=prepared_is)
        s = m["strategy"]
        rows.append({**params, "total_return": s["total_return"], "cagr": s["cagr"],
                     "max_drawdown": s["max_drawdown"], "sharpe": s["sharpe"],
                     "profit_factor": s["profit_factor"], "n_trades": s["n_trades"],
                     "win_rate": s["win_rate"]})
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("条件を満たすパラメータの組合せがありません")
    # どの指標も「大きいほど良い」（最大ドローダウンは負値なので0に近いほど大きい）
    table = table.sort_values(metric, ascending=False, na_position="last").reset_index(drop=True)

    oos = None
    if split_date:
        best = {k: table.iloc[0][k] for k in keys}
        best = {k: (int(v) if float(v).is_integer() else float(v)) for k, v in best.items()}
        o_oos = {**o, "start": split_date}
        _, m = backtest_from_config(cfg, base_dir, o_oos, best, prepared=(prepared[0], split_date, *prepared[2:]))
        oos = {"params": best, "metrics": m}
    return table, oos
