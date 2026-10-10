"""投資ルール（ストラテジー）の読み込みと売買シグナル生成。

シグナルは「target（目標ポジション比率）」の Series に統一する。
  - 1.0  : 資産の100%を買い持ち
  - 0.5  : 50%
  - 0.0  : 全て手仕舞い
  - -1.0 : 100%空売り（direction: short の場合）
  - NaN  : 何もしない（現状維持）
シグナルはその日の終値確定後に判定され、エンジン側で翌営業日に約定する（既定）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .expr import evaluate


def entries_exits_to_target(entry: pd.Series, exit: pd.Series, weight: float = 1.0) -> pd.Series:
    entry = entry.fillna(False).astype(bool)
    exit = exit.fillna(False).astype(bool)
    target = pd.Series(np.nan, index=entry.index)
    target[entry & ~exit] = weight
    target[exit] = 0.0  # 同日に両方成立した場合は手仕舞いを優先
    return target


def rule_signals(df: pd.DataFrame, rules: dict, params: dict | None = None) -> pd.Series:
    """YAMLの rules セクションから target を作る。

    rules:
      entry: "式"            # 新規エントリー条件
      exit:  "式"            # 手仕舞い条件（省略時は損切り等のリスク管理のみで決済）
      direction: long|short
      weight: 1.0            # 資産に対する投資比率
    """
    if "entry" not in rules:
        raise ValueError("rules.entry（エントリー条件）が必要です")
    direction = rules.get("direction", "long")
    if direction not in ("long", "short"):
        raise ValueError("rules.direction は long または short")
    weight = float(rules.get("weight", 1.0))
    sign = 1.0 if direction == "long" else -1.0
    entry = evaluate(rules["entry"], df, params)
    exit_expr = rules.get("exit")
    exit = evaluate(exit_expr, df, params) if exit_expr else pd.Series(False, index=df.index)
    return entries_exits_to_target(entry, exit, sign * weight)


def load_python_strategy(path: str | Path) -> tuple[Callable, dict]:
    """Python製ストラテジーを読み込む。ファイルには signals(df, params) と任意の PARAMS を定義する。"""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"user_strategy_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"ストラテジーを読み込めません: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "signals"):
        raise AttributeError(f"{path} に signals(df, params) 関数がありません")
    return mod.signals, dict(getattr(mod, "PARAMS", {}))


def python_signals(fn: Callable, df: pd.DataFrame, params: dict) -> pd.Series:
    """ユーザー関数の戻り値（target Series / entry・exit 列 / target 列）を target に正規化。"""
    out = fn(df.copy(), dict(params))
    if isinstance(out, pd.Series):
        target = out.astype(float)
    elif isinstance(out, pd.DataFrame):
        if "target" in out.columns:
            target = out["target"].astype(float)
        elif "entry" in out.columns:
            exit = out["exit"] if "exit" in out.columns else pd.Series(False, index=out.index)
            w = float(params.get("weight", 1.0))
            target = entries_exits_to_target(out["entry"], exit, w)
        else:
            raise ValueError("signals() の戻り値には target 列 か entry/exit 列が必要です")
    else:
        raise TypeError("signals() は pandas の Series か DataFrame を返してください")
    return target.reindex(df.index)


def build_signals(cfg: dict, df: pd.DataFrame, base_dir: Path, params_override: dict | None = None) -> pd.Series:
    """設定（YAML由来の dict）に従って target を生成する。"""
    params = dict(cfg.get("params") or {})
    if "python" in cfg:
        fn, default_params = load_python_strategy(base_dir / cfg["python"])
        params = {**default_params, **params, **(params_override or {})}
        return python_signals(fn, df, params)
    if "rules" in cfg:
        params.update(params_override or {})
        return rule_signals(df, cfg["rules"], params)
    raise ValueError("設定に rules（式ルール）か python（Pythonファイル）のどちらかが必要です")
