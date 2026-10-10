"""YAMLルール用の安全な式評価器。

例:  "cross_up(sma(close, 25), sma(close, 75)) and rsi(close, 14) < 70"

Python の eval は使わず、AST を解析して許可された要素（数値・変数・関数・
算術/比較/論理演算）だけを評価する。各変数・関数は日付ごとの値を持つ pandas Series。
"""
from __future__ import annotations

import ast
import operator

import numpy as np
import pandas as pd

from . import indicators as ind

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMPOPS = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


class ExprError(ValueError):
    pass


def build_namespace(df: pd.DataFrame, params: dict | None = None) -> tuple[dict, dict]:
    """式で使える変数と関数を返す。"""
    idx = df.index
    month_key = idx.year * 100 + idx.month
    mk = pd.Series(month_key, index=idx)
    variables = {
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["close"],
        "volume": df["volume"],
        "weekday": pd.Series(idx.weekday, index=idx),  # 0=月 ... 4=金
        "month": pd.Series(idx.month, index=idx),
        "day": pd.Series(idx.day, index=idx),
        "year": pd.Series(idx.year, index=idx),
        # 月初/月末の営業日（月末判定は翌営業日の月で決まるが、カレンダー上の事実なので未来情報ではない）
        "month_first_day": mk != mk.shift(1),
        # データ最終日は翌営業日が不明なので False
        "month_last_day": (mk != mk.shift(-1)) & mk.shift(-1).notna(),
        "true": pd.Series(True, index=idx),
        "false": pd.Series(False, index=idx),
    }
    for k, v in (params or {}).items():
        if k in variables:
            raise ExprError(f"パラメータ名 '{k}' は組み込み変数と重複しています")
        variables[k] = v

    def _atr(n=14):
        return ind.atr(df["high"], df["low"], df["close"], n)

    def _ew_max(a, b):
        return np.maximum(a, b)

    def _ew_min(a, b):
        return np.minimum(a, b)

    functions = {
        name: getattr(ind, name)
        for name in [
            "sma", "ema", "std", "rsi", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "highest", "lowest", "roc", "deviation",
            "ref", "cross_up", "cross_down", "rising", "falling",
        ]
    }
    functions.update({"atr": _atr, "abs": abs, "max": _ew_max, "min": _ew_min})
    return variables, functions


def evaluate(expr: str, df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """式を評価し、日付ごとの値（通常は bool）の Series を返す。"""
    variables, functions = build_namespace(df, params)
    try:
        tree = ast.parse(str(expr).strip(), mode="eval")
    except SyntaxError as e:
        raise ExprError(f"式の構文エラー: {expr!r} ({e.msg})") from e

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise ExprError(f"未定義の変数: {node.id}（使える変数: {', '.join(sorted(variables))}）")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return v
            if isinstance(node.op, ast.Not):
                return ~_as_bool(v, df.index)
        if isinstance(node, ast.BoolOp):
            vals = [_as_bool(ev(v), df.index) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out & v) if isinstance(node.op, ast.And) else (out | v)
            return out
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            result = None
            for op, comp in zip(node.ops, node.comparators):
                if type(op) not in _CMPOPS:
                    raise ExprError(f"未対応の比較演算子: {type(op).__name__}")
                right = ev(comp)
                r = _CMPOPS[type(op)](left, right)
                result = r if result is None else (result & r)
                left = right
            return result
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = functions.get(node.func.id)
            if fn is None:
                raise ExprError(
                    f"未定義の関数: {node.func.id}（使える関数: {', '.join(sorted(functions))}）"
                )
            if node.keywords:
                kwargs = {k.arg: ev(k.value) for k in node.keywords}
            else:
                kwargs = {}
            return fn(*[ev(a) for a in node.args], **kwargs)
        raise ExprError(f"式に使えない要素があります: {ast.dump(node)[:80]}")

    out = ev(tree)
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=df.index)
    return out


def _as_bool(v, index) -> pd.Series:
    if not isinstance(v, pd.Series):
        v = pd.Series(v, index=index)
    if v.dtype != bool:
        v = v.fillna(False).astype(bool)
    return v
