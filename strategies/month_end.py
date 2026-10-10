"""例4: Pythonで書くルール（アノマリー: 月末に買い、月初数日で売る）

signals(df, params) を定義し、次のどちらかを返す:
  - DataFrame(entry=bool, exit=bool)      … 条件成立日に新規/決済
  - Series（目標ポジション比率 1.0/0.5/0/-1, NaN=現状維持）
df の列: open, high, low, close, volume（日付インデックス）。
その日の終値までの情報だけを使うこと（shift(-1) など未来を見る処理は禁止）。
"""
import pandas as pd

from backtest225 import indicators as ind  # sma, ema, rsi, macd ... が使える

PARAMS = {"hold_days": 3, "trend": 0}


def signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    month = pd.Series(df.index.year * 100 + df.index.month, index=df.index)
    # 月末営業日の判定は翌営業日の日付から決まる（営業日カレンダーは事前にわかるため未来情報ではない）
    is_month_end = (month != month.shift(-1)) & month.shift(-1).notna()  # データ最終日は判定不能なので False
    days_in_month = month.groupby(month).cumcount() + 1  # 月初から何営業日目か

    entry = is_month_end
    if params["trend"]:
        entry &= df["close"] > ind.sma(df["close"], params["trend"])
    exit = days_in_month == params["hold_days"]
    return pd.DataFrame({"entry": entry, "exit": exit})
