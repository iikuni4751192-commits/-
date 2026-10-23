"""対象銘柄（ユニバース）。

一次情報: 日本取引所グループ「東証上場銘柄一覧」（毎月更新）
  https://www.jpx.co.jp/markets/statistics-equities/misc/01.html
規模区分（TOPIX Core30 / Large70 / Mid400 / Small 1 / Small 2）で範囲を選ぶ。

注意: 一覧は「取得時点で上場している銘柄」の「取得時点の規模区分」。過去の検証に使うと、
その後に上場廃止・指数除外された銘柄が入らないため、成績が実際より良く見えやすい（生存者バイアス）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
UNIVERSE_CSV = Path(__file__).resolve().parent.parent / "data" / "universe_jpx.csv"

SCOPES: dict[str, list[str]] = {
    "topix100": ["TOPIX Core30", "TOPIX Large70"],
    "topix500": ["TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"],
    "topix1000": ["TOPIX Core30", "TOPIX Large70", "TOPIX Mid400", "TOPIX Small 1"],
}


def fetch_jpx_list(out: Path = UNIVERSE_CSV) -> pd.DataFrame:
    """JPXの上場銘柄一覧を取得し、内国株式だけをCSVに保存する。"""
    import io
    import urllib.request

    req = urllib.request.Request(JPX_LIST_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = pd.read_excel(io.BytesIO(r.read()), dtype=str)
    df = raw.rename(columns={
        "日付": "list_date", "コード": "code", "銘柄名": "name", "市場・商品区分": "market",
        "33業種区分": "sector", "規模区分": "size",
    })[["list_date", "code", "name", "market", "sector", "size"]]
    df = df[df["market"].str.contains("内国株式", na=False)].reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    return df


def load_universe(scope: str = "topix500", path: Path = UNIVERSE_CSV) -> pd.DataFrame:
    """scope: topix100 / topix500 / topix1000 / prime（プライム全銘柄） / ティッカー列挙CSVのパス。

    戻り値の列: ticker（例 7203.T）, code, name, sector, size
    """
    p = Path(scope)
    if p.suffix == ".csv" and p.exists():
        df = pd.read_csv(p, dtype=str)
        if "ticker" not in df.columns:
            df["ticker"] = df["code"].str.strip() + ".T"
        return df
    if not path.exists():
        raise FileNotFoundError(f"{path} がありません。先に `python -m weekpick universe` を実行してください")
    df = pd.read_csv(path, dtype=str)
    if scope == "prime":
        df = df[df["market"].str.startswith("プライム")]
    elif scope in SCOPES:
        df = df[df["size"].isin(SCOPES[scope])]
    else:
        raise ValueError(f"未知のユニバース: {scope}（{', '.join([*SCOPES, 'prime'])} または CSV パス）")
    df = df.copy()
    df["ticker"] = df["code"].str.strip() + ".T"
    return df.reset_index(drop=True)
