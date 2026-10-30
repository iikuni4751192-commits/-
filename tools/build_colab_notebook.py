"""SOMPO 週末シグナル確認ノートブックの Google Colab 版（ファイル1つで動く版）を作る。

notebooks/weekly_signal_sompo.ipynb の本文に、計算部分（weekpick・backtest225.data）と
銘柄一覧（TOPIX500 相当）を埋め込んだ「準備」セルを先頭に足して書き出す。
weekpick を変更したら、このスクリプトを実行し直して Colab 版を作り直すこと。

    python tools/build_colab_notebook.py
"""
from __future__ import annotations

import copy
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "notebooks" / "weekly_signal_sompo.ipynb"
OUT_NB = ROOT / "notebooks" / "weekly_signal_sompo_colab.ipynb"
MODULES = [
    "weekpick/__init__.py", "weekpick/universe.py", "weekpick/data.py", "weekpick/factors.py",
    "weekpick/backtest.py", "weekpick/timing.py", "backtest225/__init__.py", "backtest225/data.py",
]
LIB = "weekpick_colab"  # Colab 上で計算部分とデータを置くフォルダ

INTRO = """# 【Colab版】週末のシグナル確認：SOMPO ホールディングス（8630）／ウィリアムズ%R(14) < -80

**このファイル1つで動く Google Colab 用の版です。** 計算に使うプログラムと銘柄一覧（TOPIX500 相当、2026/9/30 時点の JPX 一覧）を、下の「0. 準備」セルに埋め込んであります。

### 使い方
1. [Google Colab](https://colab.research.google.com/) を開き、「ファイル」→「ノートブックをアップロード」でこのファイルを選ぶ
2. 金曜の取引終了後〜日曜に、メニューの「ランタイム」→「すべてのセルを実行」
3. 「3. 今週末の判定」に、シグナルの有無と同時に出ている条件が表示される

- 毎回、全約490銘柄の株価を Yahoo Finance から取り直すため、全体で5〜10分ほどかかります（Colab はセッションが終わるとデータが消えるため）。
- 「0. 準備」セルは変更しないでください。設定は「1. 設定」で変えられます。
"""

SETUP_PREP = '''# 0. 準備（このセルは変更しないでください）
# 計算部分のプログラムと銘柄一覧をこの環境に書き出し、必要なライブラリと日本語フォントを用意する。
import subprocess
import sys
from pathlib import Path

LIB = Path("{lib}")
SOURCES = {sources}
UNIVERSE_CSV = {universe}

for rel, text in SOURCES.items():
    f = LIB / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")
(LIB / "data").mkdir(exist_ok=True)
(LIB / "data" / "universe_jpx.csv").write_text(UNIVERSE_CSV, encoding="utf-8")
if str(LIB.resolve()) not in sys.path:
    sys.path.insert(0, str(LIB.resolve()))

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "yfinance"], check=False)
    subprocess.run("apt-get -qq install -y fonts-ipafont-gothic > /dev/null", shell=True, check=False)
    from matplotlib import font_manager
    for p in Path("/usr/share/fonts").rglob("ipag*.ttf"):
        font_manager.fontManager.addfont(str(p))
print("準備完了:", LIB.resolve())'''

SETTINGS = '''TICKER = "8630.T"
NAME = "SOMPOホールディングス"
REFRESH = True        # 最新の株価を取り直す（Colab では毎回 True のままにする）
UPDATE_UNIVERSE = False  # True にすると JPX から最新の銘柄一覧を取り直す（通常は不要）
CAPITAL = 500_000     # 信用の保証金（円）。レバレッジ1倍で100株単位の株数を計算する
INTEREST = 0.028      # 信用の買方金利（年率）。お使いの証券会社の料率に合わせる
ENTRY_PRICE = None    # 寄りで約定した後に損切りの目安を出し直すときは買値を入れる（例: 6900.0）
RECENT_YEARS = 3      # 過去シグナル一覧に表示する年数
CHECK_CONDITIONS = True  # 5. の条件の検証（全銘柄で集計するため数分かかる）'''


def build() -> Path:
    nb = nbformat.read(SRC_NB, as_version=4)
    sources = {m: (ROOT / m).read_text(encoding="utf-8") for m in MODULES}
    uni = pd.read_csv(ROOT / "data" / "universe_jpx.csv", dtype=str)
    uni = uni[uni["size"].isin(["TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"])]
    prep = SETUP_PREP.format(lib=LIB, sources=repr(sources), universe=repr(uni.to_csv(index=False)))

    cells = copy.deepcopy(nb.cells)
    cells[0] = nbformat.v4.new_markdown_cell(INTRO + "\n---\n\n" + cells[0].source.split("\n", 1)[1].lstrip())
    i_settings = next(i for i, c in enumerate(cells) if c.cell_type == "code" and "TICKER =" in c.source)
    cells[i_settings] = nbformat.v4.new_code_cell(SETTINGS)
    i_load = next(i for i, c in enumerate(cells) if c.cell_type == "code" and "load_universe(\"topix500\")" in c.source)
    cells[i_load] = nbformat.v4.new_code_cell(cells[i_load].source.replace(
        'u = load_universe("topix500")',
        'if UPDATE_UNIVERSE:\n    from weekpick.universe import fetch_jpx_list\n    fetch_jpx_list()\nu = load_universe("topix500")'))
    i_md2 = next(i for i, c in enumerate(cells) if c.cell_type == "markdown" and c.source.startswith("## 2."))
    cells[i_md2] = nbformat.v4.new_markdown_cell(
        "## 2. データ更新\n\n市場全体の値動き・同業種の状況を計算するため、TOPIX500 相当の全銘柄を読み込みます。"
        "銘柄一覧は 2026/9/30 時点の JPX 一覧です。古くなったら、1. の `UPDATE_UNIVERSE` を True にしてください。")
    cells = [cells[0], nbformat.v4.new_markdown_cell("## 0. 準備"), nbformat.v4.new_code_cell(prep)] + cells[1:]
    out = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}})
    nbformat.write(out, OUT_NB)
    return OUT_NB


if __name__ == "__main__":
    print(build())
