# NF日経225（1321）投資ルール検証システム

自分で考えた売買ルールを **NEXT FUNDS 日経225連動型上場投信（1321）** の過去の日足データに当てはめ、
どれだけの損益が出たかを検証（バックテスト）するツールです。結果はターミナルとHTMLレポートで確認できます。

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

```bash
# バックテスト実行 → reports/golden_cross.html を作成
python -m backtest225 run strategies/golden_cross.yaml

# 期間・資金を一時的に変更
python -m backtest225 run strategies/golden_cross.yaml --start 2015-01-01 --end 2024-12-30 --capital 3000000

# 手持ちCSVで検証（自動取得しない）
python -m backtest225 run strategies/golden_cross.yaml --csv data/my_1321.csv

# 取引一覧をCSV保存
python -m backtest225 run strategies/golden_cross.yaml --trades-csv reports/trades.csv

# パラメータ総当たり。--split で「学習期間で最良を選び、検証期間で試す」
python -m backtest225 optimize strategies/golden_cross.yaml --split 2020-01-01

# 価格データを再取得
python -m backtest225 fetch
```

## ルールの書き方

### 1. YAML（コード不要）

```yaml
name: 移動平均クロス（25日/75日）
data:
  source: yahoo          # yahoo（自動取得） / csv
  ticker: 1321.T
  # csv: ../data/my_1321.csv
  dividends: builtin     # builtin（同梱の公表値） / yahoo / none / 分配金CSVのパス
period: {start: 2010-01-01, end: }
capital: 1000000
params: {fast: 25, slow: 75}           # 式の中で名前として使える
rules:
  entry: "cross_up(sma(close, fast), sma(close, slow))"
  exit:  "cross_down(sma(close, fast), sma(close, slow))"
  direction: long        # long / short
  weight: 1.0            # 資産の何割を投じるか
risk:                    # すべて任意
  stop_loss: 0.08        # 建値から8%逆行で損切り
  take_profit: 0.15      # 15%で利確
  trailing_stop: 0.10    # 保有中の高値から10%下落で決済
  max_hold_days: 60      # 最大保有営業日数
execution:
  timing: next_open      # next_open（翌日始値, 既定） / close（当日終値, 楽観的）
  commission_rate: 0.0   # 約定代金に対する手数料率
  commission_min: 0      # 最低手数料（円）
  commission_fixed: 0    # 1回あたり固定手数料（円）
  slippage_rate: 0.0005  # 不利方向の滑り
  short_fee_rate: 0.0    # 空売りの貸株料等（年率）
  tax_rate: 0.0          # 年間実現益への税率（例 0.20315）
position: {mode: percent}  # percent（資産×weight） / fixed_amount（amount×weight）
lot_size: auto           # auto（1321の売買単位推移） / 数値
```

**式で使える変数**: `open high low close volume`、`weekday`(0=月〜4=金) `month` `day` `year`、
`month_first_day` `month_last_day`（月初/月末営業日）、`true` `false`、`params` に書いた名前

**関数**:

| 関数 | 内容 |
|---|---|
| `sma(x, n)` / `ema(x, n)` | 単純/指数移動平均 |
| `rsi(x, n)` | RSI |
| `macd(x, 12, 26)` / `macd_signal(x, 12, 26, 9)` / `macd_hist(...)` | MACD |
| `bb_upper(x, 20, 2)` / `bb_lower(x, 20, 2)` | ボリンジャーバンド |
| `highest(x, n)` / `lowest(x, n)` | 直近n日の最高/最安（当日含む） |
| `roc(x, n)` / `deviation(x, n)` | n日変化率% / 移動平均乖離率% |
| `atr(n)` / `std(x, n)` | ATR / 標準偏差 |
| `ref(x, n)` | n日前の値 |
| `cross_up(a, b)` / `cross_down(a, b)` | aがbを上抜け/下抜けした日 |
| `rising(x, n)` / `falling(x, n)` | n日連続上昇/下落 |
| `abs` `max(a,b)` `min(a,b)` | 要素ごと |

演算子: `+ - * / %`、比較 `> >= < <= == !=`、`and or not`、カッコ。

例: `"close > highest(ref(high, 1), 20)"`（20日高値ブレイク）、
`"deviation(close, 25) < -5 and weekday == 4"`（金曜に25日線から5%以上下方乖離）

### 2. Python（自由記述）

`strategies/month_end.py` を参照。`signals(df, params)` を定義して、
`entry`/`exit` 列の DataFrame か、目標ポジション比率（1.0/0.5/0/-1、NaN=維持）の Series を返します。
YAML 側は `python: month_end.py` と書きます。**その日の終値までの情報だけを使ってください**（`shift(-1)` 等で未来を見ると結果が不正に良くなります）。

## 検証の前提（重要）

- **約定**: 当日終値でシグナル判定 → 翌営業日の始値で約定（既定）。損切り等は日中の高値/安値で判定し、窓開けで逆指値を飛び越えたら始値で約定。同日に損切りと利確の両方に届いたら損切り扱い。
- **価格**: 2026/10/7効力発生の1口→100口の受益権分割を調整した、現在の口数基準の価格で計算します。
  分割前の未調整価格のCSVを読み込んだ場合は自動で検出・調整します。
- **売買単位**: 2026/10/5より前は旧1口（＝現在の100口）、以降は10口。
  （出典: [野村アセットマネジメント 2026/8/25 お知らせ](https://nextfunds.jp/data/2026/td_260825a.pdf)）
- **分配金**: `builtin` は [nextfunds.jp の1321銘柄ページ](https://nextfunds.jp/lineup/1321/) の公表値（2026/10/10取得）から作成した `data/dividends_1321.csv` を使用。
  決算日（7/8）から受渡しルール（T+3、2019/7/16約定分からT+2）で権利落ち日を算出し、その日に受け取ったものとして計算します（実際の入金は数か月後）。税は `tax_rate` 指定時のみ年単位で概算（損失繰越なし）。
- **比較対象**: 「買い持ち」＝初日の始値で買えるだけ買って最後まで保有（同じ手数料・分配金条件）。
- **レバレッジなし**: 買いは手元資金の範囲内。空売りは証拠金・追証を考慮しない簡易計算。

## データの注意

- 自動取得は Yahoo Finance（yfinance、非公式API）です。仕様変更で取得できなくなる可能性があり、値の正確性は保証されません。
- 2026/10/10時点の取得データでは、**2010/2/3以前の始値が0**になっており、前日終値で代用しています（レポートに明記）。
  厳密な検証には証券会社等からダウンロードしたCSVを `--csv` で指定してください。
- CSVは「日付・始値・高値・安値・終値・出来高」（日本語/英語の列名、UTF-8/Shift_JIS対応）。日付・終値は必須。
- 過去の成績は将来の成果を保証しません。パラメータを総当たりで選ぶと過剰適合しやすいため、`--split` での検証期間チェックを推奨します。

## テスト

```bash
python -m pytest -q
```

---

# 週次銘柄評点ツール（weekpick）

日本株を **金曜の終値で採点 → 翌週月曜（休場なら火曜）の寄付きで上位N銘柄を均等に買う → その週の最終営業日の引けで売る**、
というやり方を過去データで検証し、週末に来週の候補を表示するツールです。何銘柄に分けるのが良いかも比べられます。

```bash
python -m weekpick universe                                 # JPXの上場銘柄一覧を取得（data/universe_jpx.csv）
python -m weekpick fetch strategies/weekpick.yaml           # 対象銘柄の日足を取得（初回のみ数分）
python -m weekpick fetch strategies/weekpick.yaml --refresh # 週末の更新（全銘柄取り直し）
python -m weekpick factors strategies/weekpick.yaml         # 指標ごとの効き方（5分位・IC、学習/検証期間別）
python -m weekpick backtest strategies/weekpick.yaml        # 保有銘柄数ごとの週次成績
python -m weekpick backtest strategies/weekpick.yaml --holdings 5,10,20 --capital 3000000  # 100株単位・資金300万円
python -m weekpick backtest strategies/weekpick.yaml --cost 0   # コストなしで比較
python -m weekpick pick strategies/weekpick.yaml --top 20   # 最新の終値で採点した来週の候補
```

結果は `reports/` にCSVで保存されます（`*_holdings.csv` 要約、`*_weekly.csv` 毎週の損益と銘柄、`*_factors.csv`、`*_pick_日付.csv`）。

## 設定（strategies/weekpick.yaml）

- `universe`: `topix100` / `topix500`（既定。TOPIX Core30+Large70+Mid400）/ `topix1000` / `prime` / 銘柄一覧CSV（`ticker` または `code` 列）
- `filters`: 直近20日の平均売買代金の下限（既定5億円）、株価の下限、上場からの営業日数
- `weights`: 指標の重み（0=不使用、マイナス=逆向き）。各銘柄の指標を毎週「対象銘柄内の順位(0〜1)」にして重み付き平均した値がスコア

| 指標 | 内容 |
|---|---|
| `reversal` | 直近5営業日の騰落率×(-1)。下げた銘柄ほど高い（短期リバーサル） |
| `momentum` | 12か月前→1か月前の騰落率 |
| `high52` | 終値 ÷ 過去250営業日の高値 |
| `volume_surge` | 5日平均出来高 ÷ 60日平均出来高 |
| `low_vol` | 60日の日次騰落率の標準偏差×(-1) |

- `holdings`: 比べる保有銘柄数。`capital` を指定すると、1銘柄あたり「資金÷銘柄数」で100株を買える銘柄だけが対象になり、端数は現金
- `execution`: 片道の手数料率と滑り。**毎週往復するため、往復コスト×52週がそのまま年率から引かれます**（片道0.1%なら年約10%）

## 検証の前提と限界（重要）

- **生存者バイアス**: 対象は「JPX一覧の取得時点（2026/9/30）で上場し、その時点で TOPIX500 に入っている銘柄」。過去に上場廃止・指数除外された銘柄は含まれないため、成績は実際より良く出やすい。
- **価格**: Yahoo Finance（非公式）。終値・始値は株式分割調整済み、配当は `Adj Close`÷`Close` の変化から権利落ち日の値下がり分を損益に含める。
  配信データの誤りへの対策として、(1) 出来高0の日の価格は使わない、(2) 前後5日が平穏で整数倍の段差は未調整の分割とみなして調整、
  (3) 100円以上の株価が1日で3倍超/3分の1未満になった日は除外、(4) 配当込み係数の不自然な変化は無視。処理内容は実行時に表示されます。
- **約定**: 寄付き・引けで必ずその価格で約定したとみなす（滑りは `slippage_rate` で概算）。ストップ高/安で約定しない場合は考慮していません。
- **イベント**: 決算発表・権利落ちなど翌週の予定は考慮していません。`pick` の候補は発注前に各社IR・TDnetで確認してください。
- **学習/検証**: `split` より前で重みを決め、後の期間で確かめます。後の期間の結果を見て重みを変えると、その期間も「学習」に使ったことになります。
- 過去の成績は将来の成果を保証しません。

## 週末の買いタイミング判定ノートブック（ANA HD・あおぞら銀行）

`notebooks/weekly_timing_ana_aozora.ipynb` を毎週末に上から実行すると、次の内容を表示します（計算は `weekpick/timing.py`）。

1. 最新の終値での RSI(2)<10 シグナルの有無と、同時に出ている条件（＋/－/？）と参考点
2. 過去のシグナル時の上昇確率（条件の有無別・参考点別・直近3年の一覧）
3. 直近1年のチャート（移動平均とシグナル週）
4. 損切り・利確の目安（損失：利益＝1：2）

```bash
pip install -r requirements.txt jupyter
jupyter notebook notebooks/weekly_timing_ana_aozora.ipynb
```

条件は 2008〜2026年の検証で、2銘柄と全銘柄とで向きがそろったものを選びました。ただし、多数の比較をしたことを補正すると統計的に確かとは言えず、参考点も売買ルールとしての検証は済んでいません。今後のデータで効き方を確認しながら使ってください。

## 週末のシグナル確認ノートブック（SOMPO HD・ウィリアムズ%R）

`notebooks/weekly_signal_sompo.ipynb` を毎週末に上から実行すると、次の内容を表示します。

1. SOMPO HD（8630）に ウィリアムズ%R(14)<-80 のシグナルが出ているか。同時に出ている条件（＋/－/？）と参考点
2. 過去のシグナル時の上昇確率（条件の有無別・参考点別・直近3年の一覧）
3. 条件の検証：各条件の有無で上昇確率がどう変わったかを、SOMPO・同業種の他銘柄・全銘柄で、前半（〜2017年）・後半（2018年〜）に分けて比較。自分で作った条件も同じ方法で検証可能
4. 直近1年のチャート、保証金から計算した株数と、損切りの目安

ノートブック冒頭の `TICKER` を変えれば他の銘柄にも使えますが、条件の説明文（分析時点の結果）は SOMPO と全銘柄の検証に基づいています。
計算は `weekpick/timing.py` です（`SIGNALS` でシグナル、`CONDITIONS` / `WR_CONDITIONS` で同時に見る条件を切り替え）。

