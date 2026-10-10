"""コマンドライン:  python -m backtest225 <run|optimize|fetch> ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import data as D
from .metrics import LABELS, fmt, trades_frame
from .report import render
from .runner import backtest_from_config, load_config, optimize


def _overrides(a) -> dict:
    return {k: getattr(a, k) for k in ("start", "end", "capital", "csv", "refresh") if getattr(a, k, None)}


def _add_common(p):
    p.add_argument("config", help="ストラテジー設定ファイル（YAML）")
    p.add_argument("--start", help="検証開始日 YYYY-MM-DD（設定ファイルより優先）")
    p.add_argument("--end", help="検証終了日 YYYY-MM-DD")
    p.add_argument("--capital", type=float, help="初期資金（円）")
    p.add_argument("--csv", help="価格データに使うCSV（指定時は自動取得しない）")
    p.add_argument("--refresh", action="store_true", help="Yahoo Finance から再取得してキャッシュを更新")


def _print_summary(m: dict):
    s, b = m["strategy"], m["benchmark"]
    w = max(len(v[0]) for v in LABELS.values())
    print(f"{'指標':<{w}}  {'このルール':>16}  {'買い持ち':>16}")
    for key, (label, kind) in LABELS.items():
        bv = fmt(b[key], kind) if key in b else "—"
        print(f"{label:<{w}}  {fmt(s.get(key), kind):>16}  {bv:>16}")


def cmd_run(a):
    cfg, base = load_config(a.config)
    res, m = backtest_from_config(cfg, base, _overrides(a))
    eq = res.equity
    print(f"■ {cfg.get('name', Path(a.config).stem)}  期間 {eq.index[0]:%Y-%m-%d} 〜 {eq.index[-1]:%Y-%m-%d}")
    for n in res.notes:
        print(f"  ・{n}")
    print()
    _print_summary(m)
    if not a.no_report:
        out = Path(a.out or f"reports/{Path(a.config).stem}.html")
        render(res, m, cfg, out, rule_text=Path(a.config).read_text(encoding="utf-8"))
        print(f"\nHTMLレポート: {out.resolve()}")
    if a.trades_csv:
        trades_frame(res).to_csv(a.trades_csv, index=False, encoding="utf-8-sig")
        print(f"取引一覧CSV: {Path(a.trades_csv).resolve()}")


def cmd_optimize(a):
    cfg, base = load_config(a.config)
    table, oos = optimize(cfg, base, _overrides(a), split_date=a.split)
    show = table.head(a.top).copy()
    for c in ("total_return", "cagr", "max_drawdown", "win_rate"):
        show[c] = show[c].map(lambda v: fmt(v, "pct"))
    for c in ("sharpe", "profit_factor"):
        show[c] = show[c].map(lambda v: fmt(v, "num"))
    period = f"（学習期間: {a.split} より前）" if a.split else ""
    print(f"■ パラメータ探索 {len(table)}通り{period} 並び順: {(cfg.get('optimize') or {}).get('metric', 'sharpe')}")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(show.to_string(index=False))
    out = Path(a.out or f"reports/{Path(a.config).stem}_optimize.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n全結果CSV: {out.resolve()}")
    if oos:
        s, b = oos["metrics"]["strategy"], oos["metrics"]["benchmark"]
        print(f"\n■ 検証期間（{a.split} 以降）で最良パラメータ {oos['params']} を試した結果")
        for key in ("total_return", "cagr", "max_drawdown", "sharpe", "n_trades", "win_rate"):
            label, kind = LABELS[key]
            print(f"  {label}: {fmt(s[key], kind)}" + (f"（買い持ち {fmt(b[key], kind)}）" if key in b else ""))
    print("\n※ 多数の組合せから最良を選ぶと過去データへの過剰適合（カーブフィッティング）が起きやすくなります。"
          "\n   --split で学習期間と検証期間を分け、検証期間でも通用するかを確認してください。")


def cmd_rulebook(a):
    from .rulebook_cli import run
    run(a)


def cmd_fetch(a):
    df = D.fetch_yahoo(a.ticker, refresh=True)
    print(f"{a.ticker}: {len(df)}日分 {df.index[0]:%Y-%m-%d} 〜 {df.index[-1]:%Y-%m-%d}")
    print(f"キャッシュ: {D.CACHE_DIR}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="backtest225", description="NF日経225(1321) 投資ルール検証ツール")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="バックテストを実行しレポートを作成")
    _add_common(r)
    r.add_argument("--out", help="HTMLレポートの出力先（既定: reports/<設定名>.html）")
    r.add_argument("--no-report", action="store_true", help="HTMLを作らない")
    r.add_argument("--trades-csv", help="取引一覧をCSVで保存")
    r.set_defaults(func=cmd_run)

    o = sub.add_parser("optimize", help="パラメータの組合せを総当たりで検証")
    _add_common(o)
    o.add_argument("--split", help="学習/検証の分割日 YYYY-MM-DD（この日より前で最適化し、以降で検証）")
    o.add_argument("--top", type=int, default=15, help="表示する上位件数")
    o.add_argument("--out", help="全結果CSVの出力先")
    o.set_defaults(func=cmd_optimize)

    rb = sub.add_parser("rulebook", help="ルール表（Excel）のスイング売買ルールを検証")
    _add_common(rb)
    rb.add_argument("--breadth", help="騰落レシオCSV（日付, 騰落レシオ%%）")
    rb.add_argument("--out", help="HTMLレポートの出力先")
    rb.add_argument("--trades-csv", help="取引一覧（部分決済・R倍数つき）をCSV保存")
    rb.set_defaults(func=cmd_rulebook)

    f = sub.add_parser("fetch", help="Yahoo Finance から価格データを取得・更新")
    f.add_argument("--ticker", default=D.DEFAULT_TICKER)
    f.set_defaults(func=cmd_fetch)

    a = p.parse_args(argv)
    try:
        a.func(a)
    except (ValueError, FileNotFoundError, AttributeError, ImportError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
