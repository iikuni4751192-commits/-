"""コマンドライン:  python -m weekpick <universe|fetch|factors|backtest|pick> ..."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from . import data as WD
from .backtest import LOT, Setup, factor_report, prepare, run, stats, universe_stats
from .factors import FACTORS
from .universe import fetch_jpx_list, load_universe

STAT_LABELS = [
    ("weeks", "週数", "int"), ("mean", "週平均", "pct"), ("median", "週中央値", "pct"),
    ("win_rate", "勝率(週)", "pct1"), ("cagr", "年率(複利)", "pct1"), ("vol", "年率ボラ", "pct1"),
    ("sharpe", "シャープ", "num"), ("max_dd", "最大DD", "pct1"), ("worst_week", "最悪週", "pct1"),
    ("excess_mean", "対比較 週超過", "pct"), ("excess_t", "超過のt値", "num"),
    ("beat_rate", "比較に勝った週", "pct1"), ("avg_held", "平均保有数", "num"), ("invested", "投資比率", "pct1"),
]


def _fmt(v, kind):
    if v is None or pd.isna(v):
        return "—"
    return {"pct": f"{v * 100:+.3f}%", "pct1": f"{v * 100:.1f}%", "num": f"{v:.2f}", "int": f"{int(v)}"}[kind]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _setup(cfg: dict, a) -> Setup:
    flt = cfg.get("filters") or {}
    ex = cfg.get("execution") or {}
    per = cfg.get("period") or {}
    return Setup(
        weights={k: float(v) for k, v in (cfg.get("weights") or {}).items()},
        min_turnover=float(flt.get("min_turnover", 5e8)),
        min_price=float(flt.get("min_price", 0)),
        min_history=int(flt.get("min_history", 260)),
        cost=a.cost if a.cost is not None else float(ex.get("commission_rate", 0)) + float(ex.get("slippage_rate", 0.001)),
        capital=float(a.capital or cfg.get("capital") or 0) or None,
        start=str(a.start or per.get("start") or "") or None,
        end=str(a.end or per.get("end") or "") or None,
    )


def _universe(cfg: dict):
    return load_universe(str(cfg.get("universe", "topix500")))


def _prepare(cfg, a):
    u = _universe(cfg)
    panel = WD.load_panel(list(u["ticker"]))
    s = _setup(cfg, a)
    print(f"■ ユニバース {cfg.get('universe', 'topix500')}: {len(panel.tickers)} 銘柄"
          f"（銘柄一覧の時点 {u['list_date'].iloc[0] if 'list_date' in u else '—'}）")
    for n in panel.notes:
        print(f"  ・{n}")
    return u, panel, s, prepare(panel, s)


def _split(df: pd.DataFrame, split: str | None):
    if not split:
        return [("全期間", df)]
    t = pd.Timestamp(split)
    return [("全期間", df), (f"学習 〜{split}", df[df.index < t]), (f"検証 {split}〜", df[df.index >= t])]


def cmd_universe(a):
    df = fetch_jpx_list()
    print(f"JPX上場銘柄一覧（{df['list_date'].iloc[0]}時点、内国株式 {len(df)} 銘柄）を {WD.CACHE_DIR.parent.parent / 'universe_jpx.csv'} に保存")
    print(df["size"].value_counts().to_string())


def cmd_fetch(a):
    cfg = load_config(a.config)
    u = _universe(cfg)
    failed = WD.fetch(list(u["ticker"]), refresh=a.refresh)
    print(f"完了: {len(u) - len(failed)} / {len(u)} 銘柄" + (f"（取得失敗: {' '.join(failed)}）" if failed else ""))


def cmd_factors(a):
    cfg = load_config(a.config)
    _, _, s, prep = _prepare(cfg, a)
    split = a.split or cfg.get("split")
    print(f"\n■ 指標ごとの効き方（毎週5分位に分けた翌週騰落率。週平均との差。コスト控除後・配当込み）")
    for k, v in FACTORS.items():
        print(f"  {k}: {v}")
    rows = []
    for label, idx in _split(prep.scores, split):
        sub = _subset(prep, idx.index)
        rep = factor_report(sub)
        rep.insert(0, "period", label)
        rows.append(rep)
        show = rep.copy()
        for c in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q5-Q1"]:
            show[c] = show[c].map(lambda v: _fmt(v, "pct"))
        for c in ["t", "IC_t"]:
            show[c] = show[c].map(lambda v: _fmt(v, "num"))
        show["IC"] = show["IC"].map(lambda v: f"{v:+.4f}")
        print(f"\n[{label}]")
        print(show.drop(columns=["period"]).to_string(index=False))
    print("\n  Q5=指標が大きい20%。Q5-Q1 が正で t値が概ね2以上なら「大きいほど翌週上がりやすい」傾向が統計的に安定。"
          "\n  負なら逆向き（weights をマイナスに）。学習期間と検証期間で符号が同じかを必ず確認してください。")
    _save(pd.concat(rows), a.out or f"reports/{Path(a.config).stem}_factors.csv")


def _subset(prep, idx):
    from dataclasses import replace

    return replace(prep, scores=prep.scores.loc[idx], factors={k: v.loc[idx] for k, v in prep.factors.items()},
                   mask=prep.mask.loc[idx])


def _save(df, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, encoding="utf-8-sig")
    print(f"\nCSV: {out.resolve()}")


def cmd_backtest(a):
    cfg = load_config(a.config)
    _, _, s, prep = _prepare(cfg, a)
    ns = [int(x) for x in a.holdings.split(",")] if a.holdings else list(cfg.get("holdings") or [1, 3, 5, 10, 20])
    split = a.split or cfg.get("split")
    w0 = prep.scores.index
    print(f"  期間: {w0[0]:%Y-%m-%d} 〜 {prep.weeks[-1].exit:%Y-%m-%d}（{len(w0)} 週）"
          f"  重み: {', '.join(f'{k}={v:g}' for k, v in s.weights.items() if v)}"
          f"  片道コスト: {s.cost * 100:.2f}%" + (f"  資金: {s.capital:,.0f}円（{LOT}株単位）" if s.capital else ""))
    results = {n: run(prep, n, s.capital) for n in ns}
    table = []
    for label, _ in _split(results[ns[0]], split):
        print(f"\n■ 保有銘柄数ごとの成績 [{label}]")
        cols = {}
        for n in ns:
            w = dict(_split(results[n], split))[label]
            cols[f"{n}銘柄"] = stats(w)
            table.append({"period": label, "holdings": n, **cols[f"{n}銘柄"]})
        base = dict(_split(results[ns[0]], split))[label]
        cols["比較:全銘柄平均"] = universe_stats(base)
        table.append({"period": label, "holdings": "universe", **cols["比較:全銘柄平均"]})
        df = pd.DataFrame({name: {lab: _fmt(st[key], kind) for key, lab, kind in STAT_LABELS}
                           for name, st in cols.items()})
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            print(df.to_string())
    print("\n  比較対象＝足切りを通った全銘柄を同じタイミング・同じコストで均等に買った場合。"
          "\n  「対比較 週超過」の t値が概ね2以上で、学習・検証の両期間で正なら、偶然ではない可能性が高まります。"
          "\n  銘柄数が少ないほど1週の振れ（最悪週・最大DD）が大きく、週平均が同じでも複利の年率は下がります。"
          "\n  毎週の売買コスト（往復）×52週が年率から差し引かれる点にも注意してください。")
    stem = Path(a.config).stem
    _save(pd.DataFrame(table).set_index("period"), a.out or f"reports/{stem}_holdings.csv")
    weekly = pd.concat({f"{n}銘柄": results[n] for n in ns}, axis=1)
    _save(weekly, f"reports/{stem}_weekly.csv")


def cmd_pick(a):
    cfg = load_config(a.config)
    u, panel, s, _ = _prepare(cfg, a)
    from .factors import compute_factors, eligible, score

    facs = compute_factors(panel)
    mask = eligible(panel, s.min_turnover, s.min_price, s.min_history)
    sc = score(facs, s.weights, mask)
    d = sc.index[-1]
    row = sc.loc[d].dropna()
    n = a.top or max(cfg.get("holdings") or [10])
    if s.capital and a.holdings:
        row = row[panel.close.loc[d, row.index] * LOT <= s.capital / a.holdings]
    top = row.nlargest(n)
    info = u.set_index("ticker")
    out = pd.DataFrame({
        "順位": range(1, len(top) + 1),
        "銘柄名": info.reindex(top.index)["name"].values,
        "業種": info.reindex(top.index)["sector"].values,
        "終値": panel.close.loc[d, top.index].round(1).values,
        "100株代金": (panel.close.loc[d, top.index] * LOT).round().astype(int).values,
        "スコア": top.round(4).values,
        **{k: facs[k].loc[d, top.index].round(4).values for k, w in s.weights.items() if w},
    }, index=top.index)
    print(f"\n■ {d:%Y-%m-%d}（{'月火水木金土日'[d.weekday()]}）終値時点のスコア上位 {len(top)} 銘柄"
          f"（対象 {len(row)} 銘柄）")
    if d.weekday() != 4:
        print("  ※ 最新データが金曜ではありません。週末に `fetch --refresh` してから実行してください")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(out.to_string(formatters={"100株代金": "{:,}".format}))
    print("  （指標列は factors の定義どおりの値。例: reversal 0.15 = 直近5営業日で15%下落）")
    print("\n  ※ 決算発表・権利落ち・適時開示など翌週のイベントは考慮していません。各社IR・TDnetで確認してください。"
          "\n  ※ 価格は Yahoo Finance（非公式）由来です。発注前に証券会社の画面で確認してください。")
    _save(out, a.out or f"reports/{Path(a.config).stem}_pick_{d:%Y%m%d}.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m weekpick", description="日本株の週次銘柄評点ツール")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("universe", help="JPXの上場銘柄一覧を取得").set_defaults(func=cmd_universe)
    p = sub.add_parser("fetch", help="ユニバースの日足を yfinance で取得")
    p.add_argument("config")
    p.add_argument("--refresh", action="store_true", help="取得済みの銘柄も取り直す（週末の更新用）")
    p.set_defaults(func=cmd_fetch)
    for name, func, hlp in [("factors", cmd_factors, "指標ごとの効き方を検証"),
                            ("backtest", cmd_backtest, "保有銘柄数ごとの週次成績を検証"),
                            ("pick", cmd_pick, "最新の終値で採点し、来週の候補を表示")]:
        p = sub.add_parser(name, help=hlp)
        p.add_argument("config")
        p.add_argument("--start")
        p.add_argument("--end")
        p.add_argument("--capital", type=float, help="資金（円）。100株単位で買える銘柄だけを対象にする")
        p.add_argument("--out")
        p.add_argument("--cost", type=float, help="片道コスト（手数料率+滑り）。設定ファイルより優先 例: 0.0005")
        if name != "pick":
            p.add_argument("--split", help="学習/検証の境目 YYYY-MM-DD")
        if name == "backtest":
            p.add_argument("--holdings", help="比べる保有銘柄数 例: 1,3,5,10,20")
        if name == "pick":
            p.add_argument("--top", type=int, help="表示件数")
            p.add_argument("--holdings", type=int, help="--capital と併用: この銘柄数に分けて買える銘柄だけ表示")
        p.set_defaults(func=func)
    a = ap.parse_args(argv)
    a.func(a)
