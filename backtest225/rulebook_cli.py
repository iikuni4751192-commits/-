"""ルール表シミュレーターの実行: python -m backtest225 rulebook <設定.yaml> [--breadth CSV]"""
from __future__ import annotations

import html
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .metrics import LABELS, compute, fmt
from .report import render
from .rulebook import RULE_NOTES, RulebookParams, load_breadth, load_events, run_rulebook, summarize
from .runner import load_config


def _nt_ratio() -> pd.Series:
    n225 = D.fetch_yahoo("^N225")["close"]
    topix_etf = D.fetch_yahoo("1305.T")["close"]
    return (n225 / topix_etf).dropna()


def run(a):
    cfg, base = load_config(a.config)
    params = dict(cfg.get("params") or {})
    allowed = {f.name for f in fields(RulebookParams)}
    bad = set(params) - allowed
    if bad:
        raise ValueError(f"params に未知の項目: {sorted(bad)}")
    p = RulebookParams(**params)
    if a.capital:
        p.capital = a.capital
    dcfg = cfg.get("data") or {}
    source = "csv" if a.csv else dcfg.get("source", "yahoo")
    csv_path = a.csv or (str(base / dcfg["csv"]) if dcfg.get("csv") else None)
    ticker = dcfg.get("ticker", D.DEFAULT_TICKER)
    df, notes = D.load_prices(source=source, ticker=ticker, csv_path=csv_path, end=a.end or (cfg.get("period") or {}).get("end"))
    start = a.start or (cfg.get("period") or {}).get("start")

    breadth_path = a.breadth or cfg.get("breadth")
    breadth = None
    if breadth_path:
        bp = Path(breadth_path)
        breadth = load_breadth(bp if bp.is_absolute() or bp.exists() else base / bp)
        cover = breadth.reindex(df.index[df.index >= pd.Timestamp(start or df.index[0])]).notna().mean()
        notes.append(f"騰落レシオ: {breadth_path}（{breadth.index[0]:%Y-%m-%d}〜{breadth.index[-1]:%Y-%m-%d}、検証期間の{cover:.0%}をカバー）")
    else:
        p.use_breadth = False
        notes.append("騰落レシオ: データなし → E5（120%以上で新規見送り）・E6/Setup C・X9 は未反映（暫定結果）")
    events = load_events() if p.use_events else None
    if events is not None:
        notes.append("重要イベント: data/events.csv（FOMC・日銀・米CPI＝各公式サイト、SQ＝計算）")
    nt = _nt_ratio()
    dividends = D.to_ex_dates(D.load_dividends_csv(D.BUILTIN_DIVIDENDS[ticker]), df.index) if ticker in D.BUILTIN_DIVIDENDS else None
    lots = D.lot_size_series(df.index, ticker, cfg.get("lot_size", "auto"))

    res, counts = run_rulebook(df, p, nt=nt, breadth=breadth, events=events, dividends=dividends, lots=lots, start=start)
    res.notes = notes
    m = compute(res)
    sm = summarize(res)

    eq = res.equity
    print(f"■ {cfg.get('name')}  期間 {eq.index[0]:%Y-%m-%d} 〜 {eq.index[-1]:%Y-%m-%d}  初期資金 {p.capital:,.0f}円")
    for n_ in notes:
        print(f"  ・{n_}")
    print()
    w = max(len(v[0]) for v in LABELS.values())
    for key, (label, kind) in LABELS.items():
        bv = fmt(m["benchmark"][key], kind) if key in m["benchmark"] else "—"
        print(f"{label:<{w}}  {fmt(m['strategy'].get(key), kind):>16}  {bv:>16}")
    print(f"\n平均R倍数: {fmt(sm.get('avg_r'), 'num')} / 中央値R: {fmt(sm.get('median_r'), 'num')} / 部分利確到達率: {fmt(sm.get('partial_rate'), 'pct')}")
    print(f"シグナル発生: {counts['signals']}  見送り内訳: 損切り幅外 {counts['skipped_stop_width']} / 騰落レシオ過熱 {counts['skipped_breadth']} / "
          f"イベント前 {counts['skipped_event']} / 損失上限 {counts['skipped_loss_limit']} / 指値不成立 {counts['unfilled']} / 口数0 {counts['skipped_size0']}")
    if "by_setup" in sm:
        print("\n[セットアップ別]\n" + sm["by_setup"].to_string(float_format=lambda v: f"{v:,.2f}"))
        print("\n[決済理由別（最終決済の理由）]\n" + sm["by_reason"].to_string(float_format=lambda v: f"{v:,.2f}"))
    periods = _period_split(res)
    print("\n[期間別] " + " / ".join(f"{k}: 取引{v['n']}件 平均R {fmt(v['avg_r'], 'num')} 損益 {fmt(v['pnl'], 'yen')}" for k, v in periods.items()))

    out = Path(a.out or f"reports/{Path(a.config).stem}.html")
    extra = _extra_html(sm, counts, periods)
    render(res, m, cfg, out, rule_text=Path(a.config).read_text(encoding="utf-8"), extra_html=extra,
           extra_notes=["【ルール解釈】" + x for x in RULE_NOTES])
    print(f"\nHTMLレポート: {out.resolve()}")
    if a.trades_csv:
        rows = [{"No": k + 1, "セットアップ": t.setup, "建日": t.entry_date.date(), "建値": round(t.entry_price, 2),
                 "初期損切り": round(t.entry_price - t.risk_per_unit, 2), "口数": t.initial_units,
                 "決済": " / ".join(f"{d:%Y-%m-%d} {u}口@{x:,.2f}({w})" for d, u, x, w in t.exits),
                 "最終決済日": t.exit_date.date(), "純損益": round(t.pnl), "R倍数": round(t.r_multiple, 2),
                 "決済理由": t.exit_reason} for k, t in enumerate(res.trades)]
        pd.DataFrame(rows).to_csv(a.trades_csv, index=False, encoding="utf-8-sig")
        print(f"取引一覧CSV: {Path(a.trades_csv).resolve()}")


def _period_split(res):
    # 2_環境認識 E9：指数構成の変化（2026/4/1 キオクシア等の採用）の前後で分ける
    cut = pd.Timestamp("2026-04-01")
    out = {}
    for name, sel in (("2026/3/31以前", lambda t: t.entry_date < cut), ("2026/4/1以降", lambda t: t.entry_date >= cut)):
        ts = [t for t in res.trades if sel(t) and not t.exit_reason.startswith("期間終了")]
        out[name] = {"n": len(ts), "avg_r": float(np.mean([t.r_multiple for t in ts])) if ts else np.nan,
                     "pnl": float(sum(t.pnl for t in ts))}
    return out


def _table(df: pd.DataFrame, pct_cols=(), yen_cols=(), r_cols=()) -> str:
    head = "<th class=l></th>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for idx, row in df.iterrows():
        tds = []
        for c in df.columns:
            v = row[c]
            if c in pct_cols:
                s = fmt(v, "pct")
            elif c in yen_cols:
                s = fmt(v, "yen")
            elif c in r_cols:
                s = fmt(v, "num")
            else:
                s = f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
            tds.append(f"<td>{html.escape(s)}</td>")
        body.append(f"<tr><td class=l>{html.escape(str(idx))}</td>{''.join(tds)}</tr>")
    return f"<div class=scroll><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _extra_html(sm, counts, periods) -> str:
    if "by_setup" not in sm:
        return "<h2>ルール表の集計</h2><p class=sub>決済済みの取引がありません。</p>"
    sig = counts["signals"]
    skip = (f"損切り幅が1〜3ATR外 {counts['skipped_stop_width']}件、騰落レシオ過熱 {counts['skipped_breadth']}件、"
            f"イベント前 {counts['skipped_event']}件、損失上限 {counts['skipped_loss_limit']}件、指値不成立 {counts['unfilled']}件、口数0 {counts['skipped_size0']}件")
    per = "".join(f"<li>{html.escape(k)}: 取引 {v['n']}件 / 平均R {fmt(v['avg_r'], 'num')} / 損益 {fmt(v['pnl'], 'yen')}</li>" for k, v in periods.items())
    return f"""
<h2>ルール表の集計（9_検証集計 相当）</h2>
<p class=sub>平均R倍数 {fmt(sm['avg_r'], 'num')} / 中央値R {fmt(sm['median_r'], 'num')} / +2R部分利確に届いた割合 {fmt(sm['partial_rate'], 'pct')}</p>
<h2>セットアップ別</h2>
{_table(sm['by_setup'], pct_cols=('勝率',), yen_cols=('純損益',), r_cols=('平均R',))}
<h2>決済理由別（最後の決済の理由）</h2>
{_table(sm['by_reason'], yen_cols=('純損益',), r_cols=('平均R',))}
<h2>シグナルと見送り</h2>
<p class=sub>シグナル発生（優先順位適用前・延べ）A {sig['A']} / B {sig['B']} / C {sig['C']}。見送り：{html.escape(skip)}</p>
<h2>指数構成変化の前後（E9）</h2>
<ul class=notes>{per}</ul>
"""
