"""バックテスト結果を自己完結型のHTMLレポートに出力する（外部ライブラリ・通信不要）。"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import Result
from .metrics import LABELS, annual_returns, fmt, monthly_returns, trades_frame

_CSS = """
:root{
  color-scheme: light;
  --surface:#fcfcfb; --surface-2:#f3f2ef; --border:#e2e1dc;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#76756f;
  --series-1:#2a78d6; --series-2:#eb6834; --grid:#e8e7e2;
  --pos:#2a78d6; --neg:#e34948; --mid:#f0efec;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --surface:#1a1a19; --surface-2:#232321; --border:#383835;
    --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#9a998f;
    --series-1:#3987e5; --series-2:#d95926; --grid:#2c2c2a;
    --pos:#3987e5; --neg:#e66767; --mid:#383835;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface:#1a1a19; --surface-2:#232321; --border:#383835;
  --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#9a998f;
  --series-1:#3987e5; --series-2:#d95926; --grid:#2c2c2a;
  --pos:#3987e5; --neg:#e66767; --mid:#383835;
}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text-primary);
  font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Yu Gothic UI","Noto Sans JP",sans-serif;
  font-size:14px;line-height:1.6}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:36px 0 10px}
.sub{color:var(--text-secondary);margin:0 0 16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.tile{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--text-secondary);font-size:12px}
.tile .v{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .b{color:var(--text-muted);font-size:12px}
.chart{position:relative;width:100%;}
.chart svg{display:block;width:100%;overflow:visible}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--text-secondary);font-size:12px;margin:4px 0}
.legend i{display:inline-block;width:16px;height:2px;vertical-align:middle;margin-right:6px}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:6px 10px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);
  white-space:nowrap;display:none;z-index:2}
.tip b{font-variant-numeric:tabular-nums}
.tip i{display:inline-block;width:12px;height:2px;vertical-align:middle;margin-right:6px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
th{color:var(--text-secondary);font-weight:500;font-size:12px;background:var(--surface)}
td.l,th.l{text-align:left}
.trades{max-height:480px;overflow:auto}
.trades thead th{position:sticky;top:0}
.notes li{color:var(--text-secondary)}
code,pre{background:var(--surface-2);border-radius:6px;font-size:12.5px}
pre{padding:10px 12px;overflow-x:auto;white-space:pre-wrap}
.axis text{fill:var(--text-muted);font-size:11px}
.warn{border-left:3px solid var(--neg);padding:8px 12px;background:var(--surface-2);border-radius:6px}
"""

_JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const yen = v => Math.round(v).toLocaleString('ja-JP') + '円';
const pct = v => (v*100).toFixed(2) + '%';
const num = v => v.toLocaleString('ja-JP', {maximumFractionDigits: 1});
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent){ const e=document.createElementNS(NS,tag); for(const k in attrs) e.setAttribute(k,attrs[k]); if(parent) parent.appendChild(e); return e; }
function niceTicks(lo, hi, n){
  const span = hi-lo || Math.abs(hi) || 1; const step0 = span/n; const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1,2,2.5,5,10].map(m=>m*mag).find(s=>s>=step0);
  const out=[]; for(let v=Math.ceil(lo/step)*step; v<=hi+1e-9; v+=step) out.push(+v.toFixed(10)); return out;
}
function lineChart(id, opt){
  const box = document.getElementById(id); const H = opt.height || 280;
  const draw = () => {
    box.innerHTML=''; const W = box.clientWidth; const m={l:72,r:16,t:10,b:24};
    const svg = el('svg',{viewBox:`0 0 ${W} ${H}`, height:H, role:'img','aria-label':opt.label}, box);
    const xs = D.dates.length; const X = i => m.l + (W-m.l-m.r) * i/(xs-1);
    let lo=Infinity, hi=-Infinity; opt.series.forEach(s=>s.values.forEach(v=>{ if(v!=null){lo=Math.min(lo,v);hi=Math.max(hi,v);} }));
    if(opt.zeroTop){ hi=0; } const pad=(hi-lo)*0.05||1; lo-=opt.zeroTop?pad:pad; hi+=opt.zeroTop?0:pad;
    const Y = v => m.t + (H-m.t-m.b) * (1-(v-lo)/(hi-lo));
    const g = el('g',{class:'axis'},svg);
    niceTicks(lo,hi,5).forEach(t=>{ el('line',{x1:m.l,x2:W-m.r,y1:Y(t),y2:Y(t),stroke:css('--grid')},g);
      const tx=el('text',{x:m.l-8,y:Y(t)+4,'text-anchor':'end'},g); tx.textContent=opt.fmtAxis(t); });
    const years=[]; let last=null; D.dates.forEach((d,i)=>{ const y=d.slice(0,4); if(y!==last){years.push([i,y]); last=y;} });
    const every = Math.ceil(years.length / Math.max(2, Math.floor((W-m.l)/56)));
    years.forEach(([i,y],k)=>{ if(k%every) return; const tx=el('text',{x:X(i),y:H-6,'text-anchor':'middle'},g); tx.textContent=y; });
    opt.series.forEach(s=>{
      let dpath=''; s.values.forEach((v,i)=>{ if(v==null) return; dpath += (dpath?'L':'M') + X(i).toFixed(1)+','+Y(v).toFixed(1); });
      if(s.area){ el('path',{d:dpath+`L${X(xs-1)},${Y(0)}L${X(0)},${Y(0)}Z`,fill:css(s.color),opacity:0.18},svg); }
      el('path',{d:dpath,fill:'none',stroke:css(s.color),'stroke-width':s.width||2,'stroke-linejoin':'round'},svg);
    });
    (opt.markers||[]).forEach(mk=>{
      const x=X(mk.i), y=Y(mk.v); const up = mk.type==='entry'; const s=6;
      const pts = up ? `${x},${y+3} ${x-s},${y+3+s*1.6} ${x+s},${y+3+s*1.6}` : `${x},${y-3} ${x-s},${y-3-s*1.6} ${x+s},${y-3-s*1.6}`;
      el('polygon',{points:pts,fill:css(up?'--series-1':'--series-2'),stroke:css('--surface'),'stroke-width':1.5},svg);
    });
    const cross = el('line',{y1:m.t,y2:H-m.b,stroke:css('--text-muted'),'stroke-width':1,visibility:'hidden'},svg);
    const dots = opt.series.map(s=>el('circle',{r:4,fill:css(s.color),stroke:css('--surface'),'stroke-width':2,visibility:'hidden'},svg));
    const tip = document.createElement('div'); tip.className='tip'; box.appendChild(tip);
    const hit = el('rect',{x:m.l,y:0,width:W-m.l-m.r,height:H,fill:'transparent'},svg);
    const move = ev => {
      const r = svg.getBoundingClientRect(); const px = (ev.clientX - r.left) * W / r.width;
      const i = Math.max(0, Math.min(xs-1, Math.round((px-m.l)/(W-m.l-m.r)*(xs-1))));
      cross.setAttribute('x1',X(i)); cross.setAttribute('x2',X(i)); cross.setAttribute('visibility','visible');
      tip.replaceChildren(); const hd=document.createElement('div'); hd.textContent=D.dates[i]; tip.appendChild(hd);
      opt.series.forEach((s,k)=>{ const v=s.values[i]; if(v==null){dots[k].setAttribute('visibility','hidden');return;}
        dots[k].setAttribute('cx',X(i)); dots[k].setAttribute('cy',Y(v)); dots[k].setAttribute('visibility','visible');
        const row=document.createElement('div'); const key=document.createElement('i'); key.style.background=css(s.color);
        const b=document.createElement('b'); b.textContent=opt.fmtTip(v); row.append(key,b,document.createTextNode(' '+s.name)); tip.appendChild(row); });
      (opt.markers||[]).filter(mk=>mk.i===i).forEach(mk=>{ const row=document.createElement('div'); row.textContent=mk.text; tip.appendChild(row); });
      tip.style.display='block'; const tw=tip.offsetWidth; let left = X(i)/W*box.clientWidth + 12;
      if(left+tw > box.clientWidth) left = X(i)/W*box.clientWidth - tw - 12; tip.style.left=left+'px'; tip.style.top='8px';
    };
    hit.addEventListener('pointermove', move); hit.addEventListener('pointerdown', move);
    hit.addEventListener('pointerleave', ()=>{ tip.style.display='none'; cross.setAttribute('visibility','hidden'); dots.forEach(d=>d.setAttribute('visibility','hidden')); });
  };
  draw(); let t; window.addEventListener('resize', ()=>{ clearTimeout(t); t=setTimeout(draw,120); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
}
lineChart('equity', {label:'資産推移', series:[
  {name:'このルール', values:D.equity, color:'--series-1'},
  {name:'買い持ち(比較)', values:D.bench, color:'--series-2'}], fmtAxis:v=>(v/10000).toLocaleString('ja-JP')+'万', fmtTip:yen});
lineChart('dd', {label:'ドローダウン', height:180, zeroTop:true, series:[
  {name:'このルール', values:D.dd, color:'--series-1', area:true},
  {name:'買い持ち(比較)', values:D.bdd, color:'--series-2', width:1.5}], fmtAxis:v=>(v*100).toFixed(0)+'%', fmtTip:pct});
lineChart('price', {label:'価格と売買', series:[{name:'終値', values:D.close, color:'--text-secondary', width:1.5}],
  markers:D.markers, fmtAxis:num, fmtTip:v=>num(v)+'円'});
"""


def _heat_style(v: float, scale: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    k = min(abs(v) / scale, 1.0) * 70
    color = "var(--pos)" if v >= 0 else "var(--neg)"
    return f' style="background:color-mix(in oklab, {color} {k:.0f}%, var(--mid))"'


def _metrics_table(m: dict) -> str:
    rows = []
    for key, (label, kind) in LABELS.items():
        s = m["strategy"].get(key)
        b = m["benchmark"].get(key)
        rows.append(
            f"<tr><td class=l>{html.escape(label)}</td><td>{fmt(s, kind)}</td>"
            f"<td>{fmt(b, kind) if key in m['benchmark'] else '—'}</td></tr>"
        )
    return ("<table><thead><tr><th class=l>指標</th><th>このルール</th><th>買い持ち(比較)</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def _annual_table(eq: pd.Series, bench: pd.Series, initial: float) -> str:
    a, b = annual_returns(eq, initial), annual_returns(bench, initial)
    rows = "".join(
        f"<tr><td class=l>{y}</td><td{_heat_style(a[y], 0.3)}>{a[y] * 100:,.1f}%</td>"
        f"<td{_heat_style(b[y], 0.3)}>{b[y] * 100:,.1f}%</td><td>{(a[y] - b[y]) * 100:+,.1f}pt</td></tr>"
        for y in a.index
    )
    return ("<table><thead><tr><th class=l>年</th><th>このルール</th><th>買い持ち</th><th>差</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def _monthly_table(eq: pd.Series, initial: float) -> str:
    t = monthly_returns(eq, initial)
    head = "".join(f"<th>{mth}月</th>" for mth in range(1, 13))
    body = []
    for y, row in t.iterrows():
        cells = "".join(
            f"<td{_heat_style(row.get(mth), 0.08)}>{row.get(mth) * 100:.1f}</td>"
            if pd.notna(row.get(mth)) else "<td></td>"
            for mth in range(1, 13)
        )
        body.append(f"<tr><td class=l>{y}</td>{cells}</tr>")
    return f"<table><thead><tr><th class=l>年</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _trades_table(result: Result) -> str:
    tf = trades_frame(result)
    if tf.empty:
        return "<p class=sub>取引はありませんでした。</p>"
    head = "".join(f"<th{' class=l' if c in ('方向', '決済理由') else ''}>{c}</th>" for c in tf.columns)
    body = []
    for _, r in tf.iterrows():
        tds = []
        for c in tf.columns:
            v = r[c]
            if c in ("損益(円)", "コスト(円)", "口数"):
                v = f"{v:,.0f}"
            elif c in ("建値", "決済値"):
                v = f"{v:,.2f}" if v is not None and pd.notna(v) else "—"
            cls = " class=l" if c in ("方向", "決済理由") else ""
            tds.append(f"<td{cls}>{html.escape(str(v))}</td>")
        body.append(f"<tr>{''.join(tds)}</tr>")
    return f"<div class='scroll trades'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def render(result: Result, metrics: dict, cfg: dict, out_path: str | Path, rule_text: str = "",
           extra_html: str = "", extra_notes: list[str] | None = None) -> Path:
    eq = result.equity["equity"]
    bench = result.benchmark
    price = result.price
    idx = {d: i for i, d in enumerate(eq.index)}
    markers = []
    for t in result.trades:
        side = "買" if t.direction == "long" else "売"
        if t.entry_date in idx:
            markers.append({"i": idx[t.entry_date], "v": float(price["close"].loc[t.entry_date]), "type": "entry",
                            "text": f"▲ 新規{side} {t.entry_price:,.1f}円"})
        if t.exit_date in idx and not t.exit_reason.startswith("期間終了"):
            markers.append({"i": idx[t.exit_date], "v": float(price["close"].loc[t.exit_date]), "type": "exit",
                            "text": f"▼ 決済({t.exit_reason}) {t.exit_price:,.1f}円 損益{t.pnl:,.0f}円"})
    payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in eq.index],
        "equity": [round(v, 0) for v in eq],
        "bench": [round(v, 0) for v in bench],
        "dd": [round(v, 5) for v in (eq / eq.cummax() - 1)],
        "bdd": [round(v, 5) for v in (bench / bench.cummax() - 1)],
        "close": [round(v, 2) for v in price["close"]],
        "markers": markers,
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    s, b = metrics["strategy"], metrics["benchmark"]
    name = html.escape(str(cfg.get("name", "バックテスト")))
    period = f"{eq.index[0]:%Y-%m-%d} 〜 {eq.index[-1]:%Y-%m-%d}"
    tiles = [
        ("最終資産", fmt(s["final_equity"], "yen"), f"買い持ち {fmt(b['final_equity'], 'yen')}"),
        ("トータルリターン", fmt(s["total_return"], "pct"), f"買い持ち {fmt(b['total_return'], 'pct')}"),
        ("年率リターン(CAGR)", fmt(s["cagr"], "pct"), f"買い持ち {fmt(b['cagr'], 'pct')}"),
        ("最大ドローダウン", fmt(s["max_drawdown"], "pct"), f"買い持ち {fmt(b['max_drawdown'], 'pct')}"),
        ("取引回数 / 勝率", f"{s['n_trades']}回 / {fmt(s['win_rate'], 'pct')}",
         f"PF {fmt(s['profit_factor'], 'num')}"),
    ]
    tiles_html = "".join(
        f"<div class=tile><div class=k>{k}</div><div class=v>{v}</div><div class=b>{html.escape(sub)}</div></div>"
        for k, v, sub in tiles
    )
    ex = cfg.get("execution") or {}
    assumptions = [
        f"初期資金: {result.initial_capital:,.0f}円 / 約定: "
        + ("シグナル翌営業日の始値" if ex.get("timing", "next_open") == "next_open" else "シグナル当日の終値（楽観的）"),
        f"手数料率 {ex.get('commission_rate', 0) * 100:g}%（最低 {ex.get('commission_min', 0):,}円、固定 {ex.get('commission_fixed', 0):,}円）"
        f" / スリッページ {ex.get('slippage_rate', 0) * 100:g}% / 税率 {ex.get('tax_rate', 0) * 100:g}%",
        "売買単位: 2026/10/5より前は旧1口（＝現在の100口）、以降は10口として計算（lot_size: auto の場合）",
        "比較の「買い持ち」は初日の始値で買えるだけ買い、最後まで保有（同じ手数料・分配金条件、税金なし）",
        "過去の成績は将来の結果を保証しません。データは非公式ソースを含むため、重要な判断の前に一次情報で確認してください",
    ] + list(result.notes) + list(extra_notes or [])
    notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in assumptions)
    warn = ""
    if s["n_trades"] < 30:
        warn = (f"<p class=warn>決済済みの取引が {s['n_trades']} 回と少ないため、勝率・PF等の統計的な信頼性は低いです。</p>")

    doc = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{name}</title><style>{_CSS}</style></head>
<body><main>
<h1>{name}</h1>
<p class=sub>NEXT FUNDS 日経225連動型上場投信（1321） / 検証期間 {period} / 作成 {datetime.now():%Y-%m-%d %H:%M}</p>
<div class=tiles>{tiles_html}</div>
{warn}
<h2>資産推移</h2>
<div class=legend><span><i style="background:var(--series-1)"></i>このルール</span><span><i style="background:var(--series-2)"></i>買い持ち(比較)</span></div>
<div id=equity class=chart></div>
<h2>ドローダウン（直近高値からの下落率）</h2>
<div class=legend><span><i style="background:var(--series-1)"></i>このルール</span><span><i style="background:var(--series-2)"></i>買い持ち(比較)</span></div>
<div id=dd class=chart></div>
<h2>価格と売買ポイント</h2>
<div class=legend><span><span style="color:var(--series-1)">▲</span> 新規建て</span><span><span style="color:var(--series-2)">▼</span> 決済</span><span>価格は分割調整後（現在の口数基準）</span></div>
<div id=price class=chart></div>
<h2>成績指標</h2>
<div class=scroll>{_metrics_table(metrics)}</div>
<h2>年別リターン</h2>
<div class=scroll>{_annual_table(eq, bench, result.initial_capital)}</div>
<h2>月別リターン（%）</h2>
<div class=scroll>{_monthly_table(eq, result.initial_capital)}</div>
{extra_html}
<h2>取引一覧（{len(result.trades)}件）</h2>
{_trades_table(result)}
<h2>ルール</h2>
<pre>{html.escape(rule_text)}</pre>
<h2>前提条件・注意</h2>
<ul class=notes>{notes_html}</ul>
</main>
<script type="application/json" id="data">{data_json}</script>
<script>{_JS}</script>
</body></html>"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path
