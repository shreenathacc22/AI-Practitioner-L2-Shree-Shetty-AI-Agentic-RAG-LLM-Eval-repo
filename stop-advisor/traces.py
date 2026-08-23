"""Read stop_traces.jsonl — the pipeline's flight recorder.

    python traces.py                 # summary of the latest run
    python traces.py --runs          # every run in the file
    python traces.py --stage embed   # dump one stage
    python traces.py --numbers       # what the maths produced, run over run
    python traces.py --provenance    # THE audit question: fixture or live?

Modelled on aventro-rag/traces.py, with one view it does not have. In a RAG demo
the interesting question is where the time went. Here the interesting question is
where the NUMBERS came from — because a stop level computed from a synthetic
fixture and a stop level computed from live market data look identical on the
console and are not remotely the same object. `--provenance` answers that for
every run in the file, which is the check you would actually want after the fact.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from trace import STAGES

console = Console()
say = console.print
FILE = Path(__file__).resolve().parent / "stop_traces.jsonl"


def load(run: str | None = None) -> list[dict]:
    if not FILE.exists():
        say("[red]No stop_traces.jsonl yet — run: "
            "python stop_advisor.py --ticker AAPL --entry 220 --offline[/red]")
        raise SystemExit(1)
    rows = [json.loads(l) for l in FILE.read_text().splitlines() if l.strip()]
    if run:
        rows = [r for r in rows if r["run"] == run]
    elif rows:
        rows = [r for r in rows if r["run"] == rows[-1]["run"]]     # latest run
    return rows


def all_rows() -> list[dict]:
    return [json.loads(l) for l in FILE.read_text().splitlines() if l.strip()]


def cmd_runs() -> None:
    agg = defaultdict(lambda: {"n": 0, "ms": 0.0, "ts": "", "tick": "", "src": ""})
    for r in all_rows():
        a = agg[r["run"]]
        a["n"] += 1
        a["ms"] += r.get("ms") or 0
        a["ts"] = a["ts"] or r["ts"]
        a["tick"] = a["tick"] or r.get("ticker", "")
        if r["stage"] == "fetch_prices":
            a["src"] = r.get("source", "")
    t = Table("run", "started", "ticker", "prices", "spans", "total ms")
    for run, a in agg.items():
        t.add_row(run, a["ts"][:19], a["tick"] or "—", a["src"] or "—",
                  str(a["n"]), f"{a['ms']:.0f}")
    say(t)


def cmd_summary(rows: list[dict]) -> None:
    say(f"[bold]run {rows[0]['run']}[/bold] · {len(rows)} spans · {rows[0]['ts'][:19]}\n")

    by = defaultdict(list)
    for r in rows:
        by[r["stage"]].append(r.get("ms") or 0.0)
    t = Table("stage", "spans", "total ms", "median ms", "max ms", title="where the time went")
    for stage in STAGES:
        if stage not in by:
            continue
        v = sorted(by[stage])
        t.add_row(stage, str(len(v)), f"{sum(v):.0f}", f"{v[len(v)//2]:.1f}", f"{v[-1]:.1f}")
    say(t)

    missing = [s for s in STAGES if s not in by]
    if missing:
        # A missing stage is a finding, not a gap in the log. No news_search span
        # means the narrative layer never ran — usually an absent TAVILY_API_KEY.
        say(f"\n[yellow]stages that did not run:[/yellow] {', '.join(missing)}")

    fp = next((r for r in rows if r["stage"] == "fetch_prices"), None)
    ci = next((r for r in rows if r["stage"] == "compute_indicators"), None)
    ns = next((r for r in rows if r["stage"] == "news_search"), None)
    ch = next((r for r in rows if r["stage"] == "chunk"), None)
    em = next((r for r in rows if r["stage"] == "embed"), None)
    rt = [r for r in rows if r["stage"] == "retrieve"]
    sy = [r for r in rows if r["stage"] == "synthesize"]

    if fp:
        say(f"\n[yellow]prices[/yellow]     {fp.get('n_bars','?')} bars · {fp.get('first')} -> "
            f"{fp.get('last')} · last close {fp.get('last_close')}")
        say(f"            [dim]{fp.get('provenance','?')}[/dim]")
    if ci:
        say(f"[yellow]indicators[/yellow] ATR({ci['atr_period']},{ci['atr_method']}) = {ci.get('atr')} · "
            f"realised vol {ci.get('realised_vol_annual_pct')}% · "
            f"{ci.get('n_swing_lows')} swing lows")
        for label, lvl in (ci.get("candidates") or {}).items():
            say(f"            [dim]{label:<18} {lvl}[/dim]")
    if ns:
        say(f"[yellow]news[/yellow]       {ns.get('n_results')} articles · "
            f"{ns.get('total_words')} words · source={ns.get('source')}")
    if ch:
        cw = ch.get("chunk_words", {})
        say(f"[yellow]chunk[/yellow]      {ch.get('n_chunks')} chunks · "
            f"min {cw.get('min')} / median {cw.get('median')} / max {cw.get('max')} words")
    if em:
        say(f"[yellow]embed[/yellow]      {em.get('n_chunks')} vectors · {em.get('model')} · "
            f"dim {em.get('dim')} · {em.get('ms')}ms")
    for r in rt:
        say(f"[yellow]retrieve[/yellow]   k={r.get('k')} · top distance {r.get('top_distance')} · "
            f"{r.get('n_distinct_articles')} distinct articles {r.get('articles')}")
    for r in sy:
        say(f"[yellow]synthesize[/yellow] {r.get('context_words')} context words · "
            f"{r.get('n_passages')} passages · mode={r.get('mode','briefing')} · {r.get('ms')}ms")


def cmd_stage(rows: list[dict], stage: str, limit: int) -> None:
    sel = [r for r in rows if r["stage"] == stage][:limit]
    if not sel:
        say(f"no '{stage}' spans in this run (stages present: "
            f"{sorted({r['stage'] for r in rows})})")
        return
    for r in sel:
        head = r.get("ticker") or r.get("query") or r.get("collection") or ""
        say(f"[yellow]{r['stage']}[/yellow] {head}  [dim]{r.get('ms')}ms[/dim]")
        for k, v in r.items():
            if k in ("ts", "run", "stage", "ms", "ok", "error", "ticker", "query"):
                continue
            say(f"    {k}: {v}")
        say("")


def cmd_numbers() -> None:
    """Every indicator computation in the file, newest last. The regression view:
    the same ticker and fixture must produce the same ATR forever."""
    rows = [r for r in all_rows() if r["stage"] == "compute_indicators"]
    if not rows:
        say("no compute_indicators spans yet")
        return
    t = Table("run", "ticker", "bars", "ATR", "method", "realised vol %", "swings",
              title="what the maths produced")
    for r in rows[-15:]:
        t.add_row(r["run"], r.get("ticker", "—"), str(r.get("n_bars", "")),
                  f"{r.get('atr', 0):.4f}", f"{r['atr_period']}/{r['atr_method']}",
                  f"{r.get('realised_vol_annual_pct', 0):.2f}", str(r.get("n_swing_lows", "")))
    say(t)


def cmd_provenance() -> None:
    """Fixture or live, per run. The question you ask when someone shows you a level."""
    say("[bold]where did each run's numbers come from?[/bold]")
    say("[dim]A fixture-sourced stop level is an arithmetic demonstration. A live-sourced one\n"
        "is a statement about a real instrument. They print identically; only this tells them apart.[/dim]\n")
    from rich.table import Column
    t = Table("run", "started", "ticker", "source", "bars", "range",
              Column("provenance", no_wrap=True, overflow="ellipsis", max_width=38))
    for r in all_rows():
        if r["stage"] != "fetch_prices":
            continue
        colour = "yellow" if r.get("source") == "fixture" else "green"
        t.add_row(r["run"], r["ts"][:19], r.get("ticker", "—"),
                  f"[{colour}]{r.get('source','?')}[/{colour}]", str(r.get("n_bars", "")),
                  f"{r.get('first','?')} -> {r.get('last','?')}",
                  (r.get("provenance") or "").split("—")[0].strip() or "?")
    say(t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="a specific run id (default: latest)")
    ap.add_argument("--stage", help="dump every span from one stage")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--runs", action="store_true")
    ap.add_argument("--numbers", action="store_true")
    ap.add_argument("--provenance", action="store_true")
    a = ap.parse_args()

    if a.runs:
        cmd_runs(); return
    if a.numbers:
        cmd_numbers(); return
    if a.provenance:
        cmd_provenance(); return
    rows = load(a.run)
    if a.stage:
        cmd_stage(rows, a.stage, a.limit)
    else:
        cmd_summary(rows)


if __name__ == "__main__":
    main()
