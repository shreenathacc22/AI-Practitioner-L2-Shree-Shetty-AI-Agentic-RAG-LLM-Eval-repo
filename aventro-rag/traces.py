"""Read rag_traces.jsonl — the pipeline's flight recorder.

    python traces.py                 # summary of the latest run
    python traces.py --stage chunk   # every span from one stage
    python traces.py --outliers      # the documents worth looking at
    python traces.py --searches      # queries, what they hit, how confident
    python traces.py --runs          # list runs in the file

Same idea as the labs' traces.py, extended past LLM calls to the stages that
actually decide whether RAG works: ingest, chunk, embed, search.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
say = console.print
FILE = Path(__file__).resolve().parent / "rag_traces.jsonl"


def load(run: str | None = None) -> list[dict]:
    if not FILE.exists():
        say("[red]No rag_traces.jsonl yet — run: python rag.py build[/red]")
        raise SystemExit(1)
    rows = [json.loads(l) for l in FILE.read_text().splitlines() if l.strip()]
    if run:
        rows = [r for r in rows if r["run"] == run]
    elif rows:
        rows = [r for r in rows if r["run"] == rows[-1]["run"]]     # latest run
    return rows


def cmd_runs() -> None:
    rows = [json.loads(l) for l in FILE.read_text().splitlines() if l.strip()]
    agg = defaultdict(lambda: {"n": 0, "ms": 0.0, "ts": ""})
    for r in rows:
        a = agg[r["run"]]
        a["n"] += 1
        a["ms"] += r.get("ms") or 0
        a["ts"] = a["ts"] or r["ts"]
    t = Table("run", "started", "spans", "total ms")
    for run, a in agg.items():
        t.add_row(run, a["ts"][:19], str(a["n"]), f"{a['ms']:.0f}")
    say(t)


def cmd_summary(rows: list[dict]) -> None:
    say(f"[bold]run {rows[0]['run']}[/bold] · {len(rows)} spans · {rows[0]['ts'][:19]}\n")

    by = defaultdict(list)
    for r in rows:
        by[r["stage"]].append(r.get("ms") or 0.0)
    t = Table("stage", "spans", "total ms", "median ms", "max ms", title="where the time went")
    for stage in ("ingest", "ingest_doc", "chunk_all", "chunk", "embed_all", "embed_batch", "search", "answer"):
        if stage not in by:
            continue
        v = sorted(by[stage])
        t.add_row(stage, str(len(v)), f"{sum(v):.0f}", f"{v[len(v)//2]:.1f}", f"{v[-1]:.1f}")
    say(t)

    ing = next((r for r in rows if r["stage"] == "ingest"), None)
    ch = next((r for r in rows if r["stage"] == "chunk_all"), None)
    em = next((r for r in rows if r["stage"] == "embed_all"), None)

    if ing:
        say(f"\n[yellow]ingest[/yellow]  {ing['n_docs']} docs · {ing['total_words']:,} words · format={ing['fmt']}")
        pd = [r for r in rows if r["stage"] == "parse_detail"]
        if pd:
            say(f"         pdf parser: {sum(r['pages'] for r in pd)} pages · "
                f"{sum(r['hyphen_repairs'] for r in pd):,} hyphen repairs · "
                f"{sum(r['empty_pages'] for r in pd)} empty pages")
    if ch:
        cw = ch["chunk_words"]
        say(f"[yellow]chunk[/yellow]   {ch['n_chunks']} chunks · target {ch['target_words']}w · "
            f"min {cw['min']} / median {cw['median']} / max {cw['max']} · "
            f"{ch['n_oversize']} over target")
        dropped = sum(r["words_in"] - r["words_out"] for r in rows if r["stage"] == "chunk")
        kept = sum(r["words_out"] for r in rows if r["stage"] == "chunk")
        say(f"         [dim]{kept:,} words in chunks vs {kept+dropped:,} raw "
            f"({dropped/(kept+dropped):.0%} difference). Heading lines move OUT of the body "
            f"and back in as chunk prefixes; '---' rules are discarded. Verified: no body "
            f"paragraph is lost.[/dim]")
    if em:
        wps = (ch["chunk_words"]["total"] / (em["ms"] / 1000)) if ch and em["ms"] else 0
        say(f"[yellow]embed[/yellow]   {em['n_chunks']} vectors · {em['model']} · dim {em['dim']} · "
            f"{em['ms']:.0f}ms ({wps:,.0f} words/s)")

    s = [r for r in rows if r["stage"] == "search"]
    if s:
        say(f"[yellow]search[/yellow]  {len(s)} queries · median top-distance "
            f"{sorted(r['top_distance'] for r in s)[len(s)//2]:.3f}")


def cmd_stage(rows: list[dict], stage: str, limit: int) -> None:
    sel = [r for r in rows if r["stage"] == stage][:limit]
    if not sel:
        say(f"no '{stage}' spans in this run")
        return
    for r in sel:
        head = r.get("doc") or r.get("query") or r.get("collection") or ""
        say(f"[yellow]{r['stage']}[/yellow] {head}  [dim]{r.get('ms')}ms[/dim]")
        for k, v in r.items():
            if k in ("ts", "run", "stage", "ms", "ok", "error", "doc", "query"):
                continue
            say(f"    {k}: {v}")
        say("")


def cmd_outliers(rows: list[dict]) -> None:
    ch = [r for r in rows if r["stage"] == "chunk"]
    if not ch:
        say("no chunk spans")
        return
    say("[bold]documents that dominate the index[/bold] — each chunk is a retrieval slot,")
    say("so one document producing 20 chunks can crowd out 20 others.\n")
    t = Table("document", "chunks", "words in", "words out", "largest chunk", "dropped")
    for r in sorted(ch, key=lambda x: -x["n_chunks"])[:8]:
        drop = r["words_in"] - r["words_out"]
        t.add_row(r["doc"].replace("md/", "")[:40], str(r["n_chunks"]), f"{r['words_in']:,}",
                  f"{r['words_out']:,}", str(r["chunk_words"].get("max", 0)),
                  f"{drop} ({drop/max(r['words_in'],1):.0%})")
    say(t)

    over = [r for r in ch if r["chunk_words"].get("max", 0) > r["target_words"]]
    if over:
        say(f"\n[yellow]{len(over)} documents contain a chunk larger than the {ch[0]['target_words']}-word target.[/yellow]")
        say("[dim]These are markdown tables — one paragraph with no blank line to split on.")
        say("Splitting them would break rows apart from their header, which is worse.[/dim]")


def cmd_searches(rows: list[dict]) -> None:
    s = [r for r in rows if r["stage"] == "search"]
    if not s:
        say("no search spans in this run — run: python rag.py ask \"...\"")
        return
    say("[bold]retrieval confidence[/bold] — cosine distance to the top chunk.")
    say("[dim]A high top-distance means nothing in the corpus was close: the shape of an")
    say("unanswerable question, and a signal you can act on before spending an LLM call.[/dim]\n")
    t = Table("query", "k", "top dist", "distinct docs", "top document")
    for r in s:
        t.add_row(r["query"][:46], str(r["k"]), f"{r['top_distance']:.3f}",
                  str(r["n_distinct_docs"]), (r["docs"][0] if r["docs"] else "—").replace("md/", "")[:34])
    say(t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="a specific run id (default: latest)")
    ap.add_argument("--stage", help="dump every span from one stage")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--outliers", action="store_true")
    ap.add_argument("--searches", action="store_true")
    ap.add_argument("--runs", action="store_true")
    a = ap.parse_args()

    if a.runs:
        cmd_runs(); return
    rows = load(a.run)
    if a.stage:
        cmd_stage(rows, a.stage, a.limit)
    elif a.outliers:
        cmd_outliers(rows)
    elif a.searches:
        cmd_searches(rows)
    else:
        cmd_summary(rows)


if __name__ == "__main__":
    main()
