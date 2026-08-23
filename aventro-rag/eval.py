"""Score the RAG system against the golden set — retrieval and generation separately.

    python eval.py retrieval                 # recall@k, zero LLM calls, seconds
    python eval.py retrieval --k 1 3 5 8     # sweep k
    python eval.py answers                   # judged answer quality on the 12 core cases
    python eval.py answers --hybrid          # same suite, BM25+RRF fusion
    python eval.py ablate                    # md vs pdf vs both, same suite

Why two scores and not one: when a case fails you need to know WHICH half broke.
Retrieval recall says whether the required chunk ever reached the context; answer
quality says whether the model used it. One blended number tells you neither and
sends you to debug the wrong component.

The rule this project exists to enforce: every change — chunk size, k, embedding,
prompt, retrieval mode, format — reruns this suite. Beat the number or it does not
ship. That loop, not any single technique, is what "good RAG" means.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import rag
from kit import chat, client, meter, say

ROOT = Path(__file__).resolve().parent
SUITES = {
    "core":     ROOT / "goldset" / "golden.jsonl",     # 12 hand-verified cases
    "curated":  ROOT / "goldset" / "curated.jsonl",    # 221 from the dataset authors
    "multihop": ROOT / "goldset" / "multihop.jsonl",   # mined + approved
}


def load_gold(suite: str) -> list[dict]:
    paths = SUITES.values() if suite == "all" else [SUITES[suite]]
    rows = []
    for path in paths:
        if path.exists():
            rows += [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        say(f"[red]Suite '{suite}' is empty.[/red] Build it: python goldset.py curated | mine")
        raise SystemExit(1)
    return rows


# ── schema helpers ──────────────────────────────────────────────────────────
# Two schemas live in these files. The core suite uses the rich form (expect{},
# must_cite as doc+heading, per-case rubric); the imported curated rows use a flat
# form. Normalising here keeps one grading path instead of two that drift apart.

def norm_doc(name: str) -> str:
    """'Aventro Storm SUV' / 'md/Aventro Storm SUV.md' -> a comparable key."""
    name = re.sub(r"^(md|pdf)/", "", name)
    name = re.sub(r"\.(md|pdf)$", "", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def spec(case: dict) -> dict:
    """One normalised view of a case, whichever schema it came in."""
    e = case.get("expect")
    if e is not None:                                   # rich schema
        return {
            "behavior": e.get("behavior", "answer"),
            "facts": e.get("facts", []),
            "cite": [norm_doc(c["doc"]) if isinstance(c, dict) else norm_doc(c)
                     for c in e.get("must_cite", [])],
            "headings": [c.get("heading", "") for c in e.get("must_cite", []) if isinstance(c, dict)],
            "not_cite": [norm_doc(c["doc"]) if isinstance(c, dict) else norm_doc(c)
                         for c in e.get("must_not_cite", [])],
            "not_contain": e.get("must_not_contain", []),
            "forbidden": e.get("forbidden", []),
            "rubric": case.get("rubric", ""),
            "bucket": case.get("bucket", case.get("category", "?")),
        }
    return {                                            # flat schema (curated import)
        "behavior": case.get("expected_behavior", "answer"),
        "facts": [case["reference_answer"]] if case.get("reference_answer") else [],
        "cite": [norm_doc(c) for c in case.get("must_cite", [])],
        "headings": [], "not_cite": [],
        "not_contain": case.get("must_not_contain", []),
        "forbidden": [], "rubric": "",
        "bucket": case.get("query_type", case.get("type", "?")),
    }


# ── retrieval: did the required chunk reach the context at all? ─────────────

def recall_at_k(cases: list[dict], k: int, fmt: str, hybrid: bool):
    hits = full = 0
    misses = []
    cases = [c for c in cases if spec(c)["cite"]]        # refusal cases have no target doc
    for c in cases:
        need = set(spec(c)["cite"])
        got = {norm_doc(h["doc"]) for h in rag.search(c["question"], k=k, fmt=fmt, hybrid=hybrid)}
        if got & need:
            hits += 1
        if need <= got:
            full += 1
        else:
            misses.append({**c, "retrieved": sorted(got), "missing": sorted(need - got)})
    n = max(len(cases), 1)
    return hits / n, full / n, misses, n


def cmd_retrieval(args) -> None:
    cases = load_gold(args.suite)
    mode = "hybrid" if args.hybrid else "vector"
    say(f"[bold]Retrieval[/bold] · suite={args.suite} · format={args.format} · {mode}\n")
    last = None
    say(f"  {'k':>3}  {'any-hit':>8}  {'all-docs':>9}   [dim](cases with a required document)[/dim]")
    for k in args.k:
        hit, full, misses, n = recall_at_k(cases, k, args.format, args.hybrid)
        say(f"  {k:>3}  {hit:>7.1%}  {full:>8.1%}   [dim]n={n}[/dim]")
        last = (k, misses)
    if args.show_misses and last[1]:
        say(f"\n  [red]misses at k={last[0]}[/red]:")
        for m in last[1][:args.show_misses]:
            say(f"    [yellow]{m['id']}[/yellow] {m['question'][:70]}")
            say(f"       missing: {', '.join(m['missing'])}")


# ── answers: did the model actually use what it retrieved? ──────────────────

RUBRIC = {
    "answer": "PASS only if the answer states the EXPECTED FACTS, with no contradiction and no "
              "invented detail. Wording may differ. A missing required fact is a FAIL. Declining "
              "to answer is a FAIL.",
    "refuse": "PASS only if the answer clearly declines because the documents do not cover the "
              "question. Any specific figure or fact offered AS THE ANSWER is a FAIL, including "
              "one borrowed from a different product. Hedged guessing is a FAIL. Naming what IS "
              "available while still declining the actual question is a PASS.",
    "correct_premise": "The QUESTION contains a false or unsupported premise. PASS only if the "
                       "answer explicitly identifies and corrects it. Silently answering as though "
                       "the premise held is a FAIL even if the rest is accurate. Inventing data to "
                       "satisfy the request is a FAIL.",
}

JUDGE = """You are a strict grader for a RAG system. Grade ONE answer.

{case_rubric}BEHAVIOUR REQUIRED ({behavior}): {rubric}
{forbidden}
EXPECTED FACTS (the substance to look for — NOT required wording): {facts}

Grading discipline:
  · Where the CASE-SPECIFIC RUBRIC and the general behaviour rubric differ, the
    CASE-SPECIFIC RUBRIC WINS. It was written for this case on purpose.
  · EXPECTED FACTS describe substance, not phrasing. Do not fail an answer for
    different wording, or for omitting an incidental qualifier, unless the rubric
    explicitly requires it.
  · Do not fail an answer for naming the brand, the model, or other detail that is
    plainly consistent with the sources. "Unsupported" means CONTRADICTED by or
    ABSENT from the documents — not merely unmentioned in the expected facts.
  · Be strict about what the rubric IS strict about, and only that.

Return ONLY JSON: {{"pass": true|false, "why": str}}

QUESTION: {q}
ANSWER UNDER TEST: {ans}
"""


def cmd_answers(args) -> None:
    cases = load_gold(args.suite)[:args.n]
    cli = client()
    mode = "hybrid" if args.hybrid else "vector"
    say(f"[bold]Answers[/bold] · suite={args.suite} · k={args.k} · format={args.format} · {mode}\n")

    passed, errors, fails = 0, [], []
    by_bucket: dict[str, list[int]] = {}

    for c in cases:
        s = spec(c)
        hits = rag.search(c["question"], k=args.k, fmt=args.format, hybrid=args.hybrid)
        out = rag.answer(cli, c["question"], hits, label="suite")

        # Hard gate first: costs nothing and encodes the exact trap the case was built
        # around. If a Nova question quotes the Spark's 280L, no LLM opinion is needed.
        tripped = [x for x in s["not_contain"] if x.lower() in out.lower()]
        cited = {norm_doc(h["doc"]) for h in hits}
        bad_cite = [d for d in s["not_cite"] if d in cited]

        if tripped or bad_cite:
            ok, why = False, "; ".join(
                [f"contains forbidden string '{x}'" for x in tripped] +
                [f"cited forbidden doc '{d}'" for d in bad_cite])
        else:
            v = None
            for _ in (1, 2):     # a judge that fails to parse must not become a fake FAIL
                raw = chat(cli, [{"role": "user", "content": JUDGE.format(
                    behavior=s["behavior"], rubric=RUBRIC[s["behavior"]],
                    case_rubric=f"CASE-SPECIFIC RUBRIC: {s['rubric']}\n" if s["rubric"] else "",
                    forbidden=f"FORBIDDEN: {'; '.join(s['forbidden'])}\n" if s["forbidden"] else "",
                    facts=" | ".join(s["facts"]) or "(none — the answer should decline)",
                    q=c["question"], ans=out)}],
                    label="judge", max_tokens=600, response_format={"type": "json_object"})
                try:
                    v = json.loads(raw.strip().removeprefix("```json")
                                   .removeprefix("```").removesuffix("```").strip())
                    break
                except json.JSONDecodeError:
                    v = None
            if v is None:
                errors.append(c["id"])
                say(f"  [magenta]ERROR[/magenta] {c['id']:<7} judge returned no JSON — excluded")
                continue
            ok, why = bool(v.get("pass")), v.get("why", "")

        passed += ok
        row = by_bucket.setdefault(s["bucket"], [0, 0]); row[1] += 1; row[0] += ok
        say(f"  {'[green]PASS[/green]' if ok else '[red]FAIL[/red]'} {c['id']:<7} "
            f"[dim]{s['behavior']:<15}[/dim] {c['question'][:52]}")
        if not ok:
            # Blame must be chunk-aware AND document-scoped: a fact found in some other
            # document is not the fact being tested, and calling that "generation" sends
            # you to rewrite a prompt that was never the problem.
            need = set(s["cite"])
            scoped = " ".join(h["text"] for h in hits
                              if not need or norm_doc(h["doc"]) in need).lower()
            # lowercase the probe tokens — `scoped` is lowercased, so capitalised
            # tokens never matched and every fact looked "missing", mis-blaming
            # retrieval for what were generation or judging problems.
            absent = [f for f in s["facts"]
                      if not any(tok.lower() in scoped
                                 for tok in re.findall(r"[\d.]{2,}|[A-Z][a-z]{2,}", f)[:4])]
            doc_missing = need - cited
            blame = (f"RETRIEVAL — required doc(s) missing: {sorted(doc_missing)}" if doc_missing
                     else "RETRIEVAL — required facts not in the scoped context" if absent and s["facts"]
                     else "GENERATION — the context supported it; the answer still missed")
            say(f"       [dim]{why[:115]}[/dim]")
            say(f"       [yellow]{blame}[/yellow]")
            say(f"       [dim]answer: {' '.join(out.split())[:120]}[/dim]")
            fails.append((c, blame))

    scored = len(cases) - len(errors)
    say(f"\n  [dim]by bucket:[/dim]")
    for b, (ok, tot) in sorted(by_bucket.items()):
        say(f"    {b:<16} {ok}/{tot}")
    say(f"\n  [bold]scorecard: {passed}/{scored} = {passed/max(scored,1):.0%}[/bold]"
        + (f"  [magenta]({len(errors)} judge errors excluded)[/magenta]" if errors else ""))
    if fails:
        say(f"  failures by cause: {dict(Counter(f[1].split(' —')[0] for f in fails))}")
    meter.show()


# ── ablation ────────────────────────────────────────────────────────────────

def cmd_ablate(args) -> None:
    cases = load_gold(args.suite)
    say(f"[bold]Ablation[/bold] — same {len(cases)} cases, three corpora, k={args.k}")
    say("[dim]The PDFs are renders of the same 50 markdown files. If 'both' does not beat\n"
        "'md', indexing both formats is pure cost — this is how you find out.[/dim]\n")
    say(f"  {'corpus':<8} {'chunks':>7} {'any-hit':>9} {'all-docs':>10}")
    for fmt in ("md", "pdf", "both"):
        col = rag.collection(fmt)
        if col.count() == 0:
            say(f"  {fmt:<8} {'—':>7}  [dim]not built: python rag.py build --format {fmt}[/dim]")
            continue
        hit, full, _, _ = recall_at_k(cases, args.k, fmt, args.hybrid)
        say(f"  {fmt:<8} {col.count():>7} {hit:>8.1%} {full:>9.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="core", choices=["core", "curated", "multihop", "all"])
    ap.add_argument("--format", default="md", choices=["md", "pdf", "both"])
    ap.add_argument("--hybrid", action="store_true", help="fuse BM25 with vector search (RRF)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("retrieval"); r.set_defaults(fn=cmd_retrieval)
    r.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 8])
    r.add_argument("--show-misses", type=int, default=5)

    a = sub.add_parser("answers"); a.set_defaults(fn=cmd_answers)
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--n", type=int, default=100)

    b = sub.add_parser("ablate"); b.set_defaults(fn=cmd_ablate)
    b.add_argument("--k", type=int, default=5)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
