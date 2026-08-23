"""Build the golden set — the instrument every other change is measured against.

    python goldset.py curated                 # import the 221 curated Q&A (single-doc)
    python goldset.py mine --n 30             # mine multi-hop cases across doc pairs
    python goldset.py review                  # print drafts as a review sheet
    python goldset.py approve mh-003 mh-007   # promote drafts to the frozen set
    python goldset.py stats

Two layers, because they answer different questions:

  curated/  the repo ships 221 Aventro Q&A rows with ground-truth SUPPORTING FACTS
            naming the source file. That is retrieval ground truth for free — it
            scores recall@k without anyone writing an answer. All 221 are
            single-document, so it measures lookup, not synthesis.

  mined/    genuine multi-hop cases: a question that cannot be answered from any
            ONE document. Mined by pairing related documents, asking the model to
            write a question spanning both, then VERIFYING the multi-hop claim by
            checking each document alone cannot answer it. Unverified candidates
            are discarded, not shipped — an unverified 'multi-hop' set that is
            secretly single-hop measures nothing.

Nothing enters golden.jsonl without passing through `approve`. The model drafts;
you decide. That is the whole point of a human reference answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from kit import chat, client, meter, say

ROOT = Path(__file__).resolve().parent
GOLD = ROOT / "goldset" / "golden.jsonl"
DRAFTS = ROOT / "goldset" / "drafts.jsonl"
RAW_QUERIES = ROOT / "goldset" / "all_queries_raw.csv"


# ── storage ─────────────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def norm_name(s: str) -> str:
    """The CSV and the filesystem disagree on whitespace: the corpus has
    ' AventroZoom SUV.md' (leading space) and 'Aventro  Connectivity.md' (double
    space) where the CSV writes them clean. Collapse whitespace to match."""
    return re.sub(r"\s+", " ", s).strip().lower()


def corpus_index() -> dict[str, str]:
    """normalised filename -> real doc id used by rag.py"""
    return {norm_name(p.name): f"md/{p.name}" for p in (ROOT / "corpus" / "md").glob("*.md")}


# ── layer 1 · the curated set (free retrieval ground truth) ─────────────────

def cmd_curated(args) -> None:
    idx = corpus_index()
    rows, unmatched = [], 0
    for r in csv.DictReader(RAW_QUERIES.open(newline="", encoding="utf-8", errors="replace")):
        if r["Enterprise Name"].strip() != "Aventro Motors":
            continue
        facts = json.loads(r["Supporting Facts"])
        docs, texts = [], []
        for f in facts:
            doc = idx.get(norm_name(f.get("filename", "")))
            if doc:
                docs.append(doc)
                texts.append(f.get("text", ""))
        if not docs:
            unmatched += 1
            continue
        rows.append({
            "id": f"cur-{len(rows)+1:03d}",
            "source": "curated",
            "type": "single_doc",
            "query_type": r["Query Type"].strip(),
            "question": r["Query"].strip(),
            "reference_answer": None,          # curated set ships facts, not answers
            "must_cite": sorted(set(docs)),
            "supporting_facts": texts,
            "status": "approved",              # ground truth from the dataset authors
        })
    existing = [g for g in load(GOLD) if g.get("source") != "curated"]
    save(GOLD, existing + rows)
    say(f"[green]imported {len(rows)} curated cases[/green] "
        f"({unmatched} skipped — no matching file) -> {GOLD.relative_to(ROOT)}")
    say(f"  these carry must_cite ground truth, so they score [bold]recall@k[/bold] with zero LLM calls.")


# ── layer 2 · mine multi-hop cases ──────────────────────────────────────────

PAIR_PROMPT = """You are building an evaluation set for a retrieval system over Aventro Motors documents.

Below are TWO documents. Write {n} questions a real customer might ask that REQUIRE
COMBINING facts from BOTH documents.

A valid question uses one of these shapes:
  · COMPARISON   — weigh a fact in A against a fact in B ("which is cheaper/larger/safer, and by how much")
  · ARITHMETIC   — combine numbers from A and B ("total cost of owning X including service Y")
  · CONDITIONAL  — a constraint stated in A determines the answer drawn from B
  · DEPENDENCY   — the fact in B is only meaningful given the fact in A

REJECT-WORTHY (do not produce these): two unrelated questions stapled together with
"and". If your question can be split into two independent questions that each stand
alone and get answered separately, it is NOT multi-hop — rewrite it so the two facts
must interact to produce a single answer.

Do NOT invent facts. Every number must appear verbatim in the documents.

Return ONLY JSON: {{"items":[{{"question":str,"reference_answer":str,"fact_a":str,"fact_b":str,"shape":str}}]}}
where fact_a is the exact sentence/table row from DOCUMENT A the answer depends on,
fact_b the same from DOCUMENT B, shape is one of comparison/arithmetic/conditional/dependency,
and reference_answer is a complete, correct answer in 1-3 sentences.

DOCUMENT A ({doc_a}):
{text_a}

DOCUMENT B ({doc_b}):
{text_b}
"""

CONJUNCTION_PROMPT = """Here is a question intended for a multi-hop retrieval benchmark.

Can it be split into two INDEPENDENT questions, each fully answerable on its own,
whose answers simply sit side by side? If yes it is a stapled conjunction, not a
genuine multi-hop question, and must be rejected.

Genuine multi-hop means the two facts INTERACT — compared, added, or one gating the other.

Return ONLY JSON: {{"stapled": true|false, "why": str}}

QUESTION: {q}
"""

VERIFY_PROMPT = """Can the QUESTION below be answered completely and correctly using ONLY the
single DOCUMENT provided? Answer strictly.

Return ONLY JSON: {{"answerable_alone": true|false, "why": str}}

QUESTION: {q}

DOCUMENT ({doc}):
{text}
"""


def read_docs() -> dict[str, str]:
    return {f"md/{p.name}": p.read_text(encoding="utf-8", errors="replace")
            for p in sorted((ROOT / "corpus" / "md").glob("*.md"))}


def related_pairs(docs: dict[str, str], limit: int) -> list[tuple[str, str]]:
    """Pair documents that are related but distinct.

    Too similar (two trim tables) and the 'multi-hop' question is a trivial
    comparison; unrelated (tyre change + careers) and no honest question spans
    them. So take the similarity band in between, using the same MiniLM that
    powers retrieval — no extra dependency."""
    import math
    from rag import EMBED
    names = list(docs)
    vecs = EMBED([docs[n][:4000] for n in names])

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)

    scored = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s = cos(vecs[i], vecs[j])
            if 0.45 <= s <= 0.82:                      # the band
                scored.append((s, names[i], names[j]))
    scored.sort(reverse=True)

    # spread the pairs across documents instead of letting one hub doc dominate
    used, out = {}, []
    for s, a, b in scored:
        if used.get(a, 0) >= 2 or used.get(b, 0) >= 2:
            continue
        used[a] = used.get(a, 0) + 1
        used[b] = used.get(b, 0) + 1
        out.append((a, b))
        if len(out) >= limit:
            break
    return out


def parse_json(s: str) -> dict:
    s = s.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def cmd_mine(args) -> None:
    cli = client()
    docs = read_docs()
    pairs = related_pairs(docs, limit=args.pairs)
    say(f"paired {len(pairs)} related document couples (similarity band 0.45–0.82)\n")

    drafts, seen = load(DRAFTS), set()
    per_pair = max(1, -(-args.n // max(len(pairs), 1)))
    kept = rejected = 0

    for a, b in pairs:
        if kept >= args.n:
            break
        out = chat(cli, [{"role": "user", "content": PAIR_PROMPT.format(
            n=per_pair, doc_a=a, doc_b=b, text_a=docs[a][:5000], text_b=docs[b][:5000])}],
            label="mine", max_tokens=1200, response_format={"type": "json_object"})
        for item in parse_json(out).get("items", []):
            q = (item.get("question") or "").strip()
            if not q or q.lower() in seen or kept >= args.n:
                continue
            seen.add(q.lower())

            # GATE 1 — stapled conjunction? cheapest check, no document text needed.
            c = parse_json(chat(cli, [{"role": "user", "content": CONJUNCTION_PROMPT.format(q=q)}],
                label="conjunction", max_tokens=500, response_format={"type": "json_object"}))
            if c.get("stapled") is True:
                rejected += 1
                say(f"  [dim]rejected (stapled conjunction): {q[:70]}[/dim]")
                continue

            # GATE 2 — verify the multi-hop claim: if either doc alone answers it, it is not multi-hop.
            alone = False
            for doc in (a, b):
                v = parse_json(chat(cli, [{"role": "user", "content": VERIFY_PROMPT.format(
                    q=q, doc=doc, text=docs[doc][:6000])}],
                    label="verify", max_tokens=500, response_format={"type": "json_object"}))
                if v.get("answerable_alone") is True:
                    alone = True
                    break
            if alone:
                rejected += 1
                say(f"  [dim]rejected (single-hop): {q[:70]}[/dim]")
                continue

            kept += 1
            drafts.append({
                "id": f"mh-{len(drafts)+1:03d}",
                "source": "mined",
                "type": "multi_hop",
                "query_type": "Multi-Hop",
                "question": q,
                "reference_answer": (item.get("reference_answer") or "").strip(),
                "must_cite": [a, b],
                "shape": item.get("shape", ""),
                "supporting_facts": [item.get("fact_a", ""), item.get("fact_b", "")],
                "status": "draft",
            })
            say(f"  [green]kept[/green] {q[:78]}")

    save(DRAFTS, drafts)
    say(f"\n[bold]{kept} drafts kept · {rejected} rejected as single-hop[/bold] -> {DRAFTS.relative_to(ROOT)}")
    say("Nothing is in the golden set yet. Run [bold]python goldset.py review[/bold], then approve.")
    meter.show()


# ── review + approve ────────────────────────────────────────────────────────

def cmd_review(args) -> None:
    drafts = [d for d in load(DRAFTS) if d["status"] == "draft"]
    if not drafts:
        say("no drafts pending — run: python goldset.py mine")
        return
    say(f"[bold]{len(drafts)} drafts awaiting your decision[/bold]\n")
    for d in drafts:
        say(f"[yellow]{d['id']}[/yellow]  [dim]{' + '.join(x.replace('md/','') for x in d['must_cite'])}[/dim]")
        say(f"  [bold]Q[/bold] {d['question']}")
        say(f"  [bold]A[/bold] {d['reference_answer']}")
        for f in d["supporting_facts"]:
            say(f"     [dim]· {f[:140]}[/dim]")
        say("")
    say("[bold]python goldset.py approve <id> ...[/bold]   (or --all)")
    say("Edit any reference answer directly in goldset/drafts.jsonl before approving —")
    say("the frozen answer should be YOURS, not the model's first draft.")


def cmd_approve(args) -> None:
    drafts = load(DRAFTS)
    ids = {d["id"] for d in drafts if d["status"] == "draft"} if args.all else set(args.ids)
    gold, moved = load(GOLD), 0
    for d in drafts:
        if d["id"] in ids and d["status"] == "draft":
            d["status"] = "approved"
            gold.append(dict(d))
            moved += 1
    save(DRAFTS, drafts)
    save(GOLD, gold)
    say(f"[green]approved {moved}[/green] -> {GOLD.relative_to(ROOT)} (now {len(gold)} cases)")


def cmd_stats(args) -> None:
    from collections import Counter
    gold = load(GOLD)
    if not gold:
        say("golden.jsonl is empty.")
        return
    say(f"[bold]{len(gold)} cases[/bold] in {GOLD.relative_to(ROOT)}")
    say(f"  by source     : {dict(Counter(g['source'] for g in gold))}")
    say(f"  by type       : {dict(Counter(g['type'] for g in gold))}")
    say(f"  with ref answer: {sum(1 for g in gold if g.get('reference_answer'))}")
    say(f"  multi-doc      : {sum(1 for g in gold if len(g['must_cite']) > 1)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("curated").set_defaults(fn=cmd_curated)
    m = sub.add_parser("mine"); m.set_defaults(fn=cmd_mine)
    m.add_argument("--n", type=int, default=30, help="how many multi-hop drafts to keep")
    m.add_argument("--pairs", type=int, default=20, help="document pairs to draw from")
    sub.add_parser("review").set_defaults(fn=cmd_review)
    a = sub.add_parser("approve"); a.set_defaults(fn=cmd_approve)
    a.add_argument("ids", nargs="*"); a.add_argument("--all", action="store_true")
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
