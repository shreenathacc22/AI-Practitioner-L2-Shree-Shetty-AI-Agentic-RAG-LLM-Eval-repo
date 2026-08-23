"""Aventro Motors RAG — Chroma + all-MiniLM-L6-v2, deliberately small.

    python rag.py build                     # index corpus/md  (the default)
    python rag.py build --format pdf        # index corpus/pdf instead
    python rag.py build --format both       # index both (duplicates every fact — see README)
    python rag.py ask "How do I change a tyre?"
    python rag.py search "adaptive cruise control" -k 5    # retrieval only, no LLM

Two moving parts and nothing else:
  · retrieval  — Chroma persists the vectors; its bundled MiniLM does the embedding,
                 so there is no embedding API key and no torch install.
  · generation — the class proxy answers ONLY from retrieved chunks, with citations.

Every knob that matters (chunk size, k, format) is a CLI flag, because the point
of eval.py is to move those knobs and watch the score.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from kit import chat, client, meter, say
from trace import emit, span, stats

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
DB_DIR = ROOT / "chroma"

# all-MiniLM-L6-v2 (384-dim), shipped with chromadb as ONNX — open source,
# runs on CPU, ~80MB downloaded once into ~/.cache/chroma. No torch.
EMBED = embedding_functions.DefaultEmbeddingFunction()


# ── load ────────────────────────────────────────────────────────────────────

def load_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf(path: Path) -> str:
    """Extract text page by page. Note what this LOSES versus the .md twin:
    tables flatten into ragged lines and LaTeX hyphenation leaves 'mod- ern'.
    We repair the hyphenation; the tables are gone for good. That gap is the
    whole argument for preferring structured source formats when you have them."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    raw = "\n".join((page.extract_text() or "") for page in reader.pages)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)       # rejoin hyphen-split words
    emit("parse_detail", doc=path.name, parser="pypdf", pages=len(reader.pages),
         chars=len(text), hyphen_repairs=len(re.findall(r"(\w)-\n(\w)", raw)),
         empty_pages=sum(1 for pg in reader.pages if not (pg.extract_text() or "").strip()))
    return text


def documents(fmt: str) -> list[tuple[str, str]]:
    """Return [(doc_id, text)]. doc_id keeps the format so citations stay unambiguous."""
    out: list[tuple[str, str]] = []
    with span("ingest", fmt=fmt) as s:
        if fmt in ("md", "both"):
            for p in sorted((CORPUS / "md").glob("*.md")):
                with span("ingest_doc", doc=f"md/{p.name}", parser="markdown") as d:
                    text = load_md(p)
                    d.update(bytes=p.stat().st_size, chars=len(text), words=len(text.split()))
                out.append((f"md/{p.name}", text))
        if fmt in ("pdf", "both"):
            for p in sorted((CORPUS / "pdf").glob("*.pdf")):
                with span("ingest_doc", doc=f"pdf/{p.name}", parser="pypdf") as d:
                    text = load_pdf(p)
                    d.update(bytes=p.stat().st_size, chars=len(text), words=len(text.split()))
                out.append((f"pdf/{p.name}", text))
        s.update(n_docs=len(out), total_words=sum(len(t.split()) for _, t in out))
    if not out:
        say(f"[red]No documents found for format '{fmt}' under {CORPUS}[/red]")
        sys.exit(1)
    return out


# ── chunk ───────────────────────────────────────────────────────────────────

def chunk(text: str, target_words: int = 180, min_words: int = 30,
          title: str | None = None) -> list[str]:
    """Paragraph packing with heading inheritance.

    The corpus is markdown with '# doc / ## section' structure, and a chunk that
    says '12.5 lakh' without carrying 'Aventro Grand SUV > Electric ZX' above it
    is unretrievable noise. So every chunk is prefixed with the heading trail it
    was found under — the cheapest context repair there is, and it costs nothing
    at query time.

    Two rules earn their keep on this corpus specifically:
      · '---' rules are dropped. These docs separate every section with one, and
        naively they become 82 chunks whose entire content is a horizontal rule —
        22% of the index, all of it competing for top-k slots.
      · a heading only ends a chunk once it has min_words of body, so a one-line
        subsection merges forward instead of becoming its own useless fragment.

    Chunk size is a knob: tune it against the golden set, not against taste.
    """
    RULE = re.compile(r"^([-*_]\s*){3,}$")            # ---, ***, ___ separators

    chunks: list[str] = []
    buf: list[str] = []
    words = 0
    trail: dict[int, str] = {}      # heading level -> most recent text at that level
    buf_trail: dict[int, str] = {}  # the trail as it was when this buffer started

    def flush() -> None:
        nonlocal buf, words
        if not buf:
            return
        # The document title leads every chunk. Without it, sibling H2 sections
        # erase the entity: 'Aventro Bolt Sedan.md' has '## Bolt Sedan' and
        # '## Key Specifications' at the SAME level, so the trail overwrote
        # 'Bolt Sedan' and the chunk holding the Bolt's 470-litre boot space
        # contained no occurrence of the word 'Bolt' at all. Neither embeddings
        # nor BM25 can retrieve an entity that is not in the text.
        parts = ([title] if title else []) + [buf_trail[k] for k in sorted(buf_trail)]
        seen_p, head_parts = set(), []
        for x in parts:                       # de-duplicate: title often repeats H1
            if x and x.lower() not in seen_p:
                seen_p.add(x.lower())
                head_parts.append(x)
        head = " > ".join(head_parts)
        body = "\n".join(buf)
        chunks.append(f"{head}\n{body}" if head else body)
        buf, words = [], 0

    for para in (p.strip() for p in text.split("\n\n")):
        if not para or RULE.match(para):
            continue

        # A heading and its body often share one block, with no blank line between
        # them ('## 1. Service Center, Mumbai' immediately followed by the address
        # bullets). Treating the whole block as a heading DISCARDS that body — it
        # silently cost this corpus 549 words from one document alone. So peel the
        # leading heading lines off, then keep whatever remains as content.
        lines = para.split("\n")
        i = 0
        while i < len(lines) and (m := re.match(r"^(#{1,6})\s+(.*)", lines[i].strip())):
            level, title = len(m.group(1)), m.group(2).strip()
            if words >= min_words:
                flush()
            trail = {k: v for k, v in trail.items() if k < level}
            trail[level] = title
            if not buf:
                buf_trail = dict(trail)
            i += 1

        body = "\n".join(lines[i:]).strip()
        if not body or RULE.match(body):
            continue

        if not buf:
            buf_trail = dict(trail)
        n = len(body.split())
        if words + n > target_words and words >= min_words:
            flush()
            buf_trail = dict(trail)
        buf.append(body)
        words += n

    flush()
    return [c for c in chunks if len(c.split()) > 5]   # drop stubs


# ── index ───────────────────────────────────────────────────────────────────

def collection(fmt: str, reset: bool = False):
    db = chromadb.PersistentClient(path=str(DB_DIR))
    name = f"aventro_{fmt}"
    if reset:
        try:
            db.delete_collection(name)
        except Exception:  # noqa: BLE001 — absent is the same as deleted
            pass
    return db.get_or_create_collection(
        name=name, embedding_function=EMBED, metadata={"hnsw:space": "cosine"})


def build(fmt: str, target_words: int) -> None:
    docs = documents(fmt)
    col = collection(fmt, reset=True)

    ids, texts, metas = [], [], []
    with span("chunk_all", fmt=fmt, target_words=target_words) as ca:
        for doc_id, text in docs:
            with span("chunk", doc=doc_id, target_words=target_words) as c_s:
                cs = chunk(text, target_words, title=Path(doc_id).stem.strip())
                sizes = [len(c.split()) for c in cs]
                # words_in vs words_out is the chunker's audit trail: a big gap means
                # the chunker DROPPED content (rules, stubs) — visible here, invisible later.
                c_s.update(n_chunks=len(cs), words_in=len(text.split()),
                           words_out=sum(sizes), chunk_words=stats(sizes))
            for i, c in enumerate(cs):
                ids.append(f"{doc_id}#{i}")
                texts.append(c)
                metas.append({"doc": doc_id, "chunk": i})
        allsizes = [len(t.split()) for t in texts]
        ca.update(n_chunks=len(ids), chunk_words=stats(allsizes),
                  n_oversize=sum(1 for x in allsizes if x > target_words))

    with span("embed_all", model="all-MiniLM-L6-v2", dim=384, n_chunks=len(ids)) as ea:
        for s in range(0, len(ids), 500):              # batch: MiniLM on CPU
            batch = ids[s:s+500]
            with span("embed_batch", model="all-MiniLM-L6-v2", batch=s // 500,
                      n=len(batch), words=sum(len(t.split()) for t in texts[s:s+500])):
                col.add(ids=batch, documents=texts[s:s+500], metadatas=metas[s:s+500])
            say(f"  embedded {min(s+500, len(ids))}/{len(ids)}")
        ea.update(collection=col.name, count=col.count())

    sizes = [len(t.split()) for t in texts]
    say(f"\n[green]built[/green] '{col.name}' — {len(docs)} docs -> {len(ids)} chunks")
    say(f"  {min(sizes)}–{max(sizes)} words per chunk (avg {sum(sizes)//len(sizes)}, "
        f"target {target_words}) · 384-dim · persisted to {DB_DIR.name}/")


# ── retrieve + answer ───────────────────────────────────────────────────────

_STORE: dict[str, list] = {}          # fmt -> (ids, texts, docs), loaded once


def _store(col, fmt: str):
    """Pull every chunk out of Chroma once so BM25 can score them. 352 chunks is
    nothing; at corpus scale you would use a real inverted index instead."""
    if fmt not in _STORE:
        d = col.get()
        _STORE[fmt] = (d["documents"], [m["doc"] for m in d["metadatas"]])
    return _STORE[fmt]


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9$%.-]+", s.lower())


_TOK_CACHE: dict[int, tuple] = {}


def _tokenized(texts: list[str]):
    """Tokenising 352 chunks on every query made hybrid search 100x slower than it
    needed to be. The corpus does not change between queries, so tokenise once."""
    key = id(texts)
    if key not in _TOK_CACHE:
        docs = [tokenize(t) for t in texts]
        avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
        df: dict[str, int] = {}
        for d in docs:
            for term in set(d):
                df[term] = df.get(term, 0) + 1
        _TOK_CACHE[key] = (docs, avgdl, df)
    return _TOK_CACHE[key]


def bm25_scores(query: str, texts: list[str]) -> list[float]:
    """BM25 — the keyword algorithm that still earns its keep. No library, so you
    can read every term. Embeddings blur exact strings ('Bolt', 'ZX', '470'); BM25
    is built on them. Different failure modes are exactly why fusing the two works."""
    k1, b = 1.5, 0.75
    docs, avgdl, dfmap = _tokenized(texts)
    n = len(docs)
    scores = [0.0] * n
    for term in set(tokenize(query)):
        df = dfmap.get(term, 0)
        if df == 0:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(term)
            if tf:
                scores[i] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(d) / avgdl))
    return scores


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal-rank fusion: agreement between two searchers beats either score.
    Correlated errors are rare, so a chunk both rankers like is strong evidence."""
    points: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            points[idx] = points.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(points, key=points.get, reverse=True)


def search(query: str, k: int = 4, fmt: str = "md", hybrid: bool = False) -> list[dict]:
    col = collection(fmt)
    if col.count() == 0:
        say(f"[red]Collection 'aventro_{fmt}' is empty — run: python rag.py build --format {fmt}[/red]")
        sys.exit(1)
    if hybrid:
        return _search_hybrid(col, query, k, fmt)
    with span("search", query=query[:120], k=k, fmt=fmt, method="vector") as s:
        r = col.query(query_texts=[query], n_results=k)
        hits = [{"doc": m["doc"], "text": t, "distance": d}
                for m, t, d in zip(r["metadatas"][0], r["documents"][0], r["distances"][0])]
        # distances are the retrieval confidence signal: a top hit at 0.6 means
        # "nothing in the corpus was close", which is the shape of an unanswerable query.
        s.update(docs=[h["doc"] for h in hits],
                 distances=[round(h["distance"], 4) for h in hits],
                 top_distance=round(hits[0]["distance"], 4) if hits else None,
                 n_distinct_docs=len({h["doc"] for h in hits}))
    return hits


def _search_hybrid(col, query: str, k: int, fmt: str) -> list[dict]:
    texts, docs = _store(col, fmt)
    with span("search", query=query[:120], k=k, fmt=fmt, method="hybrid") as s:
        n = len(texts)
        r = col.query(query_texts=[query], n_results=min(n, max(k * 5, 40)))
        pos = {t: i for i, t in enumerate(r["documents"][0])}
        sem = sorted(range(n), key=lambda i: pos.get(texts[i], n))
        bm = bm25_scores(query, texts)          # score ONCE, not once per comparison
        lex = sorted(range(n), key=lambda i: bm[i], reverse=True)
        fused = rrf([sem, lex])[:k]
        hits = [{"doc": docs[i], "text": texts[i], "distance": float("nan")} for i in fused]
        s.update(docs=[h["doc"] for h in hits], distances=[],
                 top_distance=None, n_distinct_docs=len({h["doc"] for h in hits}),
                 sem_top=[docs[i] for i in sem[:3]], lex_top=[docs[i] for i in lex[:3]])
    return hits


def answer(cli, query: str, hits: list[dict], label: str = "rag") -> str:
    ctx = "\n\n".join(f"[{i+1}] ({h['doc']})\n{h['text']}" for i, h in enumerate(hits))
    with span("answer", label=label, query=query[:120], n_sources=len(hits),
              context_words=len(ctx.split()), sources=[h["doc"] for h in hits]):
        return _answer_call(cli, query, ctx, label)


def _answer_call(cli, query: str, ctx: str, label: str) -> str:
    return chat(cli, [
        {"role": "system", "content":
            "You answer questions about Aventro Motors using ONLY the numbered sources below.\n"
            "Cite the source like [1] or [2][3] after each claim.\n"
            "If the sources do not contain the answer, say exactly: "
            "'The provided documents do not cover this.'\n"
            # Added because the golden set caught it: a question can carry a false
            # premise ('the Bolt is a hatchback'). Without this line the model finds
            # no 'Bolt hatchback', concludes the documents do not cover it, and
            # refuses — when the sources in fact CONTRADICT the question and the
            # useful answer is to say so. Refusal and correction are different acts.
            "If the question assumes something the sources contradict, do not refuse: "
            "state plainly that the premise is wrong, give the correct fact with its "
            "citation, then answer the underlying question.\n"
            # And because g6 caught this: a source can contradict ITSELF.
            "If the sources are internally inconsistent, or cannot supply everything "
            "asked for, report exactly what they do support and say what is missing. "
            "Never invent items to complete a list.\n"
            "Do not use outside knowledge.\n\n" + ctx},
        {"role": "user", "content": query},
    ], label=label, max_tokens=350)


# ── cli ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="chunk, embed and persist the corpus")
    b.add_argument("--format", default="md", choices=["md", "pdf", "both"])
    b.add_argument("--chunk-words", type=int, default=180)

    s = sub.add_parser("search", help="retrieval only — no LLM call")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=4)
    s.add_argument("--format", default="md", choices=["md", "pdf", "both"])
    s.add_argument("--hybrid", action="store_true", help="fuse BM25 with vector search (RRF)")

    a = sub.add_parser("ask", help="retrieve then answer with citations")
    a.add_argument("query")
    a.add_argument("-k", type=int, default=4)
    a.add_argument("--format", default="md", choices=["md", "pdf", "both"])
    a.add_argument("--hybrid", action="store_true", help="fuse BM25 with vector search (RRF)")

    args = ap.parse_args()

    if args.cmd == "build":
        build(args.format, args.chunk_words)
        return

    hits = search(args.query, args.k, args.format, hybrid=args.hybrid)

    if args.cmd == "search":
        say(f"[bold]{args.query}[/bold]\n")
        for i, h in enumerate(hits, 1):
            dist = "" if h["distance"] != h["distance"] else f"  [dim](cosine dist {h['distance']:.3f})[/dim]"
            say(f"  [yellow][{i}][/yellow] {h['doc']}{dist}")
            say(f"      [dim]{h['text'][:160].replace(chr(10), ' ')}…[/dim]")
        return

    out = answer(client(), args.query, hits)
    say(f"[bold]{args.query}[/bold]\n")
    say(out)
    say(f"\n[dim]sources: {', '.join(h['doc'] for h in hits)}[/dim]")
    meter.show()


if __name__ == "__main__":
    main()
