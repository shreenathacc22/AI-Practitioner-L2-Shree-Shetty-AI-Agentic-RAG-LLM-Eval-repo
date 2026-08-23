# Aventro RAG — a small RAG system built gold-set first

Corpus: [RAG-Multi-Corpus / Aventro Motors](https://github.com/udayallu/RAG-Multi-Corpus)
(50 markdown docs + the same 50 as PDF renders). Stack: **Chroma** + **all-MiniLM-L6-v2**
(ONNX, open source, CPU, no torch) + the class LLM proxy for generation and judging.

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install chromadb pypdf openai python-dotenv rich langgraph tavily-python

.venv/bin/python rag.py build                    # index corpus/md -> chroma/
.venv/bin/python rag.py ask "..." --hybrid       # answer with citations
.venv/bin/python eval.py --hybrid answers        # score against the golden set
.venv/bin/python agent.py                        # the agent, narrating every step
.venv/bin/python traces.py                       # the pipeline flight recorder
```

## Files

| file | what it is |
|---|---|
| `rag.py` | load (md/pdf) → chunk → embed → retrieve (vector or hybrid) → answer with citations |
| `goldset.py` | build the evaluation sets: import curated, mine multi-hop, review, approve |
| `eval.py` | score retrieval and generation **separately**, plus ablations |
| `agent.py` | LangGraph agent: tools, risk tags, budget cap, human checkpoint — narrated |
| `websearch.py` | Tavily web search, kept strictly subordinate to the corpus |
| `trace.py` / `traces.py` | one JSON span per pipeline stage, and the viewer |

## The golden set

`goldset/golden.jsonl` — 12 hand-verified cases, 4 per bucket, every fact checked
against the corpus and recorded in each case's `why` field.

| bucket | behaviour | what it proves |
|---|---|---|
| `happy_path` ×4 | `answer` | it finds exact values in tables and among near-identical sections |
| `unanswerable` ×4 | `refuse` | it declines when the corpus lacks the fact, **under retrieval pressure** |
| `misleading` ×4 | `correct_premise` | it corrects false premises instead of playing along |

Two further layers exist and are scored the same way:
`goldset/curated.jsonl` (221 cases shipped with the dataset, with ground-truth
supporting facts — free retrieval ground truth) and `goldset/multihop.jsonl`
(mined cross-document cases, each verified to be genuinely multi-hop).

## Results

Scored on the 12 core cases, k=8:

| retrieval | happy | unanswerable | misleading | **score** |
|---|---|---|---|---|
| vector only | 2/4 | 4/4 | 3/4 | **9/12 (75%)** |
| **hybrid (BM25+RRF)** | 4/4 | 4/4 | 4/4 | **12/12 (100%)** |

Confirmed on the independent 221-case set — recall@5: **vector 87.8% → hybrid 94.1%**.
Hybrid is not in this system because it is fashionable; it is here because removing
it costs three cases and six points of recall.

## What the golden set actually caught

The set paid for itself before it ever produced a score:

1. **A chunker bug that silently dropped content.** Headings and their bodies share
   one block in this corpus (`## 1. …Mumbai` immediately followed by the address).
   The chunker matched the block as a heading and discarded the body — 549 of 594
   words gone from one document. Fixed; corpus-wide retention now verifies clean
   (608 body blocks, 0 missing).
2. **Chunks that did not contain their own subject.** `## Bolt Sedan` and
   `## Key Specifications` are sibling H2s, so the heading trail overwrote the model
   name and the chunk holding the Bolt's 470-litre boot space contained no
   occurrence of "Bolt". Nothing could retrieve it by name. Fixed by leading every
   chunk with its document title.
3. **A prompt that refused instead of correcting.** Told to decline when the answer
   was absent, the model met a false premise ("the Bolt is a hatchback"), failed to
   find it, and declined — when the sources *contradicted* the question.
4. **Bugs in the instrument itself**, which matter more than bugs in the system:
   a reference answer asserting "ex-showroom, Delhi" that the source never says
   (it was failing correct answers); doc-level blame that called retrieval misses
   "generation"; a judge whose unparseable output silently became `FAIL`.

## Observability

Every stage emits a span to `rag_traces.jsonl` — `ingest`, `chunk`, `embed`,
`search`, `answer`, plus `agent_step` and `tool_call`.

```bash
.venv/bin/python traces.py              # where the time went, per stage
.venv/bin/python traces.py --outliers   # documents that dominate the index
.venv/bin/python traces.py --searches   # queries and retrieval confidence
```

This is what surfaced defect #1 above: the trace reported a 26% gap between words
ingested and words indexed, which turned out to be part relocation (headings moving
into chunk prefixes) and part real loss.

## MD vs PDF

The 50 PDFs are LaTeX renders of the same 50 markdown files — 96–99% vocabulary
overlap, 1:1 filenames. Indexing both duplicates every fact and halves the
diversity of top-k. `--format md` is the default for that reason; `--format both`
exists so you can measure the cost rather than take my word for it
(`eval.py ablate`).

## Web search

`websearch.py` adds Tavily as a second retriever under a **closed-world-first** rule
enforced in code, not in the prompt:

- questions naming Aventro or any model → **corpus only**, web refused
- general context (road tax, fuel prices, EV incentives) → web allowed, every
  snippet stamped `[EXTERNAL WEB · url]`

This matters because Aventro is a *fictional* company. The web knows nothing true
about it, so an unguarded web tool would not add knowledge — it would convert the
four passing `unanswerable` cases into confident wrong answers.

```bash
echo 'TAVILY_API_KEY=tvly-...' >> .env      # from app.tavily.com
.venv/bin/python websearch.py "EV road tax Maharashtra"
```

Without a key the tool returns an explicit "unavailable" string rather than
degrading into guessing.
