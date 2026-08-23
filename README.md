# AI Practitioner · L2 - Shree Shetty - AI-Agentic-RAG-LLM-Eval-repo

Modern AI Pro · Level 2 · AI Practitioner. Two systems built **gold-set first** —
the evaluation set is written before the thing it measures, so "it works" is a
number on a scorecard rather than a feeling.

| project | what it is | status |
|---|---|---|
| [`aventro-rag/`](aventro-rag/) | RAG over a 50-document corpus + a LangGraph agent + LLM-as-judge evaluation | complete, **12/12** on the golden set |
| [`stop-advisor/`](stop-advisor/) | stop-loss / exit-point decision support: market data + news RAG | complete, **8/8** numeric + **4/4** grounding |

---

## `aventro-rag/` — RAG, proven

**Chroma** + **all-MiniLM-L6-v2** (ONNX, open source, CPU, no torch) over the
[Aventro Motors](https://github.com/udayallu/RAG-Multi-Corpus) corpus.

### Results (12 core cases, k=8)

| retrieval | happy_path | unanswerable | misleading | **score** |
|---|---|---|---|---|
| vector only | 2/4 | 4/4 | 3/4 | **9/12 (75%)** |
| **hybrid (BM25 + RRF)** | 4/4 | 4/4 | 4/4 | **12/12 (100%)** |

Confirmed on an independent 221-case set — recall@5 **87.8% → 94.1%**. Hybrid
search is in this system because removing it costs three cases and six points of
recall, not because it is fashionable.

### The golden set

`aventro-rag/goldset/golden.jsonl` — 12 hand-verified cases, 4 per bucket:

- **happy_path** → must `answer`: exact values in tables, and one section among
  twenty near-identical ones
- **unanswerable** → must `refuse`: the corpus genuinely lacks the fact, *while
  retrieval still returns confident-looking context*
- **misleading** → must `correct_premise`: false premises the corpus disproves

Two further layers ship alongside it: 221 curated cases with ground-truth
supporting facts, and mined multi-hop cases each verified to be genuinely
multi-hop (single-hop and "stapled conjunction" candidates are rejected, not
shipped).

### What the golden set caught

It paid for itself before producing a score:

1. **A chunker that silently dropped content.** Headings share a paragraph block
   with their body in this corpus, so the block matched as a heading and the body
   was discarded — 549 of 594 words gone from one document.
2. **Chunks that did not contain their own subject.** `## Bolt Sedan` and
   `## Key Specifications` are sibling H2s, so the heading trail overwrote the
   model name: the chunk holding the Bolt's 470-litre boot space contained no
   occurrence of "Bolt". Nothing could retrieve it by name, at any `k`.
3. **A prompt that refused instead of correcting.** Told to decline when an answer
   was absent, the model met a false premise, failed to find it, and declined —
   when the sources *contradicted* the question.
4. **Bugs in the instrument itself**, which matter more than bugs in the system: a
   reference answer asserting a fact the source never states (failing *correct*
   answers), doc-level blame misreporting retrieval misses as generation, and a
   judge whose unparseable output silently became `FAIL`.
5. **A silent empty-completion defect in the proxy layer.** The class proxy fronts a
   *reasoning* model, which spends tokens thinking before writing. On a hard prompt
   it can burn the whole budget reasoning and return `content=''` with
   `finish_reason='length'` — every token billed, nothing returned. Measured: at
   `max_tokens=300` over a long retrieved context, **5 of 5 calls came back empty**.
   The failure lies in both directions — `''` is falsy so an answer just vanishes,
   and `json.loads('')` raises, which reads upstream as "the judge returned
   malformed JSON", sending you to debug a judge that was never asked a question.
   Fixed centrally in `kit.chat`: empty-plus-`length` is retryable, doubling the
   budget to a ceiling. Recovery verified 3/3 on the prompt that failed 5/5.

### Observability

Every pipeline stage emits a span to a JSONL flight recorder — `ingest`, `chunk`,
`embed`, `search`, `answer`, `agent_step`, `tool_call`. This is what surfaced
defect #1 above.

```bash
python traces.py              # where the time went, per stage
python traces.py --outliers   # documents that dominate the index
python traces.py --searches   # queries and retrieval confidence
```

### The agent

`agent.py` — LangGraph, narrating every hop: what the model saw, what it chose,
why the router branched, what came back, what it cost. Tools carry **structural**
risk tags (`read` / `external` / `destructive`), a budget cap bounds the loop, and
destructive actions pause for a human — failing **closed** when non-interactive.

---

## `stop-advisor/` — decision support, not prediction

Stop-loss / exit-point **decision support**: a market-data feed supplies the
numbers (ATR, realised volatility, swing lows), web search supplies the narrative
(earnings, guidance, event risk), and the two are combined into stop *candidates*
with citations.

> ⚠️ **Not investment advice, and not a price predictor.** No system predicts where
> price goes. A stop-loss is a risk-management choice given volatility and
> structure. This tool computes defensible candidates from stated maths and
> surfaces event risk with sources; it does not forecast. Requests of the form
> "will this go up?" are an explicit refusal test case.

Runs in **fixture mode** with no API keys, so the pipeline is verifiable offline.
Evaluation separates **deterministic** maths (8/8 — ATR hand-checked against exact
rationals: the `TESTCO` fixture is built so its true ranges are exactly 1…14, 30,
making Wilder ATR(14) exactly `255/28`) from **LLM-judged grounding** (4/4).

The market-data layer targets **massive.com, which is Polygon.io rebranded** —
`MASSIVE_API_KEY` against `api.massive.com/v2/aggs/...`. The endpoint is verified
real (it returns `"API Key was not provided"` bare and `"Unknown API Key"` with a
credential), but **the 200 path has never executed** — no key was available. Treat
the first live run as a test. `MassiveSource` deliberately does **not** fall back to
fixtures: a stop computed from sample data while you believe you are live is the
worst possible output.

---

## Setup

```bash
uv venv --python 3.12 .venv        # 3.14 does not work with chromadb
VIRTUAL_ENV=.venv uv pip install chromadb pypdf openai python-dotenv rich langgraph tavily-python

cp aventro-rag/.env.example aventro-rag/.env    # then fill in your keys
cd aventro-rag
python rag.py build                # index corpus/md -> chroma/
python rag.py ask "..." --hybrid   # answer with citations
python eval.py --hybrid answers    # score against the golden set
python agent.py                    # the agent, narrating every step
```

**No secrets are committed.** Each project ships `.env.example`; real `.env` files
are gitignored. The vector index is rebuilt locally with `rag.py build`.

## Attribution

Corpus from [udayallu/RAG-Multi-Corpus](https://github.com/udayallu/RAG-Multi-Corpus)
(MIT), a synthetic multi-format dataset. **Aventro Motors is a fictional company** —
which is precisely why the web-search tool is barred in code from answering
questions about it.
