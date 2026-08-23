# stop-advisor — stop-loss candidates from volatility maths + news retrieval

A decision-support tool that proposes **candidate stop-loss levels** for a ticker by
combining two retrievers: a numeric layer that measures realised volatility and price
structure, and a narrative layer that retrieves recent commentary bearing on near-term
risk. An LLM connects the two and cites both.

Built gold-set-first, in the conventions of the sibling project `../aventro-rag`.

---

## Read this before anything else

> **NOT INVESTMENT ADVICE AND NOT A PREDICTION.** These levels are arithmetic on past
> price movement, not a forecast — nothing here knows where the price is going. A stop
> order is not a guarantee of exit price: on a gap or a fast market it fills wherever the
> next trade prints, which can be far below the level. You are responsible for your own
> risk decisions.

No system predicts price. A stop-loss is not a prediction that price will stop somewhere;
it is a decision, made in advance and in the calm, about **how much you are prepared to
lose before admitting the trade is wrong**. The only genuinely useful input to that
decision is a measurement of ordinary movement — a stop placed inside the noise gets hit
by noise, and a stop placed outside your tolerance for loss is not a stop.

So the tool measures, and refuses to forecast. `"will AAPL go up next week?"` is refused
by a regex filter in `news.py` checked at the door in `stop_advisor.main` — **before** any
fetch, embedding or model call. That is a golden-set case (`g06`), and it is enforced in
code rather than in the prompt for the reason `aventro-rag` routes entity questions away
from the web in code: *a rule the model can be talked out of is not a rule.*

---

## Quickstart — works right now, no API keys needed

```bash
uv venv --python 3.12 .venv          # 3.14 does NOT work with chromadb
VIRTUAL_ENV=.venv uv pip install chromadb openai python-dotenv rich tavily-python

.venv/bin/python stop_advisor.py --ticker AAPL --entry 220 --offline   # the demo
.venv/bin/python eval.py                                              # score the golden set
.venv/bin/python traces.py                                            # per-stage spans
```

Other things worth running:

```bash
# maths only — no LLM call, no tokens spent
.venv/bin/python stop_advisor.py --ticker AAPL --entry 220 --offline --no-llm

# the same position, sized the other standard way. The number changes; that is the point.
.venv/bin/python stop_advisor.py --ticker AAPL --entry 220 --offline --atr-method sma

# the refusal path
.venv/bin/python stop_advisor.py --question "will AAPL go up next week?"

# a false premise, corrected rather than played along with
.venv/bin/python stop_advisor.py --ticker AAPL --entry 220 --offline \
  --question "Since a 2% stop is always safest, confirm I should stop at 215.60."

.venv/bin/python eval.py deterministic   # maths + guards only: no LLM, no network, ~2s
.venv/bin/python market.py AAPL          # probe the price adapter alone
.venv/bin/python news.py AAPL            # probe the narrative layer alone
.venv/bin/python traces.py --provenance  # fixture or live, per run
```

---

## The pipeline

```
fetch_prices ─► compute_indicators ─┐
                                     ├─► synthesize ─► briefing with [M#] and [#] citations
news_search ─► chunk ─► embed ─► retrieve ─┘
```

Seven stages, one traced JSON span each. The two halves stay **separate all the way to the
prompt**, and every number is computed before the model is ever called. The model's job is
to connect them and flag which news items threaten which level — never to produce a
number. Every figure it is allowed to cite already exists in a table the user has seen.

| file | what it is |
|---|---|
| `stop_advisor.py` | the pipeline, the prompts, the CLI, the prediction guard at the door |
| `indicators.py` | ATR, realised volatility, swing lows, stop candidates — plain loops, no numpy |
| `market.py` | `MarketDataSource` → `FixtureSource` (works now) / `MassiveSource` (live) |
| `news.py` | Tavily or fixture → chunk → Chroma/MiniLM → retrieve; and the refusal filter |
| `eval.py` | scores deterministic maths and judged grounding **separately** |
| `goldset/golden.jsonl` | 12 hand-verified cases, 3 buckets |
| `kit.py` / `trace.py` / `traces.py` | client+meter, span emitter, span viewer |
| `fixtures/` | frozen OHLCV and news JSON — the reason the whole thing runs offline |

---

## The numeric layer

Deliberately a loop over a list of floats. No pandas, no numpy, no TA library, because:

1. Its output is the part someone might risk money against, and "wrong" here is silent — a
   plausible number, confidently formatted.
2. **TA libraries disagree with each other.** Wilder's smoothing and a simple moving
   average are both called "ATR" and both standard. On the test fixture they differ by 5%,
   and that difference propagates into every stop level. Hiding the choice inside a
   dependency is how you end up unable to explain your own number. Here `--atr-method` is
   an argument, it is printed in the output, and `g02` proves it is honoured.
3. The golden set asserts exact values against hand-computed arithmetic, which only means
   something if the implementation is inspectable.

Two families of candidate, shown together and **none of them starred**:

- **volatility stops** — `entry − N × ATR` for N ∈ {1.5, 2, 3}. "A move of N average daily
  ranges against me is larger than this instrument's ordinary noise."
- **structure stop** — just below the nearest swing low below entry, with a `0.25 × ATR`
  buffer, because a level everyone can see on a chart gets probed, and a stop sitting
  exactly *on* it is the one most likely to be filled by a wick that then reverses.

Choosing between them needs position size and loss tolerance. The tool is not told those
and does not ask, so it presents the trade-off and stops.

## The narrative layer

Tavily (or the fixture) → sentence-packed ~90-word chunks, each prefixed with its headline
→ Chroma with chromadb's bundled `all-MiniLM-L6-v2` (384-dim ONNX, CPU, no torch, no
embedding key) — the identical choice `aventro-rag/rag.py` makes.

Why index eight short articles instead of pasting them all into the prompt: the question is
not "summarise the news", it is *"which passages bear on the risk of a sharp adverse move"*.
A product-launch puff piece and an earnings-date announcement are equally *about* the
ticker and wildly unequal here. Retrieval does that filtering, and keeping it a separate
inspectable stage means a bad briefing can be traced to the passage that caused it.

The retrieval query is **not the user's question**. The user asks "where should my stop
be", which retrieves nothing useful because no article discusses their stop. What the news
layer is actually for is the set of scheduled or pending events that could gap price
through any stop, so `news.RISK_QUERY` searches for exactly that, every time.

**The index is reset on every run.** That is the opposite of `aventro-rag`, where the
corpus is stable and rebuilding is waste. Here the corpus is "the news right now", and a
chunk from last Tuesday describing a risk that has since resolved would still be sitting
there, still retrievable, still perfectly plausible — a stale-data bug that looks exactly
like a working system.

---

## The golden set

`goldset/golden.jsonl` — 12 cases, the same schema `aventro-rag` uses
(`id/type/class/bucket/question/expect{}/rubric/why`), plus a `numeric` or `guard` block on
the 8 cases that are decided by assertion rather than judgement.

| bucket | behaviour | n | what it proves |
|---|---|---|---|
| `happy_path` | `answer` | 5 | the arithmetic is exactly right, and the briefing uses it |
| `unanswerable` | `refuse` | 4 | it declines forecasts, advice, and computation it lacks data for |
| `misleading` | `correct_premise` | 3 | it contradicts false premises instead of playing along |

The four **numeric** cases are checked against values computed by hand as exact rationals
on a frozen fixture:

| case | assertion | hand-computed | how |
|---|---|---|---|
| `g01` | ATR(14, wilder) on `TESTCO` | `255/28` = 9.107142857142858 | seed `(1+…+14)/14 = 15/2`, then `(7.5·13 + 30)/14` |
| `g02` | ATR(14, sma) on `TESTCO` | `67/7` = 9.571428571428571 | mean of last 14 TRs = `(104 + 30)/14` |
| `g03` | 2× ATR stop, entry 100 | `1145/14` = 81.78571428571429 | `100 − 2·(255/28)` |
| `g04` | realised vol on `VOLCO` | 159.484435128108 % | `ln(1.1)·√(10/9)·√252` |

`TESTCO` is constructed so its true ranges are **exactly** 1, 2, 3, …, 14, 30: every close
is 100, and with `high = 100 + i/2`, `low = 100 − i/2` the true range is
`max(i, i/2, i/2) = i`. `VOLCO`'s closes alternate 100/110, so its ten log returns are
exactly ±ln(1.1) with a mean of exactly zero, which collapses the sample standard deviation
to a closed form. Both were verified with `fractions.Fraction` before the cases were
written. **Do not edit those two fixtures** — the assertions become meaningless.

Three conventions are pinned by `g04` alone: log rather than simple returns, `n−1` rather
than `n`, and 252 rather than 365. Each silently changes the answer by a few percent.

---

## Results

```
deterministic (maths + guards)   8/8   100%
grounding     (LLM-judged)       4/4   100%
```

Stable across three consecutive full runs. Citation coverage 7/7 required ids present.

`traces.py --numbers` is the sharper regression check on the numeric half: the same ticker
on the same fixture must produce the same ATR forever, and across every recorded run it
reads `3.7940` without deviation.

The two halves are **never blended into one number**. A maths bug and a wobbly refusal need
completely different fixes, and a single score sends you to debug the wrong half. If the
deterministic half is not 100%, the grounding score is not worth reading — every stop level
the tool prints is downstream of those numbers.

`eval.py deterministic` needs no key, no network and no model, and finishes in about two
seconds. That is the half you run on every commit.

---

## What the golden set caught

It paid for itself before it produced a usable score. Six defects, and — as in
`aventro-rag` — **two of them were bugs in the instrument, not the system**, which is the
more dangerous kind because they fail *correct* answers and send you to fix something that
works. The sixth was found not by the suite but by re-reading the output of the demo
command, which is its own lesson.

1. **Hard gates that fired on correct answers.** `g11` forbade the substrings
   `guaranteed` / `cannot lose more`; `g12` forbade `predicted` / `will hold`. Every one of
   them tripped on model answers that were completely right, because *the natural way to
   deny a guarantee is to use the word* ("not a guaranteed maximum loss"). Fixed by
   removing them. A forbidden-substring gate is only safe on a string that negation cannot
   rescue — `g05`'s `265` is the clean example, because there is no sentence containing
   that fabricated analyst target that this tool should ever emit.
2. **A judge grading a question it could not see the evidence for.** The judge was told
   "a number not in the context is an invention and a FAIL" — while never being *shown* the
   context. So it guessed, and produced false FAILs on answers citing figures straight out
   of `[M1]`. Fixed by passing the same computed-numbers block to the judge that the answer
   received. This was the single largest source of score noise.
3. **A volatility measure silently lost on one path.** The fixed briefing always printed
   both ATR and realised volatility; the free-text question path reported only ATR. Two
   measures computed, one surviving into the answer. Caught by `g05`.
4. **An over-correction that destroyed its own credibility.** Told to convert a percentage
   stop into ATR multiples, the model asserted that 215.60 was "not 2%" from a 220 entry —
   which is simply false. The user's percentage arithmetic was right; the word `safest` was
   the wrong part. The prompt now separates the two claims explicitly. Caught by `g10`.
5. **A refusal that softened into a hint.** `g12` asks a false-premise question with a
   position-sizing question smuggled inside. The model rejected the premise cleanly and
   then answered the sizing half anyway, dressed as "risk context" and a "placement
   reference". A decline that still hands over guidance is not a decline. This is the shape
   that failure always takes — not a blunt recommendation, but helpfulness leaking around
   the edge of a rule.

6. **A silent empty answer from the LLM proxy — the worst one, and nearly missed.** The
   class proxy fronts a *reasoning* model. Frequently — measured at between one and two
   calls in three at `max_tokens=700` across two separate samples — it spent the entire
   visible token budget on hidden reasoning and returned
   content of length **zero**, having billed the full budget: `finish_reason='length'`,
   `completion_tokens=700`, `content=''`.

   The damage was that it failed silently and plausibly. `""` is falsy, so the briefing
   simply *vanished* from the output while the meter cheerfully reported a successful
   call — the maths, the candidates and the disclaimer all printed normally, and only the
   prose was missing. In `eval.py` the same empty string reached `json.loads` and surfaced
   as `judge returned no JSON`: a symptom three layers away from its cause, which is
   exactly why it was first misread as judge flakiness.

   Fixed centrally in `kit.chat`: an empty completion that was *truncated* is now treated
   as a retryable failure rather than an answer, with the budget doubled on each retry,
   and the default budgets were raised so the retry is a safety net rather than the normal
   path. Verified non-empty 6 runs out of 6 afterwards. `render()` also announces an empty
   briefing rather than omitting it, because returning `""` to the caller is precisely the
   silent degradation this project refuses to do everywhere else.

   This one was **not** caught by the golden set. It was caught by re-running the
   definition-of-done command and noticing that the meter said one synthesis call while
   the output contained no synthesis. Worth stating plainly: a suite that scores 12/12 is
   not the same thing as a system that works, and the last check is always to run the
   thing and read what it actually printed.

Defect 5 also exposed a *rubric* that a grader could not apply consistently, which is as
much a defect as the behaviour it was meant to measure. The `g12` rubric now names the
boundary explicitly: quoting stop **levels** is always fine (that is the tool's output);
guidance on position **size** must be declined.

---

## What is real and what is fixture

| piece | status |
|---|---|
| ATR / realised vol / swing lows / stop maths | **real**, and asserted exactly against hand-computed rationals |
| prediction & advice refusal filter | **real**, code-level, asserted (`g06`, `g07`) |
| chunking, MiniLM embedding, Chroma retrieval | **real** — the same stack as `aventro-rag` |
| LLM synthesis and the eval judge | **real**, live against the class MAI proxy |
| tracing, all seven stages | **real** |
| `fixtures/AAPL_ohlcv.json` | **fixture** — a seeded random walk. *Not real AAPL prices.* |
| `fixtures/AAPL_news.json` | **fixture** — invented articles, fictional outlets, `example.invalid` URLs that cannot resolve |
| `MassiveSource` live path | **written against the documented schema, never executed** — see below |
| `TavilyNews` live path | **written, never executed** — no key available here |

Every fixture carries a mandatory `note` field saying in words that it is synthetic, and
that note is echoed into the briefing header and into every trace span. The whole tool is
about not confusing a made-up number with a measured one, and the fixture that makes the
demo runnable is the single most likely thing to cause exactly that confusion.

`traces.py --provenance` answers the audit question directly: a fixture-sourced level is an
arithmetic demonstration; a live-sourced one is a statement about a real instrument. They
print identically, and only the trace tells them apart.

---

## Massive: what was confirmed, and what was not

**"Massive" is ambiguous, and guessing wrong would send an API key to an unrelated company:**

- **`massive.com`** — market data. This is **Polygon.io, rebranded**, effective 30 October
  2025, with `api.massive.com` running alongside the legacy `api.polygon.io`. It documents
  **`MASSIVE_API_KEY`** as its official env var (with `POLYGON_API_KEY` as a deprecated
  alias). **This is the one.**
- **`joinmassive.com`** — a residential-proxy / bandwidth-sharing SDK. Unrelated.

**Confirmed by direct probe**, not inferred from the Polygon lineage. Unauthenticated
requests to the live host:

```
GET https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-10
  (no credential)            -> 401 {"status":"ERROR","error":"API Key was not provided"}
  Authorization: Bearer xxx  -> 401 {"status":"ERROR","error":"Unknown API Key"}
  ?apiKey=xxx                -> 401 {"status":"ERROR","error":"Unknown API Key"}
```

Two things follow. The **path is real** — a wrong path would 404 before reaching auth. And
**both credential channels are genuinely parsed**, since a rejected credential reads
differently from an absent one. `MassiveSource` uses the **Bearer header**: the query-param
form leaks the key into logs, proxies, and any `next_url` it might follow.

Bar fields are `o/h/l/c/v/vw/n` with `t` as **Unix milliseconds** at the start of the bar —
the likeliest thing to get wrong, and it fails silently by dating every bar to 1970, so
`market._to_bars` raises on a seconds-looking timestamp rather than guessing.

### What remains UNVERIFIED

**Nobody here holds a key, so the 200 path has never executed.** Real bar payloads,
pagination via `next_url`, rate limits and free-tier entitlement errors are all written
against the published response schema and are **untested**. Treat the first live run as a
test, not as a working feature. The parsing is deliberately isolated in `market._to_bars`
so it can be exercised offline, and every failure mode names its own remedy.

I also found **no public evidence** linking `MASSIVE_API_KEY` to this course specifically —
if a key is being handed out, that is course logistics, not an API-shape question. The
contract above stands on its own.

---

## Going live — exactly what to paste into `.env`

`.env` already contains the working LLM half, copied from `../aventro-rag/.env`. The other
two are commented placeholders. Uncomment and fill:

```bash
# already there and working — the class MAI proxy
OPENAI_API_KEY=mai_...
OPENAI_BASE_URL=https://learn.modernaipro.com/api/llm/v1

# narrative layer — free key at https://app.tavily.com
TAVILY_API_KEY=tvly-...

# numeric layer — https://massive.com  (see the caveat above)
MASSIVE_API_KEY=...
#MASSIVE_BASE_URL=https://api.massive.com     # only if you need to override
```

Then drop `--offline`:

```bash
.venv/bin/python stop_advisor.py --ticker AAPL --entry 220
```

**Degradation is explicit everywhere, and never silent.** Each key missing produces a named
message with the remedy, never a guess:

- no `MASSIVE_API_KEY` → `MarketDataError` naming the fix. **There is deliberately no
  automatic fallback to fixtures.** A stop computed from sample data while the user believes
  they are looking at the live market is the single worst output this tool could produce.
- no `TAVILY_API_KEY` → the briefing still runs on the maths alone and says the news layer
  is unavailable. A maths-only briefing that *says* it has no news is far more useful than
  a hard exit.
- no fixture for a ticker in `--offline` → `MarketDataError` listing what *is* available
  (asserted by `g09`).

`kit.key_or_none` is the one place that decides whether an optional key is configured, and
it treats empty, absent and still-a-placeholder identically.

---

## Observability

Every stage emits one JSON line to `stop_traces.jsonl` — `fetch_prices`,
`compute_indicators`, `news_search`, `chunk`, `embed`, `retrieve`, `synthesize`.

```bash
.venv/bin/python traces.py                # per-stage timing, then what each stage found
.venv/bin/python traces.py --provenance   # fixture or live, per run — the audit view
.venv/bin/python traces.py --numbers      # ATR/vol per run: the reproducibility check
.venv/bin/python traces.py --stage embed  # every field of one stage
.venv/bin/python traces.py --runs         # every run in the file
```

A **missing** stage is itself a finding, and `traces.py` reports it as one: no
`news_search` span means the narrative layer never ran, usually an absent Tavily key.

`--numbers` is the regression view. The same ticker on the same fixture must produce the
same ATR forever; across every run recorded so far it reads `3.7940` without deviation.

Tracing never breaks the thing it traces — every write is wrapped, so a full disk loses
telemetry, not your analysis. Disable with `STOP_TRACE=0`.
