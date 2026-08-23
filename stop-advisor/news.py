"""The narrative layer: what recent commentary says about NEAR-TERM RISK.

Same two-implementation shape as market.py — TavilyNews when a key exists,
FixtureNews when it does not — then a small Chroma RAG over whatever came back.

Why a vector index over four to eight articles, when you could just paste them all
into the prompt: because the question is not "summarise the news", it is "which
passages bear on the risk of a sharp adverse move in the next few weeks". A
product-launch puff piece and an earnings-date announcement are equally 'about'
the ticker and wildly unequal here. Retrieval is what does that filtering, and
having it as a separate, inspectable stage means a bad briefing can be traced to
the passage that caused it.

Embeddings: chromadb's bundled all-MiniLM-L6-v2 (384-dim ONNX, CPU, no torch, no
embedding API key) — the identical choice aventro-rag/rag.py makes, for the
identical reason.

THE PREDICTION GUARD lives here too. It is a code-level refusal, checked before
any LLM call, for the same reason aventro-rag routes entity questions away from
the web in code rather than in the prompt: a rule the model can be talked out of
is not a rule.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from kit import key_or_none, say
from trace import span, stats

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
DB_DIR = ROOT / "chroma"

EMBED = embedding_functions.DefaultEmbeddingFunction()

# The standing query for the narrative layer. Not the user's question: the user
# asks "where should my stop be", which retrieves nothing useful because no
# article discusses their stop. What we actually need from the news is the set of
# scheduled or pending events that could produce a gap through any stop — so we
# search for THAT, explicitly, every time.
RISK_QUERY = ("upcoming earnings date, guidance change, downgrade, regulatory or "
              "litigation risk, supply chain problem, sector selloff, volatility, "
              "event that could cause a sharp price drop")


# ── the prediction guard ────────────────────────────────────────────────────

# Phrasings that ask for a forecast of direction or level. Deliberately broad: a
# false positive costs one refused question, a false negative costs a tool that
# answers "will it go up" as though anything could.
_PREDICT = [
    r"\bwill\b.{0,40}\b(go|move|rise|fall|drop|climb|rally|crash|tank|dip|rebound|recover)\b",
    r"\bwill\b.{0,30}\b(hit|reach|touch|break|beat|top|bottom)\b",
    r"\b(price|target|forecast|prediction|projection|outlook)\s+(for|of)\b.{0,30}\b(next|coming|tomorrow|week|month|quarter|year)\b",
    r"\b(predict|forecast|project)\b",
    r"\bwhere\b.{0,25}\b(price|stock|it)\b.{0,25}\b(be|go|head)\b",
    r"\b(is|are)\s+(it|this|the\s+stock|\w+)\s+going\s+(to|up|down)\b",
    r"\b(should i|shall i|do i)\s+(buy|sell|short|hold|invest)\b",
    r"\bhow (high|low|far)\b.{0,30}\b(go|rise|fall|drop)\b",
    r"\bguarantee|\bcertain(ty)?\b.{0,25}\b(profit|gain|return)\b",
    r"\bwhat('?s| is| will)\b.{0,25}\bworth\b.{0,25}\b(next|in a|by)\b",
]
_PREDICT_RE = [re.compile(p, re.I) for p in _PREDICT]

REFUSAL = (
    "REFUSED — this is a request for a price prediction, and this tool does not make them.\n\n"
    "No system, including this one, knows where a price is going. What this tool does is "
    "different in kind: it measures how much a stock has ALREADY been moving (ATR, realised "
    "volatility) and where it has ALREADY found support (swing lows), then converts those "
    "measurements into candidate stop distances. That is risk management — deciding in "
    "advance how much you are willing to lose if you are wrong — not forecasting.\n\n"
    "Ask instead: 'what stop levels does the recent volatility of TICKER imply for an entry "
    "at X?' — run:  python stop_advisor.py --ticker TICKER --entry X --offline"
)


def is_prediction_request(q: str) -> tuple[bool, str]:
    """True when the question asks which way the price will go. Returns the matched
    pattern too, so the refusal can be audited rather than taken on faith."""
    for rx in _PREDICT_RE:
        if rx.search(q or ""):
            return True, rx.pattern
    return False, ""


# ── sources ─────────────────────────────────────────────────────────────────

class NewsUnavailable(RuntimeError):
    """No news could be fetched. Like MarketDataError: never silently substituted."""


def _fixture_news(ticker: str) -> list[dict]:
    p = FIXTURES / f"{ticker.upper()}_news.json"
    if not p.exists():
        raise NewsUnavailable(
            f"no offline news fixture for {ticker.upper()} at {p.name}. "
            f"Available: {', '.join(sorted(x.name.split('_')[0] for x in FIXTURES.glob('*_news.json'))) or '(none)'}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return raw["articles"]


def _tavily_news(ticker: str, max_results: int) -> list[dict]:
    key = key_or_none("TAVILY_API_KEY")
    if not key:
        raise NewsUnavailable(
            "TAVILY_API_KEY is not configured, so live news retrieval is unavailable. "
            "Add it to stop-advisor/.env (free key at https://app.tavily.com), "
            "or run with --offline to use the bundled news fixture.")
    try:
        from tavily import TavilyClient
    except ImportError as e:
        raise NewsUnavailable("tavily-python is not installed (uv pip install tavily-python)") from e

    # topic="news" + a short window because a stop is a near-term decision. A
    # glowing article from 2023 is not evidence about the risk of the next month.
    r = TavilyClient(api_key=key).search(
        query=f"{ticker} stock earnings date guidance risk downgrade litigation volatility",
        max_results=max_results, search_depth="advanced", topic="news", days=45)
    return [{"id": f"W{i+1}", "title": x.get("title", ""), "url": x.get("url", ""),
             "published": x.get("published_date", "") or "",
             "content": x.get("content", "") or "", "provenance": "LIVE WEB"}
            for i, x in enumerate(r.get("results", []))]


def fetch_news(ticker: str, offline: bool, max_results: int = 8) -> list[dict]:
    """Stage 3. Returns articles, or raises with the fix. Never returns invented text."""
    with span("news_search", ticker=ticker.upper(),
              source="fixture" if offline else "tavily") as s:
        arts = _fixture_news(ticker) if offline else _tavily_news(ticker, max_results)
        s.update(n_results=len(arts), urls=[a.get("url", "") for a in arts],
                 total_words=sum(len((a.get("content") or "").split()) for a in arts))
    return arts


# ── chunk ───────────────────────────────────────────────────────────────────

def chunk_article(article: dict, target_words: int = 90) -> list[str]:
    """Pack sentences to ~90 words, prefixing every chunk with the headline.

    News bodies are short, so chunks are small — the aim is one claim per chunk,
    because "Q3 guidance was cut" and "the CFO is retiring" landing in one blob
    means retrieval cannot rank them separately and both arrive or neither does.

    The headline prefix is the same trick as rag.py's heading trail, and it earns
    its keep for the same reason: a chunk reading 'the company said it expects
    pressure to continue' names no company, no quarter, nothing retrievable. The
    headline puts the subject back into the text where the embedding can see it.
    """
    body = re.sub(r"\s+", " ", article.get("content", "") or "").strip()
    if not body:
        return []
    title = (article.get("title") or "").strip()
    sentences = re.split(r"(?<=[.!?])\s+", body)

    chunks, buf, words = [], [], 0
    for sent in sentences:
        n = len(sent.split())
        if words + n > target_words and buf:
            chunks.append(f"{title}\n{' '.join(buf)}" if title else " ".join(buf))
            buf, words = [], 0
        buf.append(sent)
        words += n
    if buf:
        chunks.append(f"{title}\n{' '.join(buf)}" if title else " ".join(buf))
    return [c for c in chunks if len(c.split()) > 4]


# ── index + retrieve ────────────────────────────────────────────────────────

def index_news(ticker: str, articles: list[dict], offline: bool):
    """Chunk, embed and persist. The collection is RESET on every run.

    That is the opposite of aventro-rag, where the corpus is stable and rebuilding
    is wasteful. Here the corpus is 'the news right now'. A chunk from last
    Tuesday's run describing a risk that has since resolved would still be sitting
    in the index, still retrievable, and still perfectly plausible — a stale-data
    bug that looks exactly like a working system.
    """
    db = chromadb.PersistentClient(path=str(DB_DIR))
    name = f"news_{ticker.upper()}_{'fixture' if offline else 'live'}"
    try:
        db.delete_collection(name)
    except Exception:  # noqa: BLE001 — absent is the same as deleted
        pass
    col = db.get_or_create_collection(name=name, embedding_function=EMBED,
                                      metadata={"hnsw:space": "cosine"})

    ids, texts, metas = [], [], []
    with span("chunk", ticker=ticker.upper(), n_articles=len(articles)) as cs:
        for a in articles:
            for i, c in enumerate(chunk_article(a)):
                ids.append(f"{a['id']}#{i}")
                texts.append(c)
                metas.append({"article": a["id"], "title": a.get("title", ""),
                              "url": a.get("url", ""), "published": a.get("published", ""),
                              "provenance": a.get("provenance", "")})
        sizes = [len(t.split()) for t in texts]
        cs.update(n_chunks=len(ids), chunk_words=stats(sizes))

    if not ids:
        raise NewsUnavailable("every article had empty content — nothing to index.")

    with span("embed", model="all-MiniLM-L6-v2", dim=384, n_chunks=len(ids)) as es:
        col.add(ids=ids, documents=texts, metadatas=metas)
        es.update(collection=name, count=col.count())
    return col


def retrieve(col, query: str = RISK_QUERY, k: int = 5) -> list[dict]:
    """Stage 6. Cosine distance comes back with the hits and is printed, because it
    is the honest confidence signal: if the nearest passage sits at 0.6, the news
    layer found nothing relevant and the briefing should lean on the math alone."""
    k = min(k, col.count())
    with span("retrieve", query=query[:120], k=k, collection=col.name) as s:
        r = col.query(query_texts=[query], n_results=k)
        hits = [{"text": t, "distance": d, **m} for m, t, d
                in zip(r["metadatas"][0], r["documents"][0], r["distances"][0])]
        s.update(articles=[h["article"] for h in hits],
                 distances=[round(h["distance"], 4) for h in hits],
                 top_distance=round(hits[0]["distance"], 4) if hits else None,
                 n_distinct_articles=len({h["article"] for h in hits}))
    return hits


if __name__ == "__main__":   # probe the narrative layer alone
    import sys
    tk = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()
    say(f"[bold]TAVILY_API_KEY configured:[/bold] {bool(key_or_none('TAVILY_API_KEY'))}")
    arts = fetch_news(tk, offline=True)
    col = index_news(tk, arts, offline=True)
    say(f"indexed {col.count()} chunks from {len(arts)} articles\n")
    for i, h in enumerate(retrieve(col), 1):
        say(f"[yellow][{i}][/yellow] {h['article']} · {h['title'][:60]} "
            f"[dim](cosine {h['distance']:.3f})[/dim]")
        say(f"    [dim]{h['text'][:130].replace(chr(10),' ')}…[/dim]")
