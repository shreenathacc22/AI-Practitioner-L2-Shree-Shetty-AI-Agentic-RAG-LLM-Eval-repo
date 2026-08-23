"""Web search as a SECOND retriever, kept strictly subordinate to the corpus.

    export TAVILY_API_KEY=tvly-...        (or put it in .env)
    python websearch.py "EV road tax in Maharashtra"     # probe the tool directly

Why this file is careful rather than three lines calling an API:

Aventro Motors is FICTIONAL — the dataset that provides our corpus describes
itself as synthetic, with five invented companies. So the web knows nothing true
about Aventro, and anything it returns for "Aventro Storm price" is either noise
or a genuinely different company. Wiring a web tool in naively does not add
knowledge; it adds a hallucination surface, and it would quietly convert our
four passing 'unanswerable' golden cases into confident wrong answers.

So the rule here is CLOSED-WORLD FIRST:

    entity questions  (Aventro models, pricing, policies, service centres)
        -> corpus only. Never the web. If the corpus lacks it, say so.
    context questions (road tax, fuel prices, EV incentives, general how-to)
        -> web allowed, clearly labelled as external and never merged into a
           claim about Aventro.

The routing decision is made in code, not left to the prompt, for the same reason
tool risk tags are structural in agent.py: a rule the model can be talked out of
is not a rule.
"""

from __future__ import annotations

import os
import re
import sys

from dotenv import load_dotenv

from kit import say
from trace import span

# Names that make a question CLOSED-WORLD: only our corpus can answer these.
ENTITY_TERMS = [
    "aventro", "aero", "bolt", "zoom", "nova", "swift", "glory", "storm",
    "pulse", "spark", "edge", "grand", "prime", "aventrozoom",
]


def is_entity_question(q: str) -> bool:
    """True when the question is about our fictional company, so the web is barred."""
    low = q.lower()
    return any(re.search(rf"\b{t}\b", low) for t in ENTITY_TERMS)


def available() -> bool:
    load_dotenv()
    return bool(os.environ.get("TAVILY_API_KEY", "").strip())


def web_search(query: str, max_results: int = 4) -> str:
    """Search the public web. Returns labelled snippets, or an explicit refusal.

    Two guards before the network call, both deliberate:
      1. no key            -> say so plainly; never silently degrade to guessing
      2. entity question   -> refuse; the corpus is the only authority on Aventro
    """
    load_dotenv()
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        return ("error: no TAVILY_API_KEY configured. Web search is unavailable; "
                "answer from the Aventro documents only, or say the information "
                "is not available.")
    if is_entity_question(query):
        return ("refused: this question names an Aventro product or the company itself. "
                "Aventro is covered ONLY by the internal document corpus — use search_docs. "
                "Public web results about Aventro are not authoritative and must not be used.")

    try:
        from tavily import TavilyClient
    except ImportError:
        return "error: tavily-python is not installed (uv pip install tavily-python)"

    with span("web_search", query=query[:120], max_results=max_results) as s:
        try:
            r = TavilyClient(api_key=key).search(
                query=query, max_results=max_results, search_depth="basic")
        except Exception as e:  # noqa: BLE001 — a search outage must not kill the agent
            s.update(error=str(e)[:200])
            return f"error: web search failed ({type(e).__name__}). Answer from documents only."
        results = r.get("results", [])
        s.update(n_results=len(results), urls=[x.get("url", "") for x in results])

    if not results:
        return "no web results found."
    # Every snippet is stamped EXTERNAL so the model — and the reader of the final
    # answer — can never confuse a web claim with a corpus claim.
    return "\n\n".join(
        f"[EXTERNAL WEB · {x.get('url','?')}]\n{(x.get('content') or '')[:600]}"
        for x in results)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "EV road tax exemption Maharashtra 2026"
    say(f"[bold]query:[/bold] {q}")
    say(f"[dim]key configured: {available()} · entity question: {is_entity_question(q)}[/dim]\n")
    say(web_search(q))
