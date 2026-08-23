"""Stop-loss / exit-point advisor — volatility maths + news retrieval + synthesis.

    python stop_advisor.py --ticker AAPL --entry 220 --offline
    python stop_advisor.py --ticker AAPL --entry 220 --offline --atr-method sma
    python stop_advisor.py --ticker AAPL --entry 220 --offline --no-llm     # maths only
    python stop_advisor.py --question "will AAPL go up next week?"          # refused

WHAT THIS IS NOT, stated first because everything else depends on it:

    This is not a price predictor. Nothing here forecasts anything. It is not
    investment advice, and it has no idea whether you should own this stock.

What it actually does is convert two things into candidate exit LEVELS:

    1. how much this instrument has RECENTLY been moving   (ATR, realised vol)
    2. where it has RECENTLY found a floor                 (swing lows)

A stop-loss is not a prediction that price will stop there. It is a decision, made
in advance and in the calm, about how much you are prepared to lose before
admitting the trade is wrong. The only genuinely useful input to that decision is
a measurement of ordinary movement — because a stop placed inside the noise gets
hit by noise, and a stop placed outside your tolerance for loss is not a stop.

The pipeline, one traced span per stage:

    fetch_prices -> compute_indicators -> news_search -> chunk -> embed
                 -> retrieve -> synthesize

The numeric and narrative halves stay SEPARATE all the way to the prompt, and the
numbers are computed before the model is ever called. The model's job is to
connect them and to flag which news items threaten which level — never to produce
a number. Every figure it can cite already exists in a table the user has seen.
"""

from __future__ import annotations

import argparse
import sys

import indicators
import market
import news
from kit import chat, client, meter, say
from trace import span

DISCLAIMER = (
    "NOT INVESTMENT ADVICE AND NOT A PREDICTION. These levels are arithmetic on past "
    "price movement, not a forecast — nothing here knows where the price is going. A stop "
    "order is not a guarantee of exit price: on a gap or a fast market it fills wherever "
    "the next trade prints, which can be far below the level. You are responsible for your "
    "own risk decisions."
)


# ── the prompt ──────────────────────────────────────────────────────────────

SYSTEM = """You are a risk-management assistant. You do not forecast prices and you never claim to.

You are given (a) NUMBERS already computed from price history, each with an id like [M1],
and (b) NEWS PASSAGES retrieved from recent coverage, each with an id like [1].

Write a short briefing, at most 250 words, in this structure:

VOLATILITY READ — one or two sentences on how much this instrument has been moving
lately, citing the measured numbers. Say "realised" or "recent" every time; these are
backward-looking measurements.

STOP CANDIDATES — walk through the candidates you were given. For each, cite its [M#]
id, restate the arithmetic you were handed, and say what the trade-off is (a tighter
stop risks less per share but is more likely to be hit by ordinary noise; a wider one
survives noise but costs more when hit). Do NOT recommend one. The choice depends on
position size and loss tolerance, which you were not told.

NEAR-TERM RISKS TO THESE LEVELS — the specific events in the news passages that could
cause a move THROUGH a stop rather than an orderly touch of it. Cite [#] for each.
Scheduled events that create overnight gap risk deserve explicit mention, because a
resting stop cannot be filled while the market is closed.

RULES, all of them hard:
- Never state or imply where the price will go. No targets, no probabilities of reaching
  a level, no "likely to hold", no "support should hold".
- Every number you write must be one you were given. Do not compute new ones, do not
  round differently, do not infer a number from a passage.
- If a news passage contains an analyst price target, you may note that it exists but you
  must not repeat it as an expectation — a target is an opinion about value, not a forecast
  of path, and it says nothing about drawdown risk.
- If the passages do not bear on risk, say so plainly rather than manufacturing a concern.
- Cite [M#] for every number and [#] for every news claim. An uncited claim is a defect.
- Do not tell the user whether to buy, sell or hold."""

# Question mode shares every rule above but drops the fixed three-section structure,
# and adds the one behaviour the golden set exists to force. aventro-rag learned the
# same lesson: a model told only to "refuse when unsupported" meets a false premise,
# fails to support it, and refuses — when the correct act is to CONTRADICT it. Refusal
# and correction are different behaviours and the prompt has to name both.
ASK_SYSTEM = SYSTEM.split("Write a short briefing")[0] + """Answer the user's question in at most 200 words.

If the question assumes something the numbers or the passages CONTRADICT, do not play
along and do not merely decline. Say plainly that the premise is wrong, cite the number
or passage that contradicts it, then answer the underlying question. Premises this tool
sees constantly and must always correct:

- "a fixed percentage stop is safest" — what is false is "safest", NOT the user's
  percentage arithmetic. If they say 2% of 220 is 215.60 they are right and you must not
  dispute it; saying so destroys your credibility on the part that matters. The false part
  is that a percentage is a risk level at all. It is an arbitrary distance, while
  volatility is not. Convert their distance into ATR multiples with the numbers you were
  given and show what it actually is relative to this instrument's ordinary daily movement.
- "a stop guarantees my maximum loss" — it does not. A stop becomes a market order when
  touched and fills at the next available print. Gaps, especially over an earnings date,
  can fill it far below the level.
- "you predicted / this indicates price will hold" — nothing here predicts. A swing low
  is a record of where price previously turned, not a claim it will turn again.

RULES, all of them hard:
- Never state or imply where the price will go.
- Every number you write must be one you were given.
- When the question touches volatility or stop sizing, report BOTH measures you were
  given — the ATR [M1] and the realised volatility [M2]. They answer different questions
  (how far it typically travels in a day, versus how dispersed its returns have been) and
  a briefing quoting only one is incomplete.
- Cite [M#] for numbers and [#] for news claims.
- Do not tell the user whether to buy, sell or hold.
- Do not advise on position size, and do not offer sizing "frameworks", "references" or
  "context" that function as sizing guidance. Size depends on capital, tax position and
  loss tolerance you were not given. If asked, say in one sentence that it is outside what
  this tool can answer and why, then stop — do not soften the decline into a hint."""


def build_context(a: dict, hits: list[dict]) -> tuple[str, list[str]]:
    """Number the computed figures and the retrieved passages into one prompt block.

    The numbers get [M#] ids for the same reason the passages get [#] ids: a claim
    in the briefing must be traceable to either a specific line of arithmetic or a
    specific paragraph of text. 'The stock is volatile' is unauditable; 'realised
    volatility is 34.2% annualised [M2]' can be checked against the table above it.
    """
    m: list[str] = []
    m.append(f"[M1] ATR({a['atr_period']}, {a['atr_method']}) = {a['atr']:.4f} "
             f"({a['atr_pct_of_entry']:.2f}% of the {a['entry']:.2f} entry), measured over "
             f"{a['n_bars']} daily bars from {a['first_date']} to {a['last_date']}. "
             f"This is the average true daily range — how far it typically travels in a session.")
    m.append(f"[M2] Realised volatility (annualised, last {a['vol_window']} sessions) = "
             f"{a['realised_vol_annual_pct']:.2f}%, i.e. {a['realised_vol_daily_pct']:.2f}% per day. "
             f"Backward-looking: the standard deviation of returns that already happened.")
    m.append(f"[M3] Last close {a['last_close']:.2f}; stated entry {a['entry']:.2f}.")
    for i, c in enumerate(a["candidates"], start=4):
        m.append(f"[M{i}] STOP CANDIDATE '{c['label']}' ({c['kind']}) = {c['level']:.2f}. "
                 f"Arithmetic: {c['basis']}. Risk from entry: {c['distance']:.2f} per share "
                 f"= {c['risk_pct']:.2f}% of position value.")
    if a["swing_lows"]:
        recent = a["swing_lows"][-3:]
        m.append(f"[M{len(a['candidates'])+4}] Recent swing lows (observed support, in hindsight): "
                 + "; ".join(f"{s['low']:.2f} on {s['date']} ({s['bars_ago']} bars ago)" for s in recent))

    n = [f"[{i}] ({h['article']} · {h['title']} · {h['url']} · {h['published']})\n{h['text']}"
         for i, h in enumerate(hits, 1)]
    return ("COMPUTED NUMBERS (the only figures you may cite):\n" + "\n".join(m)
            + "\n\nNEWS PASSAGES:\n" + "\n\n".join(n)), m


def synthesize(cli, ticker: str, a: dict, hits: list[dict]) -> str:
    ctx, _ = build_context(a, hits)
    with span("synthesize", ticker=ticker, n_numbers=len(a["candidates"]) + 3,
              n_passages=len(hits), context_words=len(ctx.split()),
              articles=[h["article"] for h in hits]):
        return chat(cli, [
            {"role": "system", "content": SYSTEM + "\n\n" + ctx},
            {"role": "user", "content":
                f"Ticker {ticker}, stated entry {a['entry']:.2f}. Give the stop-candidate briefing."},
        ], label="synthesize", max_tokens=700)


def ask(cli, ticker: str, a: dict, hits: list[dict], question: str) -> str:
    """Answer a free-text question against the SAME numbers and passages the briefing
    used. Sharing build_context is deliberate — if the question path could see numbers
    the briefing path could not, the golden set would be testing a different system
    than the one that ships."""
    ctx, _ = build_context(a, hits)
    with span("synthesize", ticker=ticker, mode="ask", question=question[:120],
              n_passages=len(hits), context_words=len(ctx.split()),
              articles=[h["article"] for h in hits]):
        return chat(cli, [{"role": "system", "content": ASK_SYSTEM + "\n\n" + ctx},
                          {"role": "user", "content": question}],
                    label="ask", max_tokens=600)


# ── the pipeline ────────────────────────────────────────────────────────────

def advise(ticker: str, entry: float, offline: bool, k: int = 5,
           atr_period: int = 14, atr_method: str = "wilder", vol_window: int = 20,
           swing_width: int = 2, lookback_days: int = 180,
           use_llm: bool = True, question: str | None = None) -> dict:
    """Run every stage and return the whole result. eval.py calls this too, so the
    thing under test is the thing that ships — not a reimplementation of it."""
    ticker = ticker.upper()

    bars, src = market.fetch_prices(ticker, offline, lookback_days)

    with span("compute_indicators", ticker=ticker, n_bars=len(bars),
              atr_period=atr_period, atr_method=atr_method, vol_window=vol_window) as s:
        a = indicators.analyse(bars, entry, atr_period, atr_method, vol_window, swing_width)
        s.update(atr=round(a["atr"], 6),
                 realised_vol_annual_pct=round(a["realised_vol_annual_pct"], 4),
                 n_swing_lows=len(a["swing_lows"]),
                 candidates={c["label"]: round(c["level"], 4) for c in a["candidates"]})

    result = {"ticker": ticker, "provenance": src.describe(), "analysis": a,
              "hits": [], "news_error": None, "briefing": None}

    # The narrative layer is allowed to fail without taking the numbers with it.
    # No Tavily key is the expected state on a fresh checkout, and a maths-only
    # briefing that SAYS it has no news is far more useful than a hard exit.
    try:
        arts = news.fetch_news(ticker, offline)
        col = news.index_news(ticker, arts, offline)
        result["hits"] = news.retrieve(col, news.RISK_QUERY, k)
    except news.NewsUnavailable as e:
        result["news_error"] = str(e)

    if use_llm and result["hits"]:
        result["briefing"] = (ask(client(), ticker, a, result["hits"], question) if question
                              else synthesize(client(), ticker, a, result["hits"]))
    return result


# ── output ──────────────────────────────────────────────────────────────────

def render(r: dict) -> None:
    a = r["analysis"]
    say(f"\n[bold]STOP-LOSS CANDIDATES · {r['ticker']}[/bold]  "
        f"[dim]entry {a['entry']:.2f}[/dim]")
    say(f"[dim]prices: {r['provenance']}[/dim]")
    say(f"[dim]{a['n_bars']} daily bars · {a['first_date']} -> {a['last_date']} · "
        f"last close {a['last_close']:.2f}[/dim]\n")

    say("[bold]measured — backward-looking, no forecast in any of it[/bold]")
    say(f"  ATR({a['atr_period']}, {a['atr_method']})        "
        f"[cyan]{a['atr']:.4f}[/cyan]   ({a['atr_pct_of_entry']:.2f}% of entry) "
        f"[dim]average true daily range[/dim]")
    say(f"  realised vol ({a['vol_window']}d)   "
        f"[cyan]{a['realised_vol_annual_pct']:.2f}%[/cyan]  annualised  "
        f"({a['realised_vol_daily_pct']:.2f}%/day) [dim]stdev of log returns x sqrt(252)[/dim]")
    say(f"  swing lows found     [cyan]{len(a['swing_lows'])}[/cyan]"
        + (f"   [dim]most recent: "
           + ", ".join(f"{s['low']:.2f} ({s['bars_ago']}d ago)" for s in a['swing_lows'][-3:])
           + "[/dim]" if a["swing_lows"] else "   [dim]none — no confirmed fractal low[/dim]"))

    say("\n[bold]stop candidates — the arithmetic, shown[/bold]")
    say(f"  [dim]{'level':>10}  {'risk/share':>10}  {'% of entry':>10}   derivation[/dim]")
    for i, c in enumerate(a["candidates"], start=4):
        colour = "yellow" if c["kind"] == "volatility" else "green"
        say(f"  [{colour}]{c['level']:>10.2f}[/{colour}]  {c['distance']:>10.2f}  "
            f"{c['risk_pct']:>9.2f}%   [dim]{c['label']} — {c['basis']}[/dim]  [dim]\\[M{i}][/dim]")
    say("  [dim]None of these is recommended over the others. Wider survives noise and "
        "costs more when hit;\n  tighter risks less per share and is hit more often. That "
        "trade-off is yours — it depends on\n  position size and loss tolerance, which this "
        "tool was not told.[/dim]")

    if r["news_error"]:
        say(f"\n[yellow]news layer unavailable[/yellow] — {r['news_error']}")
    elif r["hits"]:
        say(f"\n[bold]retrieved news — passages bearing on near-term risk[/bold]")
        for i, h in enumerate(r["hits"], 1):
            say(f"  [yellow][{i}][/yellow] {h['article']} · {h['title'][:66]}")
            say(f"      [dim]{h['url']} · {h['published']} · cosine {h['distance']:.3f}[/dim]")

    if r["briefing"]:
        say(f"\n[bold]── synthesis ──────────────────────────────────────────────[/bold]\n")
        say(r["briefing"])

    say(f"\n[red bold]DISCLAIMER[/red bold] [dim]{DISCLAIMER}[/dim]\n")


# ── cli ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--entry", type=float, help="your entry price (required unless --question)")
    ap.add_argument("--offline", action="store_true",
                    help="use fixtures/ instead of the live APIs — works with no keys")
    ap.add_argument("--question", help="ask something in words; prediction requests are refused")
    ap.add_argument("-k", type=int, default=5, help="news passages to retrieve")
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--atr-method", default="wilder", choices=["wilder", "sma"])
    ap.add_argument("--vol-window", type=int, default=20)
    ap.add_argument("--swing-width", type=int, default=2)
    ap.add_argument("--lookback", type=int, default=180, help="calendar days of history (live only)")
    ap.add_argument("--no-llm", action="store_true", help="skip synthesis; print the maths only")
    args = ap.parse_args()

    # The guard runs BEFORE anything else — no fetch, no embedding, no LLM call.
    # Refusing after spending money on the request would still be refusing, but it
    # would mean the rule lived downstream of the work, and rules belong at the door.
    if args.question:
        predicting, pattern = news.is_prediction_request(args.question)
        if predicting:
            say(f"\n[bold]{args.question}[/bold]\n")
            say(f"[red]{news.REFUSAL}[/red]")
            say(f"\n[dim](guard: matched /{pattern}/ in stop_advisor's prediction filter — "
                f"a code-level rule, not a prompt instruction)[/dim]\n")
            return
        say(f"[dim]not a prediction request — answering against the computed numbers[/dim]")

    if args.entry is None:
        ap.error("--entry is required (the price you got in at)")

    try:
        r = advise(args.ticker, args.entry, args.offline, k=args.k,
                   atr_period=args.atr_period, atr_method=args.atr_method,
                   vol_window=args.vol_window, swing_width=args.swing_width,
                   lookback_days=args.lookback, use_llm=not args.no_llm,
                   question=args.question)
    except (market.MarketDataError, ValueError) as e:
        say(f"\n[red]cannot produce stop levels:[/red] {e}\n")
        sys.exit(1)

    render(r)
    meter.show()


if __name__ == "__main__":
    main()
