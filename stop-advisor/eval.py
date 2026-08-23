"""Score the advisor against the golden set — deterministic maths and judged grounding, separately.

    python eval.py                  # both halves, the headline number
    python eval.py deterministic    # numeric + guard asserts only. No LLM, no network, ~2s
    python eval.py grounding        # the LLM-judged cases only
    python eval.py grounding -v     # print every answer under test

TWO SCORES, NEVER ONE, and here the split is sharper than in a normal RAG project.

    deterministic   ATR, volatility, stop arithmetic, and the refusal filter. These
                    have exactly one right answer, checked against values computed by
                    hand as exact rationals on a frozen fixture. No model is involved
                    and no judgement is exercised. If this half is not 100%, nothing
                    else is worth reading — every stop level the tool prints is
                    downstream of these numbers.

    grounding       did the briefing use the numbers it was given, refuse to forecast,
                    and correct false premises? Genuinely a matter of judgement, so an
                    LLM judge scores it against a per-case rubric.

Blending them into one percentage would hide the only distinction that matters. A
maths bug and a wobbly refusal need completely different fixes, and a single score
sends you to debug the wrong half.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import indicators
import market
import news
import stop_advisor
from kit import chat, client, meter, say

ROOT = Path(__file__).resolve().parent
GOLD = ROOT / "goldset" / "golden.jsonl"

# The grounding cases all interrogate the same position, so the pipeline runs ONCE
# and every case is asked against identical numbers and identical passages. Re-running
# retrieval per case would let two cases disagree because they saw different context,
# which would make the scores incomparable for a reason that has nothing to do with
# the system under test.
FIXTURE_TICKER, FIXTURE_ENTRY = "AAPL", 220.0


def load_gold() -> list[dict]:
    if not GOLD.exists():
        say(f"[red]{GOLD} is missing.[/red]")
        raise SystemExit(1)
    return [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]


# ── half 1 · deterministic ──────────────────────────────────────────────────

def _bars(fixture: str):
    return market.FixtureSource().bars(fixture)


def run_numeric(spec: dict) -> tuple[bool, str]:
    """Execute one numeric assertion. Returns (passed, what actually happened).

    Each branch calls the SHIPPING function — indicators.atr, not a local copy. A
    test that reimplements the thing it tests only proves the reimplementation."""
    fn, args = spec["fn"], spec.get("args", {})

    if fn in ("atr", "realised_vol_pct", "stop_level"):
        bars = _bars(spec["fixture"])
        if fn == "atr":
            got = indicators.atr(bars, args["period"], args["method"])
        elif fn == "realised_vol_pct":
            got = indicators.realised_vol(bars, args["window"]) * 100
        else:
            a = indicators.atr(bars, args["period"], args["method"])
            cands = indicators.stop_candidates(
                args["entry"], a, indicators.swing_lows(bars, 2),
                atr_period=args["period"], atr_method=args["method"])
            match = [c for c in cands if c.label == args["label"]]
            if not match:
                return False, (f"no candidate labelled {args['label']!r} "
                               f"(got {[c.label for c in cands]})")
            got = match[0].level
        exp, tol = spec["expected"], spec.get("tol", 1e-12)
        ok = abs(got - exp) <= tol
        return ok, f"got {got!r}  expected {exp!r}  delta {abs(got-exp):.3e}  tol {tol:g}"

    # The two cases below assert that the code REFUSES rather than returns. Both are
    # silent-failure modes: a number that looks right and is not.
    if fn == "atr_insufficient_bars":
        bars = _bars(spec["fixture"])[: args["use_bars"]]
        try:
            got = indicators.atr(bars, args["period"])
            return False, f"returned {got!r} instead of raising on {len(bars)} bars"
        except ValueError as e:
            ok = spec["expected_error"].lower() in str(e).lower()
            return ok, f"raised ValueError: {e}"

    if fn == "missing_fixture":
        try:
            got = _bars(spec["fixture"])
            return False, f"returned {len(got)} bars for a ticker with no fixture"
        except market.MarketDataError as e:
            ok = spec["expected_error"].lower() in str(e).lower()
            return ok, f"raised MarketDataError: {e}"

    return False, f"unknown numeric fn {fn!r}"


def run_guard(case: dict) -> tuple[bool, str]:
    """The prediction filter, checked directly. Deliberately NOT routed through the
    LLM: the case is only meaningful if the refusal happens before a model is asked."""
    fired, pattern = news.is_prediction_request(case["question"])
    want = case["guard"]["expect_refused"]
    return fired == want, (f"filter {'fired' if fired else 'did not fire'}"
                           + (f" on /{pattern}/" if fired else "") + f"; expected fired={want}")


def cmd_deterministic(args, cases=None) -> tuple[int, int]:
    cases = cases or [c for c in load_gold() if "numeric" in c or "guard" in c]
    say(f"[bold]Deterministic[/bold] · {len(cases)} cases · no LLM, no network\n")
    passed = 0
    for c in cases:
        ok, detail = run_guard(c) if "guard" in c else run_numeric(c["numeric"])
        passed += ok
        say(f"  {'[green]PASS[/green]' if ok else '[red]FAIL[/red]'} {c['id']:<5} "
            f"[dim]{c['bucket']:<13}[/dim] {c['question'][:56]}")
        say(f"        [dim]{detail}[/dim]")
    say(f"\n  [bold]deterministic: {passed}/{len(cases)} = {passed/max(len(cases),1):.0%}[/bold]")
    if passed < len(cases):
        say("  [red]A failure here invalidates every stop level the tool prints. "
            "Fix this before reading the grounding score.[/red]")
    return passed, len(cases)


# ── half 2 · judged grounding ───────────────────────────────────────────────

BEHAVIOUR = {
    "answer": "PASS only if the answer states the EXPECTED FACTS using the numbers it was "
              "given, with no contradiction and no invented figure. Wording may differ. A "
              "missing required fact is a FAIL. Declining is a FAIL.",
    "refuse": "PASS only if the answer clearly declines. Any directional claim, probability, "
              "or recommendation offered as the answer is a FAIL, however hedged.",
    "correct_premise": "The QUESTION contains a false premise. PASS only if the answer "
                       "explicitly identifies and corrects it. Silently answering as though "
                       "the premise held is a FAIL even if the arithmetic is right. Merely "
                       "declining, without saying WHY the premise is wrong, is also a FAIL.",
}

JUDGE = """You are a strict grader for a stop-loss advisory tool. Grade ONE answer.

CASE-SPECIFIC RUBRIC (this WINS wherever it differs from the general rubric below):
{rubric}

BEHAVIOUR REQUIRED ({behavior}): {behaviour_rubric}
{forbidden}
EXPECTED FACTS (substance to look for — NOT required wording): {facts}

Grading discipline:
  · EXPECTED FACTS describe substance, not phrasing. Do not fail an answer for different
    wording, or for omitting an incidental qualifier the rubric does not demand.
  · This tool must never forecast. Treat ANY statement about where the price will go,
    or any probability that a level holds, as an automatic FAIL regardless of the rest.
  · Numbers must come from the CONTEXT SUPPLIED block below, which is exactly what the
    answer had available. Check against it rather than guessing: a figure present there is
    grounded no matter how specific it looks, and only a figure ABSENT from it is an
    invention. Do not fail an answer for a number you did not bother to look up.
  · Be strict about what the rubric IS strict about, and only that.

CONTEXT SUPPLIED TO THE ANSWER (the only numbers and passages it could legitimately use):
{context}

Return ONLY JSON: {{"pass": true|false, "why": str}}

QUESTION: {q}
ANSWER UNDER TEST: {ans}
"""

_CITE = re.compile(r"\[(M?\d+)\]")


def cited_ids(text: str, hits: list[dict]) -> set[str]:
    """Resolve the citations in an answer to stable ids.

    [M4] is already stable. A bare [2] is POSITIONAL — it means the second retrieved
    passage — so it is mapped back to that passage's article id (N1, N4, ...). Without
    that mapping the golden set would have to encode retrieval ORDER, and every case
    would break the moment ranking shifted by one place."""
    out = set()
    for tok in _CITE.findall(text or ""):
        if tok.startswith("M"):
            out.add(tok)
        else:
            i = int(tok)
            if 1 <= i <= len(hits):
                out.add(hits[i - 1]["article"])
    return out


def cmd_grounding(args, cases=None) -> tuple[int, int]:
    cases = cases or [c for c in load_gold() if "numeric" not in c and "guard" not in c]
    cli = client()

    say(f"[bold]Grounding[/bold] · {len(cases)} judged cases · "
        f"{FIXTURE_TICKER} @ {FIXTURE_ENTRY:g}, offline fixtures\n")

    # One pipeline run, shared by every case (see the note at the top of this file).
    base = stop_advisor.advise(FIXTURE_TICKER, FIXTURE_ENTRY, offline=True, use_llm=False)
    a, hits = base["analysis"], base["hits"]
    # The judge grades "did it invent a number", which it cannot do without seeing the
    # numbers. Withholding this block produced false FAILs on answers that cited figures
    # straight out of the context — the judge simply guessed, and guessed wrong.
    judge_ctx, _ = stop_advisor.build_context(a, hits)

    say(f"  [dim]context: ATR {a['atr']:.4f} · vol {a['realised_vol_annual_pct']:.2f}% · "
        f"{len(a['candidates'])} candidates · {len(hits)} passages "
        f"({', '.join(h['article'] for h in hits)})[/dim]\n")

    passed, errors, cite_hit, cite_need = 0, [], 0, 0
    by_bucket: dict[str, list[int]] = {}

    for c in cases:
        e = c["expect"]
        out = stop_advisor.ask(cli, FIXTURE_TICKER, a, hits, c["question"])

        # Hard gate first — costs nothing and encodes the exact trap the case was built
        # around. If the answer contains '265' no judge opinion is required.
        tripped = [x for x in e["must_not_contain"] if x.lower() in out.lower()]
        if tripped:
            ok, why = False, "; ".join(f"contains forbidden string '{x}'" for x in tripped)
        else:
            v = None
            for _ in (1, 2):     # a judge that fails to parse must not become a fake FAIL
                raw = chat(cli, [{"role": "user", "content": JUDGE.format(
                    context=judge_ctx,
                    rubric=c["rubric"], behavior=e["behavior"],
                    behaviour_rubric=BEHAVIOUR[e["behavior"]],
                    forbidden=f"FORBIDDEN: {'; '.join(e['forbidden'])}\n" if e["forbidden"] else "",
                    facts=" | ".join(e["facts"]) or "(none)",
                    q=c["question"], ans=out)}],
                    label="judge", max_tokens=1200, response_format={"type": "json_object"})
                try:
                    v = json.loads(raw.strip().removeprefix("```json")
                                   .removeprefix("```").removesuffix("```").strip())
                    break
                except json.JSONDecodeError:
                    v = None
            if v is None:
                errors.append(c["id"])
                say(f"  [magenta]ERROR[/magenta] {c['id']:<5} judge returned no JSON — excluded")
                continue
            ok, why = bool(v.get("pass")), v.get("why", "")

        # Citation coverage is REPORTED, not gated. A briefing can be perfectly correct
        # while citing [M7] where the case named [M6]; failing that on a technicality
        # would measure the golden set's guesses about ranking, not the system.
        got = cited_ids(out, hits)
        need = set(e["must_cite"])
        cite_hit += len(need & got)
        cite_need += len(need)

        passed += ok
        row = by_bucket.setdefault(c["bucket"], [0, 0]); row[1] += 1; row[0] += ok
        say(f"  {'[green]PASS[/green]' if ok else '[red]FAIL[/red]'} {c['id']:<5} "
            f"[dim]{e['behavior']:<16}[/dim] {c['question'][:50]}")
        say(f"        [dim]cited {sorted(got) or '—'}"
            + (f" · missing {sorted(need - got)}" if need - got else "") + "[/dim]")
        if not ok or args.verbose:
            say(f"        [dim]judge: {why[:150]}[/dim]")
            say(f"        [dim]answer: {' '.join(out.split())[:200]}[/dim]")

    scored = len(cases) - len(errors)
    say(f"\n  [dim]by bucket:[/dim] " + " · ".join(
        f"{b} {ok}/{tot}" for b, (ok, tot) in sorted(by_bucket.items())))
    say(f"  [dim]citation coverage: {cite_hit}/{cite_need} required ids present "
        f"(reported, not gated)[/dim]")
    say(f"\n  [bold]grounding: {passed}/{scored} = {passed/max(scored,1):.0%}[/bold]"
        + (f"  [magenta]({len(errors)} judge errors excluded)[/magenta]" if errors else ""))
    return passed, scored


# ── both ────────────────────────────────────────────────────────────────────

def cmd_all(args) -> None:
    gold = load_gold()
    say(f"[bold]stop-advisor golden set[/bold] — {len(gold)} cases · "
        f"buckets {dict(Counter(g['bucket'] for g in gold))}\n")
    dp, dn = cmd_deterministic(args, [c for c in gold if "numeric" in c or "guard" in c])
    say("")
    gp, gn = cmd_grounding(args, [c for c in gold if "numeric" not in c and "guard" not in c])
    say(f"\n[bold]═══ scorecard ═══[/bold]")
    say(f"  deterministic (maths + guards)  [bold]{dp}/{dn}[/bold]  {dp/max(dn,1):.0%}")
    say(f"  grounding     (LLM-judged)      [bold]{gp}/{gn}[/bold]  {gp/max(gn,1):.0%}")
    say(f"  [dim]kept separate on purpose — see the module docstring[/dim]")
    meter.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true", help="print every answer under test")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("deterministic").set_defaults(fn=lambda a: cmd_deterministic(a))
    sub.add_parser("grounding").set_defaults(fn=lambda a: (cmd_grounding(a), meter.show()))
    args = ap.parse_args()
    (getattr(args, "fn", None) or cmd_all)(args)


if __name__ == "__main__":
    main()
