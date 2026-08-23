"""Aventro agent — a small LangGraph agent that SHOWS ITS WORK.

    python agent.py                                   # run the guided demo
    python agent.py --ask "Book me a test drive for the Storm ZX"
    python agent.py --ask "..." --quiet               # answer only, no narration
    python agent.py --ask "..." --yes                 # auto-approve destructive tools

A chatbot answers; an agent ACTS. The difference is a loop: the model picks a
TOOL, something outside the model runs it, the result goes back in, and the model
decides again. Same shape as lab_4 — tools with risk tags, a budget cap, and a
human checkpoint before anything destructive — pointed at the Aventro corpus and
the hybrid retriever that eval.py proved.

The one thing this adds: every hop is narrated. Each step prints what the model
saw, what it chose, why the router sent it where it did, what came back, and what
it cost. An agent you cannot watch is an agent you cannot debug.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import rag
import websearch
from kit import MODEL, client, meter, say
from trace import emit, span

# ── the tools ───────────────────────────────────────────────────────────────
# Three tools, three risk tags. The tag is STRUCTURAL: it travels with the tool,
# not the prompt, so no clever wording in a user message can talk the agent past
# it. Prompt-level rules are advice; this is a gate.

BOOKINGS: list[dict] = []


def search_docs(query: str) -> str:
    """The RAG system from rag.py, used as a tool. Hybrid, because eval.py measured
    it at 100% recall on the golden set against 87.5% for vector alone."""
    hits = rag.search(query, k=5, hybrid=True)
    return "\n\n".join(f"({h['doc']}) {h['text'][:700]}" for h in hits)


def calc(expression: str) -> str:
    """Arithmetic only. The charset fence is the security boundary — an LLM that
    can be talked into anything must not be the thing deciding what eval() sees."""
    if not all(ch in "0123456789.+-*/() " for ch in expression):
        return "error: only arithmetic allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 — fenced above
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def book_test_drive(model: str, city: str, date: str) -> str:
    BOOKINGS.append({"id": f"TD-{len(BOOKINGS)+1:03}", "model": model, "city": city, "date": date})
    return f"booked {BOOKINGS[-1]['id']}: {model} test drive in {city} on {date}"


# 'external' is its own risk class, distinct from 'read'. Reading our own corpus is
# safe and authoritative; reading the open web is neither. It cannot be trusted as a
# source about Aventro, and it leaves our machine to do it.
RISK = {"search_docs": "read", "calc": "read",
        "web_search": "external", "book_test_drive": "destructive"}
IMPL = {"search_docs": search_docs, "calc": calc,
        "web_search": websearch.web_search, "book_test_drive": book_test_drive}

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search the Aventro Motors document corpus (models, pricing, service "
                       "centres, loans, safety features). Returns cited excerpts.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calc",
        "description": "Evaluate an arithmetic expression, e.g. a price difference or EMI sum.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the PUBLIC WEB for general context that is NOT about Aventro "
                       "— road tax, fuel prices, EV incentives, general driving law. "
                       "Never use this for Aventro models, pricing, policies or services: "
                       "those live only in search_docs and the web is not authoritative "
                       "about them.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "book_test_drive",
        "description": "Book a test drive. This creates a real booking.",
        "parameters": {"type": "object",
                       "properties": {"model": {"type": "string"},
                                      "city": {"type": "string"},
                                      "date": {"type": "string"}},
                       "required": ["model", "city", "date"]}}},
]

SYSTEM = (
    "You are the Aventro Motors assistant. Use search_docs for ANY factual claim about "
    "Aventro vehicles, pricing, services or policies — never answer from memory. "
    "Cite the document names the tool returns. "
    "If the documents do not cover something, say so plainly instead of guessing. "
    "If the user's question assumes something the documents contradict, correct it. "
    "Before booking a test drive you need model, city and date — ask if any is missing.\n"
    "Aventro facts come ONLY from search_docs. web_search is for external context only "
    "(taxes, fuel prices, general law); label anything from it as external and never "
    "state it as an Aventro fact. If web_search is unavailable, do not guess."
)


# ── the narrator ────────────────────────────────────────────────────────────
# This is the educational half of the file. It is deliberately separate from the
# graph so you can read the graph without it, and delete it without breaking it.

VERBOSE = [True]
STEP = [0]


def show(kind: str, *lines: str) -> None:
    if not VERBOSE[0]:
        return
    colour = {"node": "cyan", "route": "magenta", "tool": "yellow",
              "gate": "red", "result": "green", "state": "blue"}.get(kind, "white")
    def esc(x: str) -> str:                # corpus text contains [1] citations
        return str(x).replace("[", r"\[")
    say(f"  [{colour}]{kind.upper():<7}[/{colour}] {esc(lines[0])}")
    for l in lines[1:]:
        say(f"          [dim]{esc(l)}[/dim]")


def preview(messages: list[dict]) -> str:
    """What the model is about to see — the single most useful debugging fact in
    an agent, and the one almost every framework hides from you."""
    roles = [m["role"] + ("+tools" if m.get("tool_calls") else "") for m in messages]
    words = sum(len(str(m.get("content") or "").split()) for m in messages)
    return f"{len(messages)} messages ({' > '.join(roles)}) ~{words} words"


# ── the graph ───────────────────────────────────────────────────────────────
# State is a plain dict so every hop is readable. Nodes take state and return the
# fields they changed; LangGraph wires the loop: agent -> (tools? end) -> agent.

class State(TypedDict, total=False):
    messages: list
    calls: int
    budget: int
    approve_all: bool


def make_app(cli):
    def agent(state: State) -> State:
        STEP[0] += 1
        show("node", f"step {STEP[0]} · agent — the model decides what to do next",
             f"context: {preview(state['messages'])}",
             f"budget: {state['calls']}/{state['budget']} model calls used")

        # The budget cap. A loop that chooses its own next step can choose to loop
        # forever; this makes the failure honest and bounded instead of expensive.
        if state["calls"] >= state["budget"]:
            show("gate", f"budget exhausted at {state['budget']} calls — stopping honestly")
            return {"messages": state["messages"] + [{
                "role": "assistant",
                "content": "I hit my step budget before finishing — a human should take over."}]}

        with span("agent_step", step=STEP[0], calls=state["calls"]) as sp:
            resp = cli.chat.completions.create(
                model=MODEL, messages=state["messages"], tools=TOOLS_SPEC, max_tokens=700)
            meter.add(resp.usage, "agent")
            msg = resp.choices[0].message
            sp.update(tool_calls=[t.function.name for t in (msg.tool_calls or [])],
                      tokens=resp.usage.total_tokens if resp.usage else None)

        entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {"id": t.id, "type": "function",
                 "function": {"name": t.function.name, "arguments": t.function.arguments}}
                for t in msg.tool_calls]
            show("result", f"model requested {len(msg.tool_calls)} tool call(s)",
                 *[f"{t.function.name}({t.function.arguments[:80]})" for t in msg.tool_calls])
        else:
            show("result", "model produced a final answer (no tool calls)",
                 f"{(msg.content or '')[:100]}")
        if resp.usage:
            show("state", f"cost so far: {meter.total_tokens} tokens across {meter.calls} calls")
        return {"messages": state["messages"] + [entry], "calls": state["calls"] + 1}

    def tools(state: State) -> State:
        out = state["messages"][:]
        for t in state["messages"][-1]["tool_calls"]:
            name = t["function"]["name"]
            args = json.loads(t["function"]["arguments"] or "{}")
            risk = RISK.get(name, "unknown")
            show("tool", f"running {name}  (risk: {risk})", f"args: {args}")

            # The checkpoint. Reading is reversible; ACTING is not.
            if risk == "destructive":
                if state.get("approve_all"):
                    show("gate", "destructive tool — auto-approved via --yes")
                elif not approve(name, args):
                    show("gate", "DENIED at the human checkpoint")
                    out.append({"role": "tool", "tool_call_id": t["id"],
                                "content": "DENIED by human reviewer. Do not retry; tell the "
                                           "user it needs manual action."})
                    continue
                else:
                    show("gate", "approved at the human checkpoint")

            with span("tool_call", tool=name, risk=risk) as sp:
                result = IMPL[name](**args)
                sp.update(result_words=len(result.split()))
            show("result", f"{name} returned {len(result.split())} words",
                 f"{' '.join(result.split())[:110]}")
            out.append({"role": "tool", "tool_call_id": t["id"], "content": result})
        return {"messages": out}

    def route(state: State) -> str:
        wants = bool(state["messages"][-1].get("tool_calls"))
        show("route", f"last message {'HAS' if wants else 'has NO'} tool_calls "
                      f"-> {'tools' if wants else 'END'}")
        return "tools" if wants else "end"

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
    g.add_edge("tools", "agent")
    return g.compile()


def approve(name: str, args: dict) -> bool:
    """Fail CLOSED: a non-interactive run denies rather than assuming yes."""
    if not sys.stdin.isatty():
        say("          [dim](non-interactive -> fail closed, denying)[/dim]")
        return False
    return input(f"          allow {name}({json.dumps(args)[:60]})? [y/N] > ").strip().lower().startswith("y")


def run(cli, app, question: str, budget: int = 6, approve_all: bool = False) -> str:
    STEP[0] = 0
    say(f"\n[bold]you:[/bold] {question}")
    if VERBOSE[0]:
        say("[dim]  ── the loop starts. agent decides -> router sends -> tools run -> repeat ──[/dim]")
    final = app.invoke(
        {"messages": [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": question}],
         "calls": 0, "budget": budget, "approve_all": approve_all},
        {"recursion_limit": 25})
    answer = final["messages"][-1]["content"]
    say(f"\n[bold]agent:[/bold] {answer}")
    if VERBOSE[0]:
        say(f"[dim]  loop finished in {STEP[0]} model call(s) · "
            f"{len(final['messages'])} messages in final context[/dim]")
    return answer


DEMO = [
    ("A plain question — one tool, one answer.",
     "What does the top-end electric Storm cost?"),
    ("Two tools chained: the agent must retrieve BOTH prices, then do arithmetic.",
     "How much more is the Storm ZX Electric than the Storm ZX Petrol?"),
    ("A question the corpus cannot answer. Watch it decline instead of inventing.",
     "What is the ground clearance of the Storm?"),
    ("A destructive tool. The loop PAUSES for a human before it acts.",
     "Book me a test drive for the Storm ZX in Mumbai on 2026-09-15."),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ask")
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--quiet", action="store_true", help="answer only, no step narration")
    ap.add_argument("--yes", action="store_true", help="auto-approve destructive tools")
    args = ap.parse_args()
    VERBOSE[0] = not args.quiet

    cli = client()
    app = make_app(cli)

    if args.ask:
        run(cli, app, args.ask, budget=args.budget, approve_all=args.yes)
    else:
        say("[bold]Aventro agent — guided demo[/bold]")
        say("[dim]Four questions, each exercising a different part of the loop.[/dim]")
        for why, q in DEMO:
            say(f"\n[yellow]{'─'*66}[/yellow]\n[bold]{why}[/bold]")
            run(cli, app, q, budget=args.budget, approve_all=args.yes)
    meter.show()


if __name__ == "__main__":
    main()
