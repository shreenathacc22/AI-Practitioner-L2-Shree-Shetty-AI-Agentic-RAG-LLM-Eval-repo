"""Shared plumbing — same conventions as mai-practitioner-labs/labs/_kit.py,
trimmed to what this project needs: a client, a metered chat call, a console.

Glass-box on purpose. Nothing here is clever; open it and read it.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

console = Console()
say = console.print

MODEL = "mai"  # the class proxy picks the model; this value is ignored by design


def client() -> OpenAI:
    """Load .env and return a client on the class proxy. Fails with the fix, not a trace."""
    load_dotenv()
    key = (os.environ.get("OPENAI_API_KEY", "") or os.environ.get("MAI_API_KEY", "")).strip()
    if not key or key.startswith("paste-your"):
        say("\n[red]No key found.[/red] Put OPENAI_API_KEY in .env "
            "(mint one at https://study.modernaipro.com/practice)\n")
        sys.exit(1)
    os.environ["OPENAI_API_KEY"] = key
    if not os.environ.get("OPENAI_BASE_URL", "").strip():
        os.environ["OPENAI_BASE_URL"] = "https://learn.modernaipro.com/api/llm/v1"
    return OpenAI()


@dataclass
class Meter:
    """Running cost awareness — the habit that survives the demo."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_label: dict = field(default_factory=dict)

    def add(self, usage, label: str = "call") -> None:
        if not usage:
            return
        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens or 0
        self.completion_tokens += usage.completion_tokens or 0
        row = self.by_label.setdefault(label, [0, 0])
        row[0] += 1
        row[1] += usage.total_tokens or 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def show(self) -> None:
        if not self.calls:
            return
        parts = " · ".join(f"{k} ×{v[0]} ({v[1]} tok)" for k, v in self.by_label.items())
        say(f"[dim]meter: {self.calls} calls · {self.total_tokens} tokens — {parts}[/dim]")


meter = Meter()


MAX_BUDGET = 4000        # ceiling for the auto-retry below


def chat(cli: OpenAI, messages: list[dict], label: str = "chat", **kw) -> str:
    """One metered chat call, hardened against two failure modes of this proxy.

    429 burst limit — waited out; eval suites are bursty by nature.

    EMPTY COMPLETION — the proxy fronts a REASONING model, which spends tokens
    thinking before it writes. On a hard prompt it can burn the entire budget on
    reasoning and return content='' with finish_reason='length', having billed
    every token. Measured here: at max_tokens=300 over a long retrieved context,
    5 of 5 calls came back empty.

    That failure is silent and it lies in both directions. '' is falsy, so an
    answer simply vanishes; and json.loads('') raises, which upstream reads as
    "the judge returned malformed JSON" — sending you to debug a judge that was
    never asked a question. So: detect it, and retry with a bigger budget rather
    than passing '' up the stack as if it were an answer."""
    budget = kw.get("max_tokens")
    for attempt in (1, 2, 3, 4):
        try:
            resp = cli.chat.completions.create(model=MODEL, messages=messages, **kw)
        except Exception as e:  # noqa: BLE001
            if attempt < 4 and "429" in str(e):
                say("[dim](burst limit — waiting 25s, shared classroom lane)[/dim]")
                time.sleep(25)
                continue
            raise
        meter.add(resp.usage, label)          # bill it: the tokens were spent either way
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if content or choice.finish_reason != "length" or budget is None or budget >= MAX_BUDGET:
            if not content and choice.finish_reason == "length":
                say(f"[yellow](empty completion at max_tokens={budget}; at the "
                    f"{MAX_BUDGET} ceiling — returning empty rather than guessing)[/yellow]")
            return content
        budget = min(budget * 2, MAX_BUDGET)
        kw["max_tokens"] = budget
        say(f"[dim](reasoning consumed the whole budget — retrying at "
            f"max_tokens={budget})[/dim]")
    return content
