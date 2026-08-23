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


def chat(cli: OpenAI, messages: list[dict], label: str = "chat", **kw) -> str:
    """One metered chat call. Waits out a burst-limit 429 once — eval suites are bursty."""
    for attempt in (1, 2, 3):
        try:
            resp = cli.chat.completions.create(model=MODEL, messages=messages, **kw)
            break
        except Exception as e:  # noqa: BLE001
            if attempt < 3 and "429" in str(e):
                say("[dim](burst limit — waiting 25s, shared classroom lane)[/dim]")
                time.sleep(25)
                continue
            raise
    meter.add(resp.usage, label)
    return (resp.choices[0].message.content or "").strip()
