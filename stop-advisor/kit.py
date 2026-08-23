"""Shared plumbing — client, metered chat, console. Lifted from aventro-rag/kit.py
because the conventions should not fork between two projects in the same course.

Glass-box on purpose. Nothing here is clever; open it and read it.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

console = Console()
say = console.print

MODEL = "mai"  # the class proxy picks the model; this value is ignored by design

# load_dotenv() searches upward from the CWD, which means running this from the
# parent directory would silently pick up aventro-rag/.env instead. Pin it.
ENV_FILE = Path(__file__).resolve().parent / ".env"


def load_env() -> None:
    load_dotenv(ENV_FILE)


def client() -> OpenAI:
    """Return a client on the class proxy. Fails with the fix, not a traceback."""
    load_env()
    key = (os.environ.get("OPENAI_API_KEY", "") or os.environ.get("MAI_API_KEY", "")).strip()
    if not key or key.startswith("paste-your"):
        say("\n[red]No OPENAI_API_KEY found.[/red] Put it in stop-advisor/.env "
            "(copy the one from ../aventro-rag/.env)\n")
        sys.exit(1)
    os.environ["OPENAI_API_KEY"] = key
    if not os.environ.get("OPENAI_BASE_URL", "").strip():
        os.environ["OPENAI_BASE_URL"] = "https://learn.modernaipro.com/api/llm/v1"
    return OpenAI()


def key_or_none(name: str) -> str | None:
    """The one place that decides whether an optional key is configured.

    Returns None for absent, empty, or still-a-placeholder. Callers turn that into
    an explicit 'key not configured' message — never into a silent fallback that
    invents data. In a tool that outputs risk numbers, a quiet degradation is the
    single most dangerous failure mode there is.
    """
    load_env()
    v = os.environ.get(name, "").strip()
    if not v or v.startswith(("paste", "...", "<", "your-")):
        return None
    return v


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
    """One metered chat call, with the two failure modes this proxy actually exhibits
    handled here rather than at seven call sites.

    429 burst limit — the classroom lane is shared and eval suites are bursty. Wait it out.

    EMPTY COMPLETION WITH finish_reason='length' — the interesting one, and it cost this
    project real debugging. The proxy fronts a reasoning model: it can spend the ENTIRE
    visible token budget on hidden reasoning and return content of length zero, having
    billed the full max_tokens. Measured here at roughly one call in three for the
    briefing prompt at max_tokens=700.

    The damage is that it fails SILENTLY and plausibly. `""` is falsy, so the briefing
    simply vanished from the output while the meter cheerfully reported a successful
    call, and in eval.py the same empty string arrived at json.loads and surfaced as
    "judge returned no JSON" — a symptom three layers away from its cause.

    So an empty answer that was truncated is treated as a retryable failure, not an
    answer, and the budget is doubled on each retry. Returning "" to the caller is
    exactly the silent degradation this whole project refuses to do everywhere else.
    """
    budget = kw.get("max_tokens")
    resp = None
    for attempt in (1, 2, 3):
        try:
            resp = cli.chat.completions.create(model=MODEL, messages=messages, **kw)
        except Exception as e:  # noqa: BLE001
            if attempt < 3 and "429" in str(e):
                say("[dim](burst limit — waiting 25s, shared classroom lane)[/dim]")
                time.sleep(25)
                continue
            raise
        meter.add(resp.usage, label)          # every attempt is billed, so every attempt counts
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if content or choice.finish_reason != "length" or not budget or attempt == 3:
            return content
        budget *= 2
        kw["max_tokens"] = budget
        say(f"[dim](empty completion: the visible budget went to hidden reasoning — "
            f"retrying at max_tokens={budget})[/dim]")
    return ""
