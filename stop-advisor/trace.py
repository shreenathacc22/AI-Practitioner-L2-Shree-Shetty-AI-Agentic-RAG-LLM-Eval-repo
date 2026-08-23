"""Observability for the advisor pipeline — one JSON line per stage, append-only.

Modelled on aventro-rag/trace.py, but the questions this pipeline fails on are
different, so the spans are different. A RAG trace answers "why did this query
miss". A risk-tool trace has to answer:

    where did the numbers come from?  -> source=fixture|massive, and how many bars
    is the ATR trustworthy?           -> n_bars vs period; a 14-period ATR on 9
                                         bars is arithmetic, not a measurement
    did the news layer see anything?  -> n_results, and whether the key was live
    what went into the LLM?           -> the exact numbers and passages cited

That last one is the point. This tool emits levels that someone might risk money
against, so "what did the model actually have in front of it" must be recoverable
after the fact, not reconstructed from memory.

    file    stop_traces.jsonl next to this module  (STOP_TRACE_FILE to move it)
    format  JSON Lines — one self-contained object per line, append-only, so a
            crash never corrupts it and `tail -f` shows the pipeline live
    off     STOP_TRACE=0

Stages: fetch_prices, compute_indicators, news_search, chunk, embed, retrieve, synthesize.

Read it with:  python traces.py            (or jq / pandas / DuckDB)

Design rule copied from aventro-rag: tracing must never break the thing it traces.
Every write is wrapped; a full disk loses telemetry, not your analysis.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# Pinned to the module directory, not the CWD: eval.py and stop_advisor.py get run
# from wherever, and a trace file that lands in three different places is not a
# flight recorder, it is confetti.
TRACE_FILE = Path(os.environ.get(
    "STOP_TRACE_FILE", str(Path(__file__).resolve().parent / "stop_traces.jsonl")))
RUN_ID = uuid.uuid4().hex[:8]        # groups every span from one process run
_enabled = os.environ.get("STOP_TRACE", "1") != "0"

# The order stages are expected to occur in — traces.py prints by this, and a
# missing stage is itself a finding ("no news_search span" = the key was absent).
STAGES = ["fetch_prices", "compute_indicators", "news_search",
          "chunk", "embed", "retrieve", "synthesize"]


def emit(stage: str, **fields) -> None:
    """Append one span. Never raises."""
    if not _enabled:
        return
    try:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run": RUN_ID,
            "stage": stage,
            **fields,
        }
        with TRACE_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 — telemetry must not kill the pipeline
        pass


@contextmanager
def span(stage: str, **fields):
    """Time a stage and emit it. Fields added to the yielded dict are merged in,
    so a stage can report what it DISCOVERED, not just what it was told:

        with span("compute_indicators", ticker=t) as s:
            ind = indicators(bars)
            s["atr14"] = ind.atr
    """
    extra: dict = {}
    t0 = time.perf_counter()
    err = None
    try:
        yield extra
    except Exception as e:  # noqa: BLE001
        err = e
        raise
    finally:
        emit(stage, ms=round((time.perf_counter() - t0) * 1000, 1),
             ok=err is None,
             error=f"{type(err).__name__}: {err}" if err else None,
             **fields, **extra)


def stats(values: list[float]) -> dict:
    """Min/median/max/total for a list — the shape questions every stage answers."""
    if not values:
        return {"n": 0}
    s = sorted(values)
    return {"n": len(s), "min": s[0], "median": s[len(s) // 2], "max": s[-1],
            "total": round(sum(s), 6)}
