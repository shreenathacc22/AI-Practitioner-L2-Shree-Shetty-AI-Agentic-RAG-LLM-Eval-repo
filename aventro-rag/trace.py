"""Observability for the RAG pipeline — one JSON line per stage, append-only.

The labs trace LLM calls (labs/_kit.py -> traces.jsonl). That answers "what did
the model cost and say". It cannot answer the questions RAG actually fails on:

    why did this query miss?      -> was the right chunk ever embedded at all?
    why is the index this size?   -> which document exploded into 40 chunks?
    what did the parser drop?     -> chars_in vs chars_out on the PDF path
    where did the time go?        -> load vs chunk vs embed vs search

So every stage emits a span: ingest, chunk, embed, search, answer.

    file    rag_traces.jsonl in the project root  (RAG_TRACE_FILE to move it)
    format  JSON Lines — one self-contained object per line, append-only, so a
            crash never corrupts it and `tail -f` shows the pipeline live
    off     RAG_TRACE=0

Read it with:  python traces.py            (or jq / pandas / DuckDB)

Design rule copied from the labs: tracing must never break the thing it traces.
Every write is wrapped; a full disk loses telemetry, not your index.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

TRACE_FILE = Path(os.environ.get("RAG_TRACE_FILE", "rag_traces.jsonl"))
RUN_ID = uuid.uuid4().hex[:8]        # groups every span from one process run
_enabled = os.environ.get("RAG_TRACE", "1") != "0"


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
    so a stage can report what it discovered, not just what it was told:

        with span("chunk", doc=name) as s:
            chunks = chunk(text)
            s["n_chunks"] = len(chunks)
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


def stats(values: list[int]) -> dict:
    """Min/median/max/total for a list — the shape questions every stage answers."""
    if not values:
        return {"n": 0}
    s = sorted(values)
    return {"n": len(s), "min": s[0], "median": s[len(s) // 2], "max": s[-1], "total": sum(s)}
