"""The price feed, behind one interface with two implementations.

    MarketDataSource      the contract: give me daily OHLCV bars for a ticker
      +- FixtureSource    reads fixtures/*.json. Works now, offline, deterministic.
      +- MassiveSource    live HTTP against api.massive.com. Needs MASSIVE_API_KEY.

Why an interface at all for two methods: the golden set's numeric cases must run
against a byte-identical price series forever, or they stop being a regression
test and start being a market report. FixtureSource is what makes ATR assertions
possible. MassiveSource is what makes the tool useful. Neither can be the other,
so the pipeline talks to the abstraction and `--offline` picks the implementation.

── WHAT WE ESTABLISHED ABOUT MASSIVE, AND HOW ──────────────────────────────────

"Massive" is ambiguous in a way worth writing down, because guessing wrong here
would send requests carrying an API key to an unrelated company:

    massive.com       market data. Polygon.io, rebranded 30 Oct 2025. THIS ONE.
    joinmassive.com   a residential-proxy / bandwidth-sharing SDK. Not market data.

The env var name disambiguates: massive.com documents MASSIVE_API_KEY as its
official variable (with POLYGON_API_KEY as a deprecated alias).

The request shape below is NOT inferred from the Polygon lineage — it was probed
directly against the live host, unauthenticated, and the responses distinguish the
two failure modes:

    GET /v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-10
      (no credential)            -> 401 {"error":"API Key was not provided"}
      Authorization: Bearer xxx  -> 401 {"error":"Unknown API Key"}
      ?apiKey=xxx                -> 401 {"error":"Unknown API Key"}

Two things follow. The path is real — a wrong path would 404 before it ever
reached auth. And both credential channels are genuinely parsed, since a rejected
credential reads differently from an absent one. We use the Bearer header: the
query-param form leaks the key into logs, proxies and any `next_url` we might follow.

WHAT REMAINS UNVERIFIED: nobody here holds a key, so the 200 path — real bar
payloads, pagination, rate limits, entitlement errors on a free tier — has never
executed. The parsing below is written against massive.com's published response
schema. Treat the first live run as a test, not as a working feature, and read
`_to_bars` with that in mind.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from indicators import Bar
from kit import key_or_none
from trace import span

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MASSIVE_DEFAULT_BASE = "https://api.massive.com"


class MarketDataError(RuntimeError):
    """Raised when bars cannot be produced. Always carries the remedy in the message.

    There is deliberately no fallback path from MassiveSource to FixtureSource. A
    risk number silently computed from stale sample data, while the user believes
    they are looking at the live market, is worse than no number at all.
    """


class MarketDataSource:
    """Contract: daily OHLCV bars, oldest first, for one ticker.

    Implementations must guarantee, because everything downstream assumes it:
      · ascending by date, no duplicate dates
      · every bar has open/high/low/close as floats, high >= low
      · a session with no trades is ABSENT, never forward-filled — a synthetic
        flat bar has a true range of zero and would drag the ATR down for 14 days
      · fewer bars than asked for is normal (holidays, listing date) and is the
        caller's problem to notice; returning padding to hit a count is not allowed
    """

    name = "abstract"

    def bars(self, ticker: str, lookback_days: int = 180) -> list[Bar]:
        raise NotImplementedError

    def describe(self) -> str:
        """One line naming the provenance of the numbers. Printed in the briefing
        and stamped on every citation, so a reader never has to guess whether they
        are looking at the market or at a sample file."""
        raise NotImplementedError


# ── fixtures ────────────────────────────────────────────────────────────────

class FixtureSource(MarketDataSource):
    """Bars from fixtures/{TICKER}_ohlcv.json — the offline, deterministic path.

    File format is the source of truth for the fixtures, so it stays boring:

        {"ticker": "AAPL", "note": "...", "bars": [
            {"date": "2026-01-02", "open": 1, "high": 2, "low": 0.5, "close": 1.5,
             "volume": 1000}, ...]}

    `note` is mandatory in every fixture we ship and says, in words, that the data
    is synthetic. It is echoed into the briefing header. The whole tool is about
    not confusing a made-up number with a measured one; the fixture that makes the
    demo runnable is the most likely thing to cause exactly that confusion.
    """

    name = "fixture"

    def __init__(self, directory: Path = FIXTURES):
        self.dir = directory

    def path_for(self, ticker: str) -> Path:
        return self.dir / f"{ticker.upper()}_ohlcv.json"

    def bars(self, ticker: str, lookback_days: int = 180) -> list[Bar]:
        p = self.path_for(ticker)
        if not p.exists():
            available = sorted(x.name.split("_")[0] for x in self.dir.glob("*_ohlcv.json"))
            raise MarketDataError(
                f"no offline fixture for {ticker.upper()} at {p.name}. "
                f"Available fixtures: {', '.join(available) or '(none)'}. "
                f"Either pick one of those, or drop --offline and configure MASSIVE_API_KEY.")
        raw = json.loads(p.read_text(encoding="utf-8"))
        self._note = raw.get("note", "")
        bars = [Bar.from_dict(b) for b in raw["bars"]]
        # Sort here rather than trusting the file: a hand-edited fixture with one
        # row out of order produces a negative "return" and a plausible, wrong vol.
        bars.sort(key=lambda b: b.date)
        # lookback_days is a calendar window, and fixtures are short; slicing by
        # count would quietly change what the numeric golden cases are asserting on.
        return bars

    def describe(self) -> str:
        return f"FIXTURE (offline) — {getattr(self, '_note', 'synthetic sample data, not real market data')}"


# ── massive.com ─────────────────────────────────────────────────────────────

class MassiveSource(MarketDataSource):
    """Live daily bars from massive.com (the market-data company, ex-Polygon.io).

        GET {base}/v2/aggs/ticker/{TICKER}/range/1/day/{from}/{to}
            ?adjusted=true&sort=asc&limit=50000
        Authorization: Bearer {MASSIVE_API_KEY}

    Response (their documented schema; `results` is the bar array):

        {"status":"OK","ticker":"AAPL","resultsCount":2,"adjusted":true,
         "results":[{"t":1577941200000,"o":74.06,"h":75.15,"l":73.7975,
                     "c":75.0875,"v":135647456,"vw":74.6099,"n":1}],
         "next_url":"https://api.massive.com/v2/aggs/...?cursor=..."}

    `t` is Unix MILLISECONDS at the start of the bar, not seconds — the single
    likeliest thing to get wrong, and it fails silently by dating every bar to 1970.

    Uses urllib rather than requests: it is one GET with one header, and the venv
    should not grow a dependency for that.
    """

    name = "massive"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or key_or_none("MASSIVE_API_KEY")
        self.base = (base_url or key_or_none("MASSIVE_BASE_URL") or MASSIVE_DEFAULT_BASE).rstrip("/")

    def available(self) -> bool:
        return bool(self.api_key)

    def bars(self, ticker: str, lookback_days: int = 180) -> list[Bar]:
        if not self.api_key:
            raise MarketDataError(
                "MASSIVE_API_KEY is not configured, so live prices are unavailable. "
                "Add it to stop-advisor/.env (get one at https://massive.com), "
                "or run with --offline to use the bundled fixtures.")

        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=lookback_days)
        path = (f"/v2/aggs/ticker/{urllib.parse.quote(ticker.upper())}"
                f"/range/1/day/{start.isoformat()}/{today.isoformat()}")
        url = f"{self.base}{path}?" + urllib.parse.urlencode(
            {"adjusted": "true", "sort": "asc", "limit": 50000})

        rows: list[dict] = []
        pages = 0
        while url and pages < 10:          # a cursor loop with no ceiling is an outage waiting to happen
            payload = self._get(url)
            status = payload.get("status")
            if status not in ("OK", "DELAYED"):
                raise MarketDataError(
                    f"massive.com returned status={status!r} for {ticker}: "
                    f"{payload.get('error') or payload.get('message') or '(no message)'}")
            rows.extend(payload.get("results") or [])
            nxt = payload.get("next_url")
            # Their own pagination URLs can carry the key as a query param. We
            # authenticate by header, so strip it rather than emit a second copy
            # of the credential into logs and redirect chains.
            url = _strip_key(nxt) if nxt else None
            pages += 1

        if not rows:
            raise MarketDataError(
                f"massive.com returned no bars for {ticker} over the last {lookback_days} days. "
                f"Check the ticker (it is case-sensitive) and your plan's history entitlement.")
        return _to_bars(rows)

    def _get(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "stop-advisor/0.1",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            # 401/403 are the ones a user can actually fix, so name the fix.
            if e.code in (401, 403):
                raise MarketDataError(
                    f"massive.com rejected the credential (HTTP {e.code}): {body}. "
                    f"Check MASSIVE_API_KEY in stop-advisor/.env.") from e
            if e.code == 429:
                raise MarketDataError(
                    "massive.com rate-limited this key (HTTP 429). Free tiers are "
                    "typically a few calls per minute — wait, or use --offline.") from e
            raise MarketDataError(f"massive.com HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise MarketDataError(f"could not reach {self.base}: {e.reason}") from e

    def describe(self) -> str:
        return f"LIVE — massive.com daily adjusted bars ({self.base})"


def _strip_key(url: str) -> str:
    """Remove any apiKey/apikey param from a pagination URL."""
    parts = urllib.parse.urlsplit(url)
    q = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
         if k.lower() != "apikey"]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(q)))


def _to_bars(rows: list[dict]) -> list[Bar]:
    """Map massive.com aggregate rows to Bar. Isolated and tested offline
    (tests/ has no network) because this mapping is where a live integration
    silently goes wrong: a missing key, or seconds mistaken for milliseconds."""
    out = []
    for r in rows:
        try:
            ts = int(r["t"])
        except (KeyError, TypeError, ValueError) as e:
            raise MarketDataError(f"bar is missing a usable timestamp 't': {r}") from e
        # Their `t` is milliseconds. A 10-digit value would be seconds and would
        # date the bar to 1970 — loud failure beats a chart starting at the epoch.
        if ts < 10_000_000_000:
            raise MarketDataError(
                f"timestamp {ts} looks like seconds, not milliseconds — "
                f"the response schema may have changed; refusing to guess.")
        day = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat()
        try:
            out.append(Bar(date=day, open=float(r["o"]), high=float(r["h"]),
                           low=float(r["l"]), close=float(r["c"]),
                           volume=float(r.get("v", 0) or 0)))
        except (KeyError, TypeError, ValueError) as e:
            raise MarketDataError(f"malformed bar from massive.com: {r}") from e
    out.sort(key=lambda b: b.date)
    return out


# ── selection ───────────────────────────────────────────────────────────────

def get_source(offline: bool) -> MarketDataSource:
    """Pick a source. `offline` is explicit rather than 'try live, fall back' —
    see MarketDataError: an automatic fallback would let a fixture masquerade as
    the market on the day the API happens to be down."""
    return FixtureSource() if offline else MassiveSource()


def fetch_prices(ticker: str, offline: bool, lookback_days: int = 180) -> tuple[list[Bar], MarketDataSource]:
    """Traced entry point for stage 1 of the pipeline."""
    src = get_source(offline)
    with span("fetch_prices", ticker=ticker.upper(), source=src.name,
              lookback_days=lookback_days) as s:
        bars = src.bars(ticker, lookback_days)
        s.update(n_bars=len(bars), first=bars[0].date, last=bars[-1].date,
                 last_close=bars[-1].close, provenance=src.describe())
    return bars, src


if __name__ == "__main__":   # probe the adapter without running the whole pipeline
    import sys
    from kit import say
    tk = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()
    live = MassiveSource()
    say(f"[bold]MASSIVE_API_KEY configured:[/bold] {live.available()}  base={live.base}")
    for src in (FixtureSource(), live):
        try:
            b = src.bars(tk)
            say(f"[green]{src.name}[/green]: {len(b)} bars, {b[0].date} -> {b[-1].date}, "
                f"last close {b[-1].close}")
        except MarketDataError as e:
            say(f"[yellow]{src.name}[/yellow]: {e}")
