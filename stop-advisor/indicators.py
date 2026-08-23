"""The numeric layer: volatility and structure, in arithmetic you can check by hand.

Every function here is deliberately a loop over a list of floats. No pandas, no
numpy, no TA library. Three reasons, in order of importance:

  1. This module's output is the part a person might risk money against. If the
     ATR is wrong the whole briefing is wrong, and "wrong" here is silent — a
     plausible number, confidently formatted. So the code has to be small enough
     that a reader can verify it against a textbook definition in one sitting.
  2. TA libraries disagree with each other. `ta.ATR` defaults to Wilder, `pandas_ta`
     has shipped both, and some tutorials use a plain SMA. Those give materially
     different stops on the same bars. Hiding that choice inside a dependency is
     how you end up unable to explain your own number.
  3. The golden set asserts exact values against hand-computed arithmetic
     (goldset/golden.jsonl, the `numeric` cases). That only means something if the
     implementation is inspectable.

NOTHING HERE PREDICTS ANYTHING. ATR is a backward-looking average of realised
daily range. Realised volatility is the standard deviation of returns that already
happened. A "stop candidate" is that measured noise level converted into a distance
from entry — it says "moves this size are ordinary for this instrument lately",
never "price will not go below here".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

TRADING_DAYS = 252     # the conventional annualisation factor for daily bars


@dataclass(frozen=True)
class Bar:
    """One OHLCV bar. `date` is an ISO date string — kept as text so fixtures round-trip
    through JSON without a timezone library deciding what 'day' means."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @staticmethod
    def from_dict(d: dict) -> "Bar":
        return Bar(date=str(d["date"]), open=float(d["open"]), high=float(d["high"]),
                   low=float(d["low"]), close=float(d["close"]),
                   volume=float(d.get("volume", 0) or 0))


# ── true range and ATR ──────────────────────────────────────────────────────

def true_ranges(bars: list[Bar]) -> list[float]:
    """Wilder's True Range for every bar that HAS a previous close.

        TR = max(high - low, |high - prev_close|, |low - prev_close|)

    The two gap terms are the whole point of TR over a plain high-low range: an
    instrument that gaps down 5% overnight and then trades a quiet 1% intraday
    range moved 5%, and a stop sized on the 1% would be nonsense.

    The first bar is DROPPED rather than given TR = high - low. Some libraries seed
    it that way; it silently mixes a different quantity into the average, and on a
    14-bar window that is 7% of the answer. Dropping it costs one bar of history
    and keeps every element of the list the same measurement.
    """
    out: list[float] = []
    for prev, cur in zip(bars, bars[1:]):
        out.append(max(cur.high - cur.low,
                       abs(cur.high - prev.close),
                       abs(cur.low - prev.close)))
    return out


def atr(bars: list[Bar], period: int = 14, method: str = "wilder") -> float:
    """Average True Range.

    Two methods, both standard, and they do NOT agree — which is exactly why the
    method is an argument and gets printed in the output rather than assumed:

      wilder  the original (Wilder 1978). Seed = simple mean of the first `period`
              TRs, then ATR = (prev*(period-1) + TR) / period. An exponential-ish
              smoothing, so ALL history leaks into the value with decaying weight.
      sma     plain mean of the last `period` TRs. Reacts faster and forgets
              completely; easier to explain to someone reading the briefing.

    Raises on insufficient data instead of returning a shorter-window average.
    A 14-period ATR computed over 6 bars is not a 14-period ATR, and a risk tool
    that quietly substitutes one for the other is lying about its own precision.
    """
    trs = true_ranges(bars)
    if period < 1:
        raise ValueError("period must be >= 1")
    if len(trs) < period:
        raise ValueError(
            f"need at least {period + 1} bars for a {period}-period ATR "
            f"(got {len(bars)} bars -> {len(trs)} true ranges)")

    if method == "sma":
        window = trs[-period:]
        return sum(window) / period

    if method == "wilder":
        value = sum(trs[:period]) / period          # seed
        for tr in trs[period:]:
            value = (value * (period - 1) + tr) / period
        return value

    raise ValueError(f"unknown ATR method {method!r} (use 'wilder' or 'sma')")


# ── realised volatility ─────────────────────────────────────────────────────

def log_returns(bars: list[Bar]) -> list[float]:
    """ln(C_t / C_{t-1}). Log rather than simple returns because they are additive
    over time, which is what makes the sqrt-of-time annualisation below legitimate."""
    out = []
    for prev, cur in zip(bars, bars[1:]):
        if prev.close <= 0 or cur.close <= 0:
            raise ValueError(f"non-positive close at {cur.date} — cannot take a log return")
        out.append(math.log(cur.close / prev.close))
    return out


def realised_vol(bars: list[Bar], window: int | None = None,
                 periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised realised volatility = stdev(log returns) * sqrt(252).

    SAMPLE standard deviation (n-1). The bars are a sample of the instrument's
    behaviour, not the population of all its days; with a 20-day window the
    difference between n and n-1 is ~2.6% of the answer, which is not nothing when
    the number is being turned into a stop distance.

    REALISED, not implied. This is the volatility that already occurred. It is a
    measurement of the recent past and carries no claim about the future — the
    reason the output calls it 'realised' every single time it appears.
    """
    rets = log_returns(bars)
    if window is not None:
        rets = rets[-window:]
    if len(rets) < 2:
        raise ValueError(f"need at least 3 bars for a sample stdev (got {len(bars)})")
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(periods_per_year)


# ── structure: swing lows ───────────────────────────────────────────────────

@dataclass(frozen=True)
class Swing:
    index: int
    date: str
    low: float
    bars_ago: int


def swing_lows(bars: list[Bar], width: int = 2) -> list[Swing]:
    """Fractal swing lows: a bar whose low is <= the lows of `width` bars either side.

    `<=` not `<` on purpose. Flat bases are common and a strict inequality throws
    away the most important support level on the chart — the one price has tested
    repeatedly at the same price.

    The last `width` bars can never qualify, because the confirming bars to their
    right have not happened yet. That is not a bug to patch around: a swing low is
    only a swing low in hindsight, and pretending otherwise is exactly the kind of
    lookahead that makes backtests lie.
    """
    out: list[Swing] = []
    n = len(bars)
    for i in range(width, n - width):
        lo = bars[i].low
        left = all(lo <= bars[j].low for j in range(i - width, i))
        right = all(lo <= bars[j].low for j in range(i + 1, i + width + 1))
        if left and right:
            out.append(Swing(index=i, date=bars[i].date, low=lo, bars_ago=n - 1 - i))
    return out


def nearest_swing_low_below(swings: list[Swing], price: float) -> Swing | None:
    """The highest swing low that still sits below `price` — the first structural
    level a decline would have to break. Ties go to the more RECENT bar: a level
    tested last week is better evidence than the same level from six months ago."""
    below = [s for s in swings if s.low < price]
    if not below:
        return None
    return max(below, key=lambda s: (s.low, s.index))


# ── stop candidates ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StopCandidate:
    label: str
    level: float          # the price
    basis: str            # the arithmetic, spelled out, for the briefing and the citation
    distance: float       # entry - level, in currency
    risk_pct: float       # distance / entry, as a percentage of position value
    kind: str             # "volatility" | "structure"

    def as_dict(self) -> dict:
        return asdict(self)


def stop_candidates(entry: float, atr_value: float, swings: list[Swing],
                    multiples: tuple[float, ...] = (1.5, 2.0, 3.0),
                    atr_period: int = 14, atr_method: str = "wilder") -> list[StopCandidate]:
    """Turn measured volatility and observed structure into candidate stop LEVELS.

    Two families, because they fail differently and a reader should see both:

      volatility stops   entry - N x ATR. Says "a move of N average daily ranges
                         against me is larger than this instrument's ordinary
                         noise". Wider N = fewer stop-outs on noise, larger loss
                         when hit. That trade-off is the user's to make, not ours,
                         which is why three multiples are shown and none is starred.

      structure stop     just under the nearest swing low below entry. Says "this
                         is where the observed floor was". The 0.25 x ATR buffer
                         below it exists because a level everyone can see on a
                         chart gets probed; a stop exactly ON it is the one most
                         likely to be filled by a wick that then reverses.

    `basis` carries the substituted arithmetic as a string. That is what the LLM
    cites and what appears in the briefing — the number and its derivation travel
    together, so a claim can never be repeated without its math.
    """
    if entry <= 0:
        raise ValueError("entry price must be positive")
    if atr_value <= 0:
        raise ValueError("ATR must be positive — a zero-range series has no volatility to size against")

    out: list[StopCandidate] = []
    for m in multiples:
        level = entry - m * atr_value
        out.append(StopCandidate(
            label=f"{m:g}x ATR",
            level=level,
            basis=(f"entry {entry:.2f} - {m:g} x ATR({atr_period},{atr_method}) "
                   f"{atr_value:.4f} = {level:.2f}"),
            distance=entry - level,
            risk_pct=(entry - level) / entry * 100,
            kind="volatility",
        ))

    sw = nearest_swing_low_below(swings, entry)
    if sw is not None:
        buffer = 0.25 * atr_value
        level = sw.low - buffer
        out.append(StopCandidate(
            label="below swing low",
            level=level,
            basis=(f"swing low {sw.low:.2f} ({sw.date}, {sw.bars_ago} bars ago) "
                   f"- 0.25 x ATR {buffer:.4f} = {level:.2f}"),
            distance=entry - level,
            risk_pct=(entry - level) / entry * 100,
            kind="structure",
        ))
    return out


# ── one call that produces everything the briefing needs ────────────────────

def analyse(bars: list[Bar], entry: float, atr_period: int = 14,
            atr_method: str = "wilder", vol_window: int = 20,
            swing_width: int = 2) -> dict:
    """Every number the synthesis step is allowed to cite, computed in one place.

    Returned as a plain dict because it goes three places that must not disagree:
    the console table, the trace span, and the LLM prompt. Computing it once and
    passing the same object around is the cheapest guarantee that the number the
    model cited is the number the user was shown.
    """
    a = atr(bars, atr_period, atr_method)
    swings = swing_lows(bars, swing_width)
    return {
        "n_bars": len(bars),
        "first_date": bars[0].date,
        "last_date": bars[-1].date,
        "last_close": bars[-1].close,
        "entry": entry,
        "atr_period": atr_period,
        "atr_method": atr_method,
        "atr": a,
        "atr_pct_of_entry": a / entry * 100,
        "vol_window": vol_window,
        "realised_vol_annual_pct": realised_vol(bars, vol_window) * 100,
        "realised_vol_daily_pct": realised_vol(bars, vol_window) / math.sqrt(TRADING_DAYS) * 100,
        "swing_lows": [asdict(s) for s in swings],
        "candidates": [c.as_dict() for c in
                       stop_candidates(entry, a, swings, atr_period=atr_period,
                                       atr_method=atr_method)],
    }
