"""Status output in the terminal: numbered steps with their duration, and a progress bar
with the time left. It goes to stderr, so results and tables on stdout stay clean.

    steps = Steps(3)
    with steps("Loading the tracking"):                 # [1/3] Loading the tracking ...
        ...                                             #       done in 1.8 s
    with Progress(400, "matching") as bar:              #       matching [████░░░░] 160/400  40%  1:31 elapsed, ~2:16 left
        for item in items:
            bar.update()

Outside a terminal (redirected to a file, in tests) the bar writes one line per 10 % instead.
"""
from __future__ import annotations

import sys
import time

INDENT = "      "  # lines inside a step line up under its text
BAR_WIDTH = 30


def duration(seconds: float) -> str:
    """12.3 s below a minute, m:ss above."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d} min"


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def say(text: str) -> None:
    """A status line, indented to sit inside the current step."""
    print(f"{INDENT}{text}", file=sys.stderr, flush=True)


class Step:
    """Prints "[2/5] text ..." when entered and "done in 12.3 s" when left."""

    def __init__(self, text: str, number: int | None = None, total: int | None = None, enabled: bool = True):
        self.text, self.enabled = text, enabled
        self.prefix = f"[{number}/{total}] " if number is not None and total else ""

    def __enter__(self):
        self.start = time.monotonic()
        if self.enabled:
            print(f"{self.prefix}{self.text} ...", file=sys.stderr, flush=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.enabled and exc_type is None:
            say(f"done in {duration(time.monotonic() - self.start)}")
        return False


class Steps:
    """Numbers the steps of one run: steps = Steps(5); with steps("Loading"): ..."""

    def __init__(self, total: int, enabled: bool = True):
        self.total, self.enabled, self.number = total, enabled, 0

    def __call__(self, text: str) -> Step:
        self.number += 1
        return Step(text, self.number, self.total, self.enabled)


class Progress:
    """A progress bar on one terminal line, with elapsed time and an estimate of the time left."""

    def __init__(self, total: int, label: str, min_interval: float = 0.2):
        self.total = max(int(total), 1)
        self.label = label
        self.count = 0
        self.min_interval = min_interval
        self.stream = sys.stderr
        self.tty = self.stream.isatty()
        self.start = self._last = time.monotonic()
        self._decile = -1
        self._draw()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def update(self, n: int = 1) -> None:
        self.count = min(self.count + n, self.total)
        if self.tty:
            now = time.monotonic()
            if now - self._last >= self.min_interval or self.count == self.total:
                self._last = now
                self._draw()
        elif self.count * 10 // self.total > self._decile:
            self._draw()

    def close(self) -> None:
        self._draw()
        if self.tty:
            self.stream.write("\n")
            self.stream.flush()

    def _draw(self) -> None:
        done = self.count / self.total
        elapsed = time.monotonic() - self.start
        timing = f"{_clock(elapsed)} elapsed"
        if 0 < self.count < self.total:
            timing += f", ~{_clock(elapsed / done - elapsed)} left"
        if self.tty:
            filled = int(BAR_WIDTH * done)
            bar = "█" * filled + "░" * (BAR_WIDTH - filled)
            self.stream.write(f"\r{INDENT}{self.label} [{bar}] {self.count}/{self.total} {done:4.0%}  {timing}   ")
            self.stream.flush()
        elif self.count * 10 // self.total > self._decile:
            self._decile = self.count * 10 // self.total
            self.stream.write(f"{INDENT}{self.label}: {self.count}/{self.total} ({done:.0%}), {timing}\n")
            self.stream.flush()
