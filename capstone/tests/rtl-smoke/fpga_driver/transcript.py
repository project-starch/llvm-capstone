#!/usr/bin/env python3
"""Read a board transcript the way the record requires (ISSUES M-11).

The console delivers the UART in chunks (median payload 7 characters) and the runner logs each one to
stderr as `[fpga] [uart] <repr(text)>`, interleaved with `[fpga] [event] ...`, `[fpga] emit ...` and
`[stages] ...` lines; the monitor's share-trace markers (`ECSA:00000004` and friends, one per line) land
in the middle of a domain's output, even mid-token. A parser that reads that log line by line therefore
reads a present result as absent (a summary printed `sublet: 0x []` for arm C, "no result line" for a
present mark) or, worse, reads a truncated number as the value (`SPEEDTEST1-CYCLES 8562` for
1,166,594,074), and in five archived boots the control rung's own `RESULT k800 retval=4` was readable
only after joining. This module is the one place that does the joining; every driver summary and the
watchdog read through it, so a summary-only reader is safe by construction.

Two texts, and the distinction matters:
  joined  = uart_text(scope_to_run(framed))   -- the UART with the markers KEPT: marker rows (ALEN:, RC..:)
                                                 and the stall marker (last_marker) read this one;
  clean   = strip_markers(joined)             -- the domain's lines: every result pattern reads this one.
Stripping first would blind the watchdog (it deletes SHA5/SHA6 too).

"No data" is an error, not a zero: `require` raises on an empty input, and an unparseable frame raises
unless it is the last line of a log still being written.

Command line (for the watchdog and for reading a transcript by hand):
    python3 transcript.py uart [driver.log]          # the joined, marker-stripped UART after this run's load_image
    python3 transcript.py last-marker [driver.log]   # the last SHA5/SHA6 marker on the joined text, '' if none
stdin is read when no path is given; exit 2 on a TranscriptError.
"""
import ast
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

__all__ = ["TranscriptError", "read", "scope_to_run", "uart_chunks", "uart_text", "strip_markers",
           "last_marker", "Arm", "arm_segments", "find_all", "require"]


class TranscriptError(Exception):
    """The transcript cannot be read: no UART after the run's load_image, or a frame that does not parse."""


UART_LINE = re.compile(r"^\[fpga\] \[uart\] (.*)$", re.M)                 # payload: repr(str), either quote style, or +NB
STAGE_START = re.compile(r"^\[stages\] --> TEST (\d+)/(\d+)\s+(.*)$", re.M)
STAGE_END = re.compile(r"^\[stages\] <-- TEST (\d+)/(\d+)\s+(.*)$", re.M)
MARKER = re.compile(r"[A-Z0-9]{4}:[0-9A-F]{8}\n?")                        # UNANCHORED on purpose: markers land mid-token
SQ_LINE = re.compile(r"^SQ: [^\n]*\n", re.M)
SHA_MARK = re.compile(r"SHA[56]:[0-9A-F]{8}")                              # no terminator: in a real stall the newline may never arrive
LOAD_IMAGE = "monitor load_image"


def read(path: str) -> str:
    """The file as text; the frames already carry U+FFFD from the console's own decode, so replace, never latin1."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def scope_to_run(framed: str) -> str:
    """Everything from this run's own `monitor load_image` on. Takes the FRAMED log: the anchor is a gdb emit
    line and never occurs inside UART text, and the console replays the previous boot before it."""
    i = framed.rfind(LOAD_IMAGE)
    return framed[i:] if i >= 0 else framed


def uart_chunks(framed: str, *, strict: bool = False) -> List[str]:
    """Every UART frame's text, in order. `[uart] +0B` is an empty chunk. A payload that does not parse is
    an error naming the line -- unless it is the LAST frame and strict is off (a live log mid-write)."""
    out: List[str] = []
    matches = list(UART_LINE.finditer(framed))
    for k, m in enumerate(matches):
        payload = m.group(1)
        if re.fullmatch(r"\+\d+B", payload):
            out.append("")
            continue
        try:
            value = ast.literal_eval(payload)
            if not isinstance(value, str):
                raise ValueError("frame payload is not a string literal")
        except (ValueError, SyntaxError) as e:
            if k == len(matches) - 1 and not strict:
                break
            line = framed.count("\n", 0, m.start()) + 1
            raise TranscriptError(f"line {line}: unparseable UART frame: {payload[:80]!r}") from e
        out.append(value)
    return out


def uart_text(framed: str) -> str:
    """The frames joined into one stream, newlines normalised. Markers are KEPT (see the module docstring)."""
    return "".join(uart_chunks(framed)).replace("\r\n", "\n").replace("\r", "")


def strip_markers(text: str, *, sq: bool = False) -> str:
    """Delete the monitor's share-trace markers with their newline (they splice domain lines mid-token), and
    the `SQ: ` lines when asked (the R1 bundle's convention)."""
    t = MARKER.sub("", text)
    return SQ_LINE.sub("", t) if sq else t


def last_marker(joined: str) -> Optional[str]:
    """The last SHA5/SHA6 share marker on the JOINED (unstripped) text: the watchdog's entry-stall reading."""
    found = SHA_MARK.findall(joined)
    return found[-1] if found else None


@dataclass
class Arm:
    index: int
    total: int
    label: str
    framed: str            # this arm's slice of the framed log (runner lines live here: TRAP LOG, <-- TEST, [wedge])
    uart: str              # strip_markers(uart_text(slice)): the domain's own lines
    end_line: Optional[str]  # the `[stages] <-- TEST` line, or None when the arm never ended

    @property
    def returned(self) -> Optional[bool]:
        """True: the runner saw the arm return; False: it recorded NO RETURN; None: no end line at all."""
        if self.end_line is None:
            return None
        return "NO RETURN" not in self.end_line


def arm_segments(framed: str) -> List[Arm]:
    """One Arm per `[stages] --> TEST k/n` line, its slice running to the next such line; the arm's UART is
    joined from the frames inside that slice, so a mark split across chunks reads whole."""
    starts = list(STAGE_START.finditer(framed))
    arms: List[Arm] = []
    for k, m in enumerate(starts):
        end = starts[k + 1].start() if k + 1 < len(starts) else len(framed)
        seg = framed[m.start():end]
        e = STAGE_END.search(seg)
        arms.append(Arm(int(m.group(1)), int(m.group(2)), m.group(3).strip(), seg,
                        strip_markers(uart_text(seg)), e.group(0) if e else None))
    return arms


def find_all(pattern: str, text: str, flags: int = 0) -> List[str]:
    """Every FULL match (a capturing group in the pattern no longer turns the row into its group text)."""
    return [m.group(0) for m in re.finditer(pattern, text, flags)]


def require(value, what: str):
    """No data is an error, not a zero."""
    if not value:
        raise TranscriptError(f"no {what}")
    return value


def _main(argv: List[str]) -> int:
    usage = "usage: transcript.py {uart|last-marker} [driver.log]   (stdin when no path)"
    if len(argv) < 2 or argv[1] not in ("uart", "last-marker"):
        print(usage, file=sys.stderr)
        return 2
    text = read(argv[2]) if len(argv) > 2 else sys.stdin.read()
    try:
        joined = uart_text(scope_to_run(text))
        if argv[1] == "uart":
            sys.stdout.write(strip_markers(joined))
        else:
            print(last_marker(joined) or "")
    except TranscriptError as e:
        print(f"transcript: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
