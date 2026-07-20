#!/usr/bin/env python3
"""Extract Socket.IO event names + payload hints from the console's client JS.

Feed this the client JS bundle(s) The collaborator sends (or a JS file saved from
the live site). It greps for the Socket.IO surface -- `socket.emit(...)`,
`socket.on(...)` / `.once(...)`, `io(...)` connection setup -- and prints the
event names it finds, grouped emit vs listen, with a little surrounding context
so you can read off the payload shape. Turn that into config.py by hand (it does
NOT rewrite config.py -- you want a human eye on the payload mapping).

    python extract_from_js.py app.bundle.js [more.js ...]

No dependencies; pure text scan. Minified bundles are fine -- event *names* are
string literals and survive minification even when variable names don't.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

# `<recv>.emit("event"` / `.on("event"` / `.once("event"` -- capture the receiver
# token (to spot the socket variable) and the event literal.
EMIT_RE = re.compile(r"(\w+)\s*\.\s*emit\s*\(\s*['\"]([^'\"]+)['\"]")
ON_RE = re.compile(r"(\w+)\s*\.\s*(?:on|once)\s*\(\s*['\"]([^'\"]+)['\"]")
# io("/ns", {..}) or io(url, {path: "..."}) connection setup.
IO_RE = re.compile(r"\bio\s*\(([^)]*)\)")
PATH_RE = re.compile(r"path\s*:\s*['\"]([^'\"]+)['\"]")
AUTH_RE = re.compile(r"auth\s*:\s*(\{[^}]*\}|\w+)")


def scan(paths: List[Path]) -> None:
    emits: Dict[str, Set[str]] = defaultdict(set)   # event -> {receiver tokens}
    listens: Dict[str, Set[str]] = defaultdict(set)
    io_calls: List[str] = []
    paths_found: Set[str] = set()
    auths: Set[str] = set()

    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        for recv, ev in EMIT_RE.findall(text):
            emits[ev].add(recv)
        for recv, ev in ON_RE.findall(text):
            listens[ev].add(recv)
        for args in IO_RE.findall(text):
            io_calls.append(args.strip())
        paths_found.update(PATH_RE.findall(text))
        auths.update(a.strip() for a in AUTH_RE.findall(text))

    def dump(title: str, table: Dict[str, Set[str]]) -> None:
        print(f"\n== {title} ({len(table)}) ==")
        for ev in sorted(table):
            print(f"  {ev:30s}  via {{{', '.join(sorted(table[ev]))}}}")

    dump("client -> server  (socket.emit)", emits)
    dump("server -> client  (socket.on/once)", listens)

    print("\n== io(...) connection setup ==")
    for a in io_calls or ["(none found)"]:
        print(f"  io({a})")
    if paths_found:
        print(f"  socketio path candidates: {sorted(paths_found)}")
    if auths:
        print(f"  auth: {sorted(auths)}")

    print("\nNext: map the emit names onto config.EMIT (the five actions), the "
          "on names onto config.LISTEN (uart/status/trace), fill payload shapes "
          "from the call sites, set PROTOCOL_SOURCE='verified', then run "
          "test_dryrun.py after updating the mock to match.")


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    files = [Path(a) for a in argv]
    missing = [f for f in files if not f.is_file()]
    if missing:
        print(f"not found: {missing}", file=sys.stderr)
        return 2
    scan(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
