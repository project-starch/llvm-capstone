#!/usr/bin/env python3
"""Collect speedtest1-guest.py runs into result lines fit to commit.

    collect-speedtest1.py SRC_DIR DEST_DIR RUN[:LOG] [RUN[:LOG] ...]

For each RUN, SRC_DIR/RUN/result.json is copied to DEST_DIR/RUN.json with every string scrubbed: home
paths become ~/ and user@host strings <user>@<host>; lines are kept, only those parts are replaced.
A record without console lines (written before the runner kept them) gets the matching lines of its
own serial log, SRC_DIR/RUN/vm/serial.log, under "console_from_serial" (the whole boot: these runs
did not record where the run began). With :LOG, the last 25 lines of the runner's own output
(SRC_DIR/LOG, e.g. a timeout's traceback) are added as "runner_log_tail". The serial log is not copied.
"""
import json
from pathlib import Path
import re
import sys

HOME = re.compile(r'/home/[^/\s"\']+/')
USERHOST = re.compile(r'\b[\w.-]+@(?!127\.0\.0\.1\b)[\w.-]+')
KEEP = re.compile(r'panic|trap|fatal|revok|poison|KDB|Stopped at|exception|signal|lock of', re.I)


def scrub(value):
    if isinstance(value, str):
        return USERHOST.sub('<user>@<host>', HOME.sub('~/', value))
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


def main():
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[3:]:
        run, _, log = spec.partition(':')
        r = json.loads((src / run / 'result.json').read_text())
        serial = src / run / 'vm/serial.log'
        if not r.get('console') and serial.exists():
            text = serial.read_bytes().decode(errors='replace').replace('\r', '')
            r['console_from_serial'] = [l.strip() for l in text.splitlines() if KEEP.search(l)][:200]
        if log:
            r['runner_log_tail'] = (src / log).read_text(errors='replace').splitlines()[-25:]
        r = scrub(r)
        out = json.dumps(r, indent=1) + '\n'
        if '/home/' in out or USERHOST.search(out.replace('<user>@<host>', '')):
            sys.exit(f'{run}: a home path or user@host survived the scrub')
        (dest / f'{run}.json').write_text(out)
        print(f'{run}: exit={r.get("exit")} tests={len(r.get("tests") or [])}')


if __name__ == '__main__':
    main()
