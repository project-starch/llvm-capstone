#!/usr/bin/env python3
"""Collect run.py runs into result lines fit to commit.

    collect.py SRC_DIR DEST_DIR RUN[:LOG] [RUN[:LOG] ...]

For each RUN, SRC_DIR/RUN/result.json is copied to DEST_DIR/RUN.json with every string scrubbed: home
paths become ~/ and user@host strings <user>@<host>; lines are kept, only those parts are replaced.
With :LOG, the last 25 lines of the runner's own output (SRC_DIR/LOG, e.g. a timeout's traceback) are
added as "runner_log_tail". The serial log is not copied. A record that still holds a home path or a
user@host string after the scrub is refused.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform'))
import records  # noqa: E402


def main():
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[3:]:
        run, _, log = spec.partition(':')
        r = json.loads((src / run / 'result.json').read_text())
        if log:
            r['runner_log_tail'] = (src / log).read_text(errors='replace').splitlines()[-25:]
        r = records.scrub(r)
        out = json.dumps(r, indent=1) + '\n'
        if records.leaks(out):
            sys.exit(f'{run}: a home path or user@host survived the scrub')
        (dest / f'{run}.json').write_text(out)
        print(f'{run}: exit={r.get("exit")} tests={len(r.get("tests") or [])}')


if __name__ == '__main__':
    main()
