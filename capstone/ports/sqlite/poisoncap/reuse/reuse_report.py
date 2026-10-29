"""reuse-wrap.c's report lines and speedtest1's test lines, read in one place.

    sys.path.insert(0, str(<this folder>)); import reuse_report

A report line, as reuse-wrap.c prints it (the last of a label counts):
  REUSE-GAP memsys5 at=LABEL installed=.. heap=.. unit=.. attempts=.. issues=.. releases=.. reuses=..
    distinct=.. failed=.. moved=.. inplace=.. live=.. maxunit=.. error=.. first_error=.. bins=b0,..,b31
  SQL-CHECK step_row=.. step_done=.. step_other=.. first_step_other=.. reset_ok=.. reset_err=.. first_reset_err=..
"""
import re

COUNTERS = ('attempts', 'issues', 'releases', 'reuses', 'distinct', 'failed', 'moved', 'inplace', 'live')
# The identities every admissible report satisfies, by name.
IDENTITIES = (
    ('attempts == issues + failed', lambda n, bins: n['attempts'] == n['issues'] + n['failed']),
    ('issues == distinct + reuses', lambda n, bins: n['issues'] == n['distinct'] + n['reuses']),
    ('sum(bins) == reuses', lambda n, bins: sum(bins) == n['reuses']),
    ('live == issues - releases', lambda n, bins: n['live'] == n['issues'] - n['releases']),
)
# speedtest1's lines: one completed test ("100 - name ......  1.234s"), and the total.
TEST = re.compile(r'^\s*(\d{3}) - .*?\.{3,}\s*([0-9.]+)s\s*$', re.M)
TOTAL = re.compile(r'^\s*TOTAL\.+\s*([0-9.]+)s\s*$', re.M)


def report_line(text, label='exit'):
    """The last REUSE-GAP line of LABEL in TEXT, or None."""
    found = re.findall(rf'^REUSE-GAP memsys5 at={re.escape(label)} .*$', text, re.M)
    return found[-1] if found else None


def sql_line(text):
    """The last SQL-CHECK line in TEXT, or None."""
    found = re.findall(r'^SQL-CHECK .*$', text, re.M)
    return found[-1] if found else None


def fields(line):
    """A report or SQL-CHECK line's key=value pairs, as strings (bins as a list of ints)."""
    d = dict(kv.split('=', 1) for kv in line.split() if '=' in kv and not kv.startswith('at='))
    if 'bins' in d:
        d['bins'] = [int(x) for x in d['bins'].split(',')]
    return d


def snapshot(text, label='exit'):
    """The last report of LABEL in TEXT as fields(), or None."""
    line = report_line(text, label)
    return fields(line) if line else None


def counters(d):
    return {k: int(d[k]) for k in COUNTERS}


def failed_identities(d):
    """The names of the identities report D breaks."""
    n = counters(d)
    return [name for name, holds in IDENTITIES if not holds(n, d['bins'])]
