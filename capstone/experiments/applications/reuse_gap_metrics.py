"""Validate an opt-in inner-allocator release-to-reissue histogram.

The histogram is indexed by successful new-lifetime issues. It is a
conditional distribution over starts that were actually reused, and is not
the fixed-follow-up retirement metric in memory-metrics.md.
"""
import re


def parse_reuse_gap(stderr, prefix):
    if prefix not in ('PYM_REUSE_GAP', 'PG_REUSE_GAP', 'PERL_REUSE_GAP'):
        raise ValueError('unknown reuse-gap observer')
    lines = [re.sub(r'^(backend> )+', '', line) for line in stderr.splitlines()]
    lines = [line for line in lines if line.startswith(prefix + ' ')]
    if len(lines) != 1:
        raise ValueError('expected exactly one reuse-gap report')
    words = [word.split('=', 1) for word in lines[0].split()[1:]]
    if any(len(word) != 2 for word in words):
        raise ValueError('malformed reuse-gap report')
    fields = dict(words)
    keys = {'attempts', 'issues', 'releases', 'reuses', 'distinct',
            'capacity', 'error', 'bins'}
    if len(words) != len(fields) or set(fields) != keys:
        raise ValueError('incomplete reuse-gap report')
    try:
        bins = [int(value) for value in fields.pop('bins').split(',')]
        report = {key: int(value) for key, value in fields.items()}
    except ValueError as error:
        raise ValueError('noninteger reuse-gap report') from error
    if (len(bins) != 32 or any(value < 0 for value in bins) or
            report['error'] != 0 or report['capacity'] < 2 or
            report['capacity'] & (report['capacity'] - 1) or
            not 0 <= report['reuses'] <= report['releases'] <= report['issues'] <= report['attempts'] or
            report['distinct'] + report['reuses'] != report['issues'] or
            report['distinct'] > report['capacity'] or
            sum(bins) != report['reuses']):
        raise ValueError('inconsistent reuse-gap report')
    report['bins'] = bins
    return report
