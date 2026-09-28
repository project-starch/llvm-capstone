"""Validation shared by the Capstone and default CheriBSD application runners."""


def allocation_samples(stderr):
    return [dict((k, v if k == 'phase' else int(v))
                 for k, v in (word.split('=', 1) for word in line.split()[1:]))
            for line in stderr.splitlines() if line.startswith('EXP-ALLOC ')]


def valid_allocations(stderr, phases):
    try:
        samples = allocation_samples(stderr)
        if [s['phase'] for s in samples] != phases: return False
        for s in samples:
            if any(v < 0 for k, v in s.items() if k != 'phase'): return False
            if s['errors'] or s['unknown'] or s['live'] > s['peak']: return False
            if s['blocks'] > s['peak_blocks'] or s['unique'] > s['allocations']: return False
            if s['allocations'] - s['frees'] != s['blocks']: return False
            if s['reused'] > s['allocations'] or s['observer'] <= 0: return False
            if sum(s[k] for k in ('reuse1', 'reuse8', 'reuse64', 'reuse512',
                                  'reuse4096', 'reuse_more')) != s['reused']: return False
        for a, b in zip(samples, samples[1:]):
            for key in ('peak', 'peak_blocks', 'requested', 'allocations', 'frees',
                        'failures', 'unique', 'reused'):
                if a[key] > b[key]: return False
        return bool(samples)
    except (ValueError, KeyError):
        return False
