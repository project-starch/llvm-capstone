"""Unpinned fault evidence, separate from the persistent-pin census."""
import re


def attach(log, runs, hooks=False):
    current = None
    for line in log.splitlines():
        marker = re.match(r'HOOK_PINS_BEFORE:(\w+):(\w+):' if hooks else
                          r'PIN_EXIT:(\d+):(\w+):', line)
        if marker:
            key = (marker[1] if hooks else int(marker[1]), marker[2])
            current = runs.get(key)
            if current is not None:
                current.update(raw_events=[], raw_windows=[], raw_retire={})
        if current is None:
            continue
        event = re.search(r'unpinned fault arena 0x([0-9a-f]+) address 0x([0-9a-f]+) '
                          r'cause (\d+) pc 0x([0-9a-f]+) route (\w+) number (\d+)', line)
        if event:
            current['raw_events'].append(dict(arena=int(event[1], 16), address=int(event[2], 16),
                cause=int(event[3]), pc=int(event[4], 16), route=event[5], number=int(event[6])))
        window = re.search(r'unpinned window arena 0x([0-9a-f]+) address 0x([0-9a-f]+) pages (\d+) limit (\d+)', line)
        if window:
            current['raw_windows'].append((int(window[1], 16), int(window[2], 16), int(window[3]), int(window[4])))
        retire = re.search(r'unpinned retire arena 0x([0-9a-f]+) events=(\d+) resolves=(\d+) touch=(\d+) ahead=(\d+)', line)
        if retire:
            current['raw_retire'][int(retire[1], 16)] = dict(zip(
                ('events', 'resolves', 'touch', 'ahead'), map(int, retire.group(2, 3, 4, 5))))
        ahead = re.search(r'unpinned ahead arena 0x([0-9a-f]+) page (\d+) want (\d+) resolved (\d+) fault 0x([0-9a-f]+)', line)
        if ahead and hooks:
            current['ahead'].append((int(ahead[1], 16), *map(int, ahead.group(2, 3, 4)), int(ahead[5], 16)))


def region_arena(run):
    region = run.get('region')
    return None if region is None else next((base for base, size in run['arenas']
                                            if base <= region < base + size), None)


def region_counts(run):
    """Distinct virtual pages resolved in the region and counted domain faults.

    These are resolution witnesses, never pin acquisitions. Retries of the
    same CV_RESOLVE do not manufacture pages or domain fault events.
    """
    arena = region_arena(run)
    if run.get('region') is None:
        return (0, 0)
    if arena is None or arena not in run.get('raw_retire', {}):
        return None
    start = run['region']
    end = start + run.get('span', next(size for base, size in run['arenas'] if base == arena))
    events = [e for e in run['raw_events'] if e['arena'] == arena and start <= e['address'] < end]
    pages = {e['address'] // 4096 for e in events}
    for base, address, count, _ in run['raw_windows']:
        if base == arena:
            pages.update(p // 4096 for p in range(address, address + 4096 * count, 4096)
                         if start <= p < end)
    return len(pages), len(events)


def windows_stay_below(run, bound):
    """Every window the region's arena opened from an address up to `bound` (the window of the last page before it included) also PLANNED
    (limit, before any Linux refusal) to end at or below it, and at least one such window exists.

    The bound is a VMA end. Beside a read-only VMA a missing clip is hidden by Linux itself (the
    write fault on the neighbour is refused, so the window resolves 0 pages either way); the
    planned limit is what the missing clip changes, and this reads it.
    """
    arena = region_arena(run)
    mine = [(a, n, lim) for base, a, n, lim in run.get('raw_windows', []) if base == arena and a <= bound]
    return bool(mine) and all(a + lim * 4096 <= bound for a, n, lim in mine)


def counter_counts(run):
    """Public fault total less explicitly attributed faults outside the region."""
    counts = region_counts(run)
    stats = run.get('stats')
    if counts is None or not stats:
        return None
    return counts[0], stats['faults'] - (len(run['raw_events']) - counts[1])


def accounted(run):
    stats = run.get('stats')
    events = run.get('raw_events', [])
    retired = run.get('raw_retire', {})
    return bool(stats and retired and stats['pages'] == stats['peak'] == 0 and
        stats['faults'] > 0 and [e['number'] for e in events] == list(range(1, stats['faults'] + 1)) and
        all(e['arena'] in retired and e['cause'] in (13, 15) and
            any(base == e['arena'] and base <= e['address'] < base + size
                for base, size in run['arenas']) for e in events) and
        sum(v['events'] for v in retired.values()) == len(events) and
        all(v['events'] == sum(e['arena'] == base for e in events) and
            v['ahead'] == sum(w[2] for w in run['raw_windows'] if w[0] == base)
            for base, v in retired.items()) and
        sum(v['resolves'] for v in retired.values()) > 0 and
        sum(v['ahead'] for v in retired.values()) == sum(w[2] for w in run['raw_windows']))
