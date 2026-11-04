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


def disjoint(run):
    """Every page of an arena is resolved ONCE: by a demand fault (an event) or by the window of an
    earlier fault, never both and never by two windows. A prefaulted page that then takes its own
    demand fault (or is prefaulted again) is wasted work and a wrong window: the sets of event
    pages and window pages must not meet, whichever came first.

    Not applicable (True) to a run that ends in the denial it tests: a window prefaults a read-only
    page for reading, and the denied store that follows is a legitimate event on a window page.
    """
    if run.get('exit') != 0:
        return True
    events, windows = {}, {}
    for e in run.get('raw_events', []):
        events.setdefault(e['arena'], set()).add(e['address'] // 4096)
    for base, address, count, _ in run.get('raw_windows', []):
        pages = windows.setdefault(base, [])
        pages.extend(address // 4096 + i for i in range(count))
    return all(len(pages) == len(set(pages)) and not set(pages) & events.get(base, set())
               for base, pages in windows.items())


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
        disjoint(run) and
        sum(v['resolves'] for v in retired.values()) > 0 and
        sum(v['ahead'] for v in retired.values()) == sum(w[2] for w in run['raw_windows']))


# Pages of peak residency the 'vm' launch may add to the 'normal' launch's. The launch maps 64 MiB it
# never touches (16384 pages); the bound is set from the measured difference of the two launches.
UNUSED_PAGES_BOUND = 256


def residency(lines, start, expected_stats):
    """The runtime gate's ON page predicates, from LINUX's accounting instead of the module's.

    Unpinned mode keeps no page counters (pages == peak == 0 by construction), so the pinned
    predicates linux_pages_released, shared_heap_pages_released and unused_pages_not_populated have
    nothing to read. The launcher prints, with each CAPSTONE_VM_STATS record, a CAPSTONE_VM_RSS
    record: its VmRSS now and VmHWM at its peak, in pages (CAPSTONE_VM_RSS=1). `lines` is the
    console, `start` the index of the first record of the recycling/sqlite group (run.py), and
    `expected_stats` the number of records a complete run prints.
    """
    def record(prefix):
        return [dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', s))
                for s in lines if s.startswith(prefix)]
    stats, rss = record('CAPSTONE_VM_STATS '), record('CAPSTONE_VM_RSS ')
    base = rss[:2] + rss[start:start + 5]
    witness = len(rss) == len(stats) == expected_stats and all(
        r.get('hwm_pages', 0) >= r.get('rss_pages', 1) > 0 for r in rss)
    return {
        'linux_rss_witness': witness,
        # mapped pages given back during the run (munmap, free): the peak is above the end
        'linux_pages_released': witness and sum(r['hwm_pages'] > r['rss_pages'] + 300 for r in base) == 3,
        'shared_heap_pages_released': witness and rss[4]['hwm_pages'] - rss[4]['rss_pages'] >= 768,
        # the 'vm' launch maps 64 MiB it never touches: its peak residency stays near a launch that
        # maps nothing (a window that populated the unused region would add 16384 pages)
        'unused_pages_not_populated': witness and rss[2]['hwm_pages'] - rss[0]['hwm_pages'] < UNUSED_PAGES_BOUND,
    }
