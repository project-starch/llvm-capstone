#!/usr/bin/env python3
"""Mutation controls against real, passing ON gate transcripts."""
import argparse
import importlib.util
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PINS = r'nr_foll_pin_acquired (\d+) nr_foll_pin_released (\d+)'


def pin_taken(text, tag):
    """One more pin acquired AND released at the first rmmod reading: balanced, but not zero."""
    return re.sub(r'^(' + tag + r'_PINS_END:[^\n]*?)' + PINS,
                  lambda m: f'{m[1]}nr_foll_pin_acquired {int(m[2]) + 1} nr_foll_pin_released {int(m[3]) + 1}',
                  text, count=1, flags=re.M)


def dead_counters(text, tag):
    """The counters never move, including in the pinned liveness launch: zero everywhere."""
    return re.sub(r'^(' + tag + r'_PINS_(?:LIVE|END)[^\n]*?)' + PINS,
                  lambda m: f'{m[1]}nr_foll_pin_acquired 0 nr_foll_pin_released 0', text, flags=re.M)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin', type=Path, required=True)
    p.add_argument('--hooks', type=Path, required=True)
    p.add_argument('--runtime', type=Path, help='the runtime gate log (runtime.log) of an ON arm')
    a = p.parse_args()
    pin, hooks = load('run-pin-ahead'), load('run-step-hooks')
    names = [n for n in hooks.CONFIGS if hooks.app(n) == 'pin']
    readers = [('pin', a.pin, lambda s: pin.evaluate(s, True, True, True)[0]),
               ('hooks', a.hooks, lambda s: hooks.evaluate(s, names, True, True, True)[0])]
    failed = []
    if a.runtime:
        checks = load('unpinned_checks')
        text = a.runtime.read_text(errors='replace').replace('\r', '')

        def residency(s):
            lines = [x.strip() for x in s.splitlines()]
            return checks.residency(lines, 6, 12)

        def edit_rss(s, index, **fields):
            """Rewrite fields of the index-th CAPSTONE_VM_RSS record."""
            seen = [-1]

            def one(m):
                seen[0] += 1
                if index is not None and seen[0] != index:
                    return m[0]
                line = m[0]
                for k, v in fields.items():
                    line = re.sub(k + r'=\d+', lambda _: f'{k}={v(line) if callable(v) else v}', line)
                return line
            return re.sub(r'^CAPSTONE_VM_RSS .*$', one, s, flags=re.M)

        def field(line, k):
            return int(re.search(k + r'=(\d+)', line)[1])
        positive = residency(text)
        if not all(positive.values()):
            failed.append('runtime: positive transcript fails ' + ','.join(k for k, v in positive.items() if not v))
        else:
            # (mutated transcript, the check that must be the one to flip)
            mutations = {
                'nothing_released': (edit_rss(text, None, hwm_pages=lambda l: field(l, 'rss_pages')), 'linux_pages_released'),
                'unused_region_populated': (edit_rss(text, 2, hwm_pages=lambda l: field(l, 'hwm_pages') + 16384),
                                            'unused_pages_not_populated'),
                'shared_heap_kept': (edit_rss(text, 4, hwm_pages=lambda l: field(l, 'rss_pages')), 'shared_heap_pages_released'),
                'no_rss_records': (re.sub(r'^CAPSTONE_VM_RSS .*\n', '', text, flags=re.M), 'linux_rss_witness'),
            }
            for label, (changed, key) in mutations.items():
                rejected = changed != text and residency(changed)[key] is False
                print(('PASS ' if rejected else 'FAIL ') + 'runtime_' + label + ' flips ' + key)
                if not rejected:
                    failed.append('runtime_' + label)
    for name, path, evaluate in readers:
        text = path.read_text(errors='replace').replace('\r', '')
        positive = evaluate(text)
        if not all(positive.values()):
            failed.append(name + ': positive transcript fails ' + ','.join(k for k, v in positive.items() if not v))
            continue
        tag = 'PIN' if name == 'pin' else 'HOOK'
        mutations = {
            'balanced_pin_taken': pin_taken(text, tag),
            'dead_foll_pin_counters': dead_counters(text, tag),
            'refaulted_window_page': re.sub(r'^(.*unpinned window arena 0x[0-9a-f]+ address 0x[0-9a-f]+ pages [1-9]\d* limit \d+.*\n)',
                                            r'\1\1', text, count=1, flags=re.M),
            'wrong_mode': re.sub(r'(MEMORY_MODE:)Y', r'\1N', text),
            'zero_fault_totals': re.sub(r'(CAPSTONE_VM_STATS [^\n]*?faults=)\d+', r'\g<1>0', text),
            'missing_events': re.sub(r'^.*unpinned fault arena.*\n', '', text, flags=re.M),
            'missing_windows': re.sub(r'^.*unpinned window arena.*\n', '', text, flags=re.M),
        }
        if name == 'hooks':
            mutations.update(
                no_injection=re.sub(r'\binjections=\d+', 'injections=0', text),
                bad_entry=re.sub(r'\bviolations=0', 'violations=1', text),
                pending_fault=re.sub(r'\bfault_pending=0', 'fault_pending=1', text),
                oversized_batch=re.sub(r'\bfault_max=\d+', 'fault_max=17', text),
                no_deferral=re.sub(r'\bfault_deferred=\d+', 'fault_deferred=0', text),
                no_partial=re.sub(r'\bfault_short=\d+', 'fault_short=0', text))
        for label, changed in mutations.items():
            rejected = changed != text and not all(evaluate(changed).values())
            print(('PASS ' if rejected else 'FAIL ') + name + '_' + label)
            if not rejected:
                failed.append(name + '_' + label)
    for failure in failed:
        print('FAIL ' + failure)
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
