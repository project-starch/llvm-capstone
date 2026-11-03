#!/usr/bin/env python3
"""Validate the matched windows and print per-process layer/function deltas."""
import collections
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
means = {}
functions = {}
for arm in ('plain', 'launcher', 'native'):
    directory = root / ('measure-' + arm)
    windows = []
    for i in range(6):
        invocation = json.loads((directory / 'invocations' / f'{i:04d}.json').read_text())
        output = (directory / 'invocations' / f'{i:04d}.out').read_text()
        assert invocation['rc'] == 0 and invocation['detail']['window'] == i
        assert 'bad=0' in output and f'n={10 if i % 2 == 0 else 100} ' in output
        windows.append(json.loads((directory / 'profile' / f'window-{i}.json').read_text()))
    pairs = []
    layers, funcs = collections.Counter(), collections.Counter()
    for i in range(0, 6, 2):
        small, large = windows[i:i+2]
        pairs.append((large['total_attempts'] - small['total_attempts']) / 90)
        for sign, w in ((-1, small), (1, large)):
            for c, value in w['classes'].items(): layers[c] += sign * value / 270
            for c, f, value in w['all_functions']: funcs[c, f] += sign * value / 270
        print(f'{arm} pair={i//2+1} W10={small["total_attempts"]} W100={large["total_attempts"]} per_process={pairs[-1]:.2f}')
    means[arm], functions[arm] = layers, funcs
    print(f'{arm} mean={statistics.mean(pairs):.2f} range={min(pairs):.2f}..{max(pairs):.2f} busy={statistics.mean(pairs)-layers["kernel idle"]:.2f}')
    if arm == 'native':
        placements = {(directory / 'invocations' / f'{i:04d}.addresses').read_text().strip() for i in range(6)}
        assert len(placements) == 1
        print('native placement base,trampoline=' + ','.join(hex(int(x)) for x in placements.pop().split()))
    elif arm == 'launcher':
        placements = {(directory / 'invocations' / f'{i:04d}.pilot').read_text().strip() for i in range(6)}
        assert len(placements) == 1
        print('launcher placement=' + placements.pop())
print('layer plain launcher native native-minus-launcher')
for layer in sorted(set().union(*(x.keys() for x in means.values()))):
    p, a, b = (means[x][layer] for x in ('plain', 'launcher', 'native'))
    print(f'{layer}: {p:.2f} {a:.2f} {b:.2f} {b-a:+.2f}')
deltas = {key: functions['native'][key] - functions['launcher'][key]
          for key in functions['native'].keys() | functions['launcher'].keys()}
print('largest function deltas native-minus-launcher per process')
for (layer, function), delta in sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:20]:
    print(f'{layer} {function} {delta:+.2f}')
old, new = (sum(means[a].values()) for a in ('launcher', 'native'))
print(f'native reduction={(1-new/old)*100:.3f}% speedup={old/new:.3f}x')
