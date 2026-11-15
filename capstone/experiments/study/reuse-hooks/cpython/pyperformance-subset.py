"""A fixed subset of pyperformance 1.14.0 in one interpreter, for the pymalloc reuse study.

    python -S pyperformance-subset.py [--only NAME[,NAME...]] [--loops N]

The runner (run-cpython-reuse.py --workload pyperformance) stages this file beside pyperf/ (pyperf
2.10.0) and benchmarks/bm_NAME/ (pyperformance 1.14.0, unchanged), so `import pyperf` resolves
from this file's directory. Each benchmark's own run_benchmark.py runs as __main__, so its setup
code and default arguments are the suite's. Only the timing harness differs: pyperf.Runner is
replaced by FixedLoopRunner, which runs the benchmark in this process with a fixed loop count
instead of calibrating loops and spawning worker processes (exec is not available to a protected
program). bench_func(name, func, *args) calls func(*args) LOOPS times, as one pyperf value of
LOOPS loops does; bench_time_func(name, func, *args) calls func(LOOPS, *args) once, as pyperf
does. No warm-up: it serves timing, and this run measures allocation.

stdout holds one line per benchmark with the benchmark's return value where it is deterministic,
then PYPERFORMANCE-OK. Both arms must print the same stdout. Timings go to stderr.
--only (with --loops) and --loops serve calibration and override SUBSET.
"""
import argparse
import gc
import os
import runpy
import sys
import time

import pyperf

PYPERF_VERSION = '2.10.0'

# The pre-registered subset (capstone/docs/plans/2026-11-15-cpython-established-workload.md),
# in run order, with its loop counts: about 200 s of guest time per benchmark in the stock
# calibration run, and at least one loop.
SUBSET = (
    ('json_dumps', 6),
    ('json_loads', 167),
    ('richards', 1),
    ('deltablue', 9),
    ('chaos', 1),
    ('go', 1),
    ('nbody', 1),
    ('gc_collect', 37),
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--only', help='comma-separated benchmark names (calibration)')
    p.add_argument('--loops', type=int, help='loop count for every benchmark (calibration)')
    a = p.parse_args()

    if pyperf.__version__ != PYPERF_VERSION:
        raise SystemExit(f'pyperf is {pyperf.__version__}, the plan pins {PYPERF_VERSION}')
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmarks')
    plan = list(SUBSET)
    if a.only is not None:
        if a.loops is None:
            raise SystemExit('--only needs --loops')
        plan = [(n, a.loops) for n in a.only.split(',') if n]
    elif a.loops is not None:
        plan = [(n, a.loops) for n, _ in plan]
    for name, _ in plan:
        if not os.path.isfile(os.path.join(root, 'bm_' + name, 'run_benchmark.py')):
            raise SystemExit(f'{name}: not staged under {root}')

    pyperf.Runner = FixedLoopRunner
    for name, loops in plan:
        run_one(root, name, loops)
    print(f'PYPERFORMANCE-OK {len(plan)}', flush=True)


def run_one(root, name, loops):
    FixedLoopRunner.loops = loops
    FixedLoopRunner.calls = []
    path = os.path.join(root, 'bm_' + name, 'run_benchmark.py')

    t0 = time.monotonic_ns()
    namespace = runpy.run_path(path, run_name='__main__')
    elapsed = time.monotonic_ns() - t0

    if len(FixedLoopRunner.calls) != 1:
        raise SystemExit(f'{name}: {len(FixedLoopRunner.calls)} bench calls, expected 1')
    bench, result = FixedLoopRunner.calls[0]
    shown = '-' if result is None else repr(result)
    print(f'BENCH {name} bench={bench} loops={loops} result={shown}', flush=True)
    print(f'PYPERF-TIME {name} loops={loops} ns={elapsed}', file=sys.stderr, flush=True)

    # Drop the benchmark's globals and its cyclic garbage before the next one, as the end of
    # a pyperf worker process would.
    namespace.clear()
    del namespace
    gc.collect()


class FixedLoopRunner:
    """Stands in for pyperf.Runner inside one benchmark's __main__ block."""

    loops = None
    calls = []

    def __init__(self, *args, add_cmdline_args=None, **kwargs):
        self.metadata = {}
        self.argparser = argparse.ArgumentParser(add_help=False)
        self.args = None

    def parse_args(self, args=None):
        # The suite's defaults: no benchmark argument is passed.
        self.args = self.argparser.parse_args([])
        return self.args

    def bench_func(self, name, func, *args, inner_loops=None, metadata=None):
        result = None
        for _ in range(self.loops):
            result = func(*args)
        FixedLoopRunner.calls.append((name, result))

    def bench_time_func(self, name, func, *args, inner_loops=None, metadata=None):
        func(self.loops, *args)
        FixedLoopRunner.calls.append((name, None))


if __name__ == '__main__':
    main()
