"""Run the threading gate inside the no-fork domain, with explicit skip reasons.

Usage: python run-thread-gate.py [TEST_MODULE ...]
Requires patch 0015 (subprocess via posix_spawn) and real thread-locals. First
check the ABI's fork refusal beside live threads, then tell upstream's existing
requires_fork decorators that fork is unavailable. Subprocess-only tests remain
enabled; no method-name filters hide their assertions. Prints a JSON result after
unittest's report, including every skip and failure.
"""
import errno
import json
import os
import sys
import threading
import unittest
import warnings
from test import support


def check_fork_refusal():
    stop = threading.Event()
    ready = [threading.Event() for _ in range(4)]

    def worker(started):
        started.set()
        stop.wait()

    threads = [threading.Thread(target=worker, args=(event,)) for event in ready]
    try:
        for thread in threads:
            thread.start()
        if not all(event.wait(10) for event in ready):
            raise AssertionError('fork control workers did not start')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                pid = os.fork()
        except OSError as exc:
            if exc.errno != errno.ENOSYS:
                raise
        else:
            if pid == 0:
                os._exit(97)
            os.waitpid(pid, 0)
            raise AssertionError('this gate requires fork() to return ENOSYS')
    finally:
        stop.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join()
    print('fork beside four live threads: ENOSYS; all four joined', flush=True)


check_fork_refusal()
support.has_fork_support = False
modules = sys.argv[1:] or [
    'test.test_threading', 'test.test_thread', 'test.test_threading_local',
    'test.test_queue', 'test.test_concurrent_futures.test_thread_pool',
]
suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
result = unittest.TextTestRunner(verbosity=2).run(suite)
print('THREAD_GATE_RESULT=' + json.dumps({
    'modules': modules, 'ran': result.testsRun,
    'skipped': [{'test': test.id(), 'reason': reason} for test, reason in result.skipped],
    'errors': [{'test': test.id(), 'traceback': tb} for test, tb in result.errors],
    'failures': [{'test': test.id(), 'traceback': tb} for test, tb in result.failures],
    'unexpected_successes': [test.id() for test in result.unexpectedSuccesses],
    'success': result.wasSuccessful(),
}), flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
