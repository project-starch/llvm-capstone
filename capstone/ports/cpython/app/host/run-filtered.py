"""unittest -v for one test module, leaving out the tests named on the command line."""
import sys, unittest
module = __import__(sys.argv[1], fromlist=["*"])
skip = set(sys.argv[2:])
loader = unittest.defaultTestLoader
def keep(test):
    if isinstance(test, unittest.TestSuite):
        kept = unittest.TestSuite(t for t in map(keep, test) if t is not None)
        return kept
    return None if test.id().split(".")[-1] in skip else test
suite = keep(loader.loadTestsFromModule(module))
print("left out:", sorted(skip), flush=True)
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
