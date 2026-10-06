#!/usr/bin/env python3
"""Trigger for gh144169 -- the upstream fix's own regression test.

Upstream fix: bc92e7878f25
Test file at the fix: Lib/test/test_ast/test_ast.py
Test: ASTConstructorTests.test_non_str_kwarg

Run with the pinned 3.13.7 host-oracle interpreter. The test METHOD is
named explicitly so the run cannot pass by skipping it; the harness
requires the interpreter to confirm "Ran 1 test".
"""

# test.test_ast is pruned from the guest's Lib on the cheribsd arm, so
# "from test.test_ast.utils import to_tuple" at upstream_test.py:26 killed this
# case before any defect ran -- and the row then read as the arm staying quiet.
# The two modules the upstream file needs are vendored beside this trigger and
# registered under their stdlib names. The real package is used wherever it
# exists; this runs only when the import fails, so the host and the Capstone
# arms are unaffected.
def _vendor_test_ast():
    import importlib.util, os, sys, types
    try:
        import test.test_ast.utils, test.test_ast.snippets  # noqa: F401
        return "stdlib"
    except ImportError:
        pass
    import test
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = types.ModuleType("test.test_ast")
    pkg.__path__ = []
    sys.modules["test.test_ast"] = pkg
    test.test_ast = pkg
    for name, fn in (("utils", "test_ast_utils.py"), ("snippets", "test_ast_snippets.py")):
        full = "test.test_ast." + name
        spec = importlib.util.spec_from_file_location(full, os.path.join(here, fn))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        setattr(pkg, name, mod)
        spec.loader.exec_module(mod)       # snippets imports utils, already registered
    return "vendored"

_vendor_test_ast()
import runpy, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v", "ASTConstructorTests.test_non_str_kwarg"]
runpy.run_path("upstream_test.py", run_name="__main__")
