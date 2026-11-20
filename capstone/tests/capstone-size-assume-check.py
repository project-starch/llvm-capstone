#!/usr/bin/env python3
"""capstone-size-assume-check.py <elf>...: check a statically linked Capstone image against the size assumptions in its
.capstone.size_assume section.

The check itself is capstone-cc's (capstone/runtime/host/capstone_vm/compiler.py, check_size_assumptions), which runs
it on every image it links and removes an image that fails. This is the same function for an image linked some other
way, or linked before the driver ran it. Run it on the linked image before stripping: it needs .symtab.

Exit 0: every record holds, or the image carries no records (it prints how many it checked).
Exit 1: a record fails.
Exit 2: an image could not be read, or it has records but no symbol table.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runtime" / "host"))
from capstone_vm.compiler import SizeAssumptionError, check_size_assumptions  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rc = 0
    for path in sys.argv[1:]:
        try:
            symbols, records = check_size_assumptions(Path(path).read_bytes(), path)
        except SizeAssumptionError as error:
            print(f"FAIL {error}")
            rc = max(rc, 1)
            continue
        except (OSError, ValueError) as error:
            print(f"{path}: cannot check ({error})")
            rc = max(rc, 2)
            continue
        if records:
            print(f"{path}: {symbols} symbols / {records} records checked, 0 failing")
        else:
            print(f"{path}: no .capstone.size_assume section, 0 assumptions")
    sys.exit(rc)


main()
