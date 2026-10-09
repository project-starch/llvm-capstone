# memcached/allocator-repros 06 and 07 under field bounds -- 2026-10-09

The two spatial cases that stay inside one slab chunk, built with each compiler's field-bounds flag
through the corpus's own seam (shared/build-cases.sh, CFLAGS appended to the toolchain's):

    CFLAGS='-Xclang -fcapstone-subobject-bounds' ... cmake --preset capstone-domain -DMCP_CORPUS_SRC=<case.c>
    CFLAGS='-Xclang -cheri-bounds=subobject-safe' bash shared/build-cases.sh cheribsd <out>

and run by runners/capstone-domain/run-defects.py (--cases 6,7 --modes spatial) and
runners/cheribsd/run-defects.py (--cases 6,7, stock CheriBSD, revocation on). The flags were
confirmed in each build's compile_commands.json. LIMIT: these two runners have no field-crossing
control of their own; the same flags with the same compilers DID narrow a field in the other
boots of this day (tools/run-capstone-domain.py's subobj control, CAUGHT; tools/subobj-control.c
on CheriBSD, SIGPROT), not in these.

## Capstone (application domain, level0 heap + C1 field bounds)

    case 6: expected complete, completed=1, passed=True, image 845c58a3d3921d55
    case 7: expected complete, completed=1, passed=True, image 86568ea5c092afbd

## CheriBSD (stock purecap, -cheri-bounds=subobject-safe)

    controls: cheribsd-abi exit 0 PASS; cheribsd-bounds exit 162 PASS; revocation-control exit 162 fault {'signal': 34, 'code': 2, 'addr': '0x101e4a', 'pc': '0x101e4a'}
    case 6: exit 0, completed=1, faults_seen=0, passed=True
    case 7: exit 0, completed=1, faults_seen=0, passed=True

These two arms were NOT pre-registered by commit: they were added during the session, after the
2852edcdd747 pre-registration. The expectation (NOT caught on both, the item storage being a
flexible array member neither compiler narrows) was stated before the runs and held.
