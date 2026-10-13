# Managed mallocng qualification, 2026-10-12

Virtual has one outer malloc/free implementation: Capstone-compiled musl
mallocng with managed group and object lifetimes. Allocation policy remains
musl's. Linear heap loans, per-slot split ownership, revocation handles and
UNINIT restoration are removed. The private metadata index and payload tag
clearing on free remain. See [the representation](../../mallocng.md).

The [manifest](qualification.json) identifies source files and build scope.
QEMU was built at pin `7dbfdd534eea`, with a matching fresh adapter, libc and
SDK. The compiler and Linux/firmware artifacts were reused. This is one-hart
QEMU evidence, with no new performance or RTL claim.

| Check | Result | Record |
| --- | --- | --- |
| Virtual musl compilation | 1,361/1,361 units | [Manifest](qualification.json) |
| Allocation policy | 22 checks pass; UNIT, group-count and realloc mutations rejected | [Controls](policy-controls.json) |
| Allocator | 19/19, including 70,000 live objects, 200,000 lifetimes, tagged transfers, exact-site denials and Linux mremap | [Allocator](allocator.json) |
| Runtime | 43/43, including nested retirement, threads, both recycling cases, SQLite persistence, mruby and Perl | [Runtime](runtime.json) |
| CPython, PostgreSQL, FFmpeg, tshark workloads | 16/16 with native output and changed-input controls | [Ports](ports.json) |
| FFmpeg carve | 13/13 missed by outer-only protection; 13/13 caught with inner protection; all fixed cases complete | [Stock](ffmpeg-stock/inputs.json), [protected](ffmpeg-protected/inputs.json) |
| PostgreSQL contexts | 5/5 missed by outer-only protection; 5/5 caught with inner protection; each safe variant completes | [Stock](postgres-stock/inputs.json), [protected](postgres-protected/inputs.json) |
| memcached outer-only | 81/81 checks | [Stock](memcached-stock.json) |
| memcached protected pools | **82/83; fixture 24 remains unresolved** | [Protected](memcached-protected.json) |

Fresh source builds covered memcached/libevent, the PostgreSQL component
corpus and both FFmpeg carve arms. Seven other full applications used cached
virtual upstream objects linked against the fresh libc/runtime; [commands
and input hashes](relink.json) identify them. This does not requalify their
entire upstream suites. The malloc contract's native RV64 reference uses
glibc for API/content checks, not a claim of identical mallocng placement.

The carved corpus also retires and reissues an unaligned range, checks a live
sibling and requires the old alias to fault at the labelled probe. PostgreSQL's
new `--virtual-kit` path runs defects and safe variants in one fresh boot.
memcached's opcode gate checks allocator objects against stock compilations:
whole-image counts now also include outer mallocng. Fixture 18 prints the
actual store target, base + 32; exact-address attribution remains required.

Fixture 24 returns `1800021` on the protected arm. Both page moves complete
with the reversal flag enabled, but this does not prove the intended
orphan-header dereference occurred. Its temporal-fault prediction is retained;
safe fixture 25 returns `1900020`. Establishing the mover's actual path is
still required. The protected gate is recorded as FAIL, not normalized to PASS.
Established since, on 2026-10-13: path witnesses in the mover show the reversed
header-only free branch is never taken for the fixture's header, on either arm,
so the orphan is never built and `1800021` was a return from a sequence that had
not created its own precondition. See
[ports/memcached/app/results/2026-10-13-fixture-24-path](../../../../ports/memcached/app/results/2026-10-13-fixture-24-path/README.md).

The [CPython archive audit](cpython-archive-audit.json) recomputes all 2,183
stock/protected outcomes in the committed sharded campaign: they agree,
including the completed long `pop` repeat. Earlier unsharded failures and
differences from native remain historical evidence. This is not a new suite run.

Host verification: 34 common tests pass with one skip, plus 25 runtime Python
tests, 84 corpus-judge tests, 87 runtime CTests and three safety-judge tests.
[Build controls](build-controls.json) verify rejection of physical heap
selectors in virtual builders and runners, including profiles read from VM
state. Corpus structure, port mappings, derived views and script syntax pass.

## Reproduce

Source `capstone/tests/capstone-test-env.sh`; build the matching QEMU,
adapter and SDK as in the [runtime guide](../../README.md). Use fresh build
directories after the mallocng representation change. The core gates are
`runtime/virtual/run-malloc.py`, `run.py` and `run-ports.py`; `--help` lists
their explicit inputs, and the records above identify executable hashes.

Use `bug-corpora/tools/run-virtual-cases.py` with the FFmpeg carved corpus,
virtual SDK and kit. Select `--arm virtual-malloc`, or
`--arm virtual-nested-pools --cc-arg=-DFFC_SUBLET_CARVE` plus the runtime
include path. PostgreSQL's [corpus recipe](../../../../bug-corpora/postgres/mmgr-repros/README.md)
documents paired component builds and `--virtual-kit`.

Build memcached with `ports/memcached/app/host/build-virtual.sh NEW_OUTPUT`;
run `host/run-virtual.py --arm virtual` and `--arm sublet` with the matching
platform inputs. The latter must retain the unresolved fixture-24 failure
until its defect or trigger is resolved.
