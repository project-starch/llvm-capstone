# Q-21: the application runtime's `urevavail` read removed — census, equivalence, and the cost

The fix is `f6dfd19d9396`. This is its evidence: the census two-sided, the behaviour shown
unchanged, and the cost measured rather than asserted.

## Census — the gate is 0, and the pre-fix images are the positive control

Counted by ENCODING (opcode `0x73`, funct3 in {1,2,3,5,6,7}, CSR field `[31:20]`), never by
mnemonic: objdump prints some custom words as `.4byte` and a mnemonic grep reads 0.

| image | sha256/16 | insns | 0xcc0 / 0x5c0 reads |
|---|---|---:|---:|
| contract.dom (fixed) | `125705371837ea92` | 33,337 | **0** |
| sqlite3.dom (fixed) | `a5942a1add489597` | 351,558 | **0** |
| mruby (fixed) | `a902e740fc5d3b0f` | 455,792 | **0** |
| contract.dom (pre-fix) | `f2017064bea77dd2` | 33,345 | 2 — `malloc+0x3c0`, `realloc+0x438` |
| sqlite3.dom (pre-fix) | `233640f7193b034c` | 351,566 | 2 — same offsets |
| mruby (pre-fix) | `bba891bc16576d98` | 455,800 | 2 — same offsets |

Census exits 0 on the fixed images and 1 on the pre-fix ones, so it is proven able to produce both
answers. The pre-fix counts reproduce the board lane's census exactly, from an independently built
`llvm-objdump`.

**A cheap corroboration worth keeping:** every image's instruction count fell by **exactly 8** —
two `csrr` plus the add and compare/branch each guarded, at two call sites. A wholesale codegen
change would not land on exactly 8, so the delta is the removed pre-check and nothing else.

## Equivalence — the behaviour is unchanged

Both Tier records re-run on the same pin `852c5bc76774` with the fixed images:

```
Tier 1   T1_EXIT:ok:0  stale:139  cleanup:0   app_ok=1  cause25=1  cause24=0  cause26=0
         geometry node_bytes=1048576 node_capacity=65536
         serial 103,921 bytes -- the SAME size as the pre-fix run
Tier 2   T2_EXIT:sqlite_write:0  sqlite_read:0  mruby:0  cleanup:0
         SQLITE_SUM=5050  SQLITE_PERSIST=100  MRUBY_SUM=5050  MRUBY_FILE=virtual
         cause24=0  cause26=0  cause29=0  Scalar(0x1d)=0
```

Identical in every reading to the pre-fix records, which is what "the module already makes the same
decision" predicts: `ensure_nodes()` opens with `if (available >= required) return 0;` over the
table header, the authoritative copy of what the CSR mirrored.

## The cost, measured

| | service `steps` | QEMU elapsed |
|---|---:|---:|
| Tier 2 pre-fix | 1,953 | 4,980.9 ms |
| Tier 2 post-fix | **2,986 (+1,033, +52.9%)** | 5,050.3 ms (+1.39%) |
| Tier 1 pre-fix | 534 | 708.9 ms |
| Tier 1 post-fix | **547 (+13, +2.4%)** | 704.7 ms (−0.6%) |

**Read the step COUNT, not the milliseconds.** The elapsed figures are QEMU's and are not silicon
costs — the project's own rule is to take patterns from simulation and cycle values from silicon,
and Tier 1's −0.6% is plainly inside noise. The portable number is **+1,033 service steps** on the
app workload, a 52.9% increase in service steps, from the three `ensure_ids(256)` sites that
previously returned without a call.

**If that proves expensive on silicon**, the fast path can be restored WITHOUT a CSR: the module
already owns the available count and could publish it into the context frame, which the domain
reads as ordinary memory. That keeps Q-21's rule — the application never reads a CSR for
availability — while removing the per-mapping service call. Not done here, because it is an ABI
change and the measured wall-clock cost does not yet justify one.

## Scope of the records this supersedes

The Tier 1 and Tier 2 records on `852c5bc76774` taken with the pre-fix images were QEMU-only passes
with respect to this construct — the C-27 shape one level out. The runtime lane withdrew the Tier 2
record's "39/41 on one slot" reading on that basis and the withdrawal is correct. The C-27
conclusion is NOT affected: boot 3-T1 passed `__capstone_cap_init` on silicon, which is what that
fix was for, and faulted later and elsewhere.

The records above still run on a QEMU that implements `0xcc0`, so they cannot prove its absence
matters. The final re-record belongs on the pin the RTL lane is producing with `0xcc0` and `0x5c0`
removed from the default, where a read is illegal as on the RTL. Until that pin exists, **the census
is the gate** — and it is a static one, which is why it is reported for every image handed over.
