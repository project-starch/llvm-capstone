# Block addresses were never materialized, and that was one `isa<>`

`CapstoneCapGlobalInit` synthesizes a per-module `__capstone_cap_init` that turns
every pointer in static data into a real capability at startup, because a tag
cannot live in an ELF image. Its leaf predicate was:

```cpp
return isa<GlobalVariable>(C) || isa<Function>(C);
```

A `BlockAddress` is neither, so a computed-goto dispatch table kept its link-time
addresses and the first `goto *tbl[i]` jumped outside the image. It read like an
ABI limit for weeks. It was a missing case.

## What it affected, measured

Counted over the WAMR amalgamation's global initializers (`scan-capleaves.py` over
`-emit-llvm` output), classic interpreter, one translation unit:

| | computed goto | switch |
|---|---|---|
| capability slots in initializers | 304 | 48 |
| materialized | 39 | 39 |
| null / undef (no tag needed) | 41 | 9 |
| **blockaddress, skipped** | **224** | 0 |

All 224 sit in one global, `wasm_interp_call_func_bytecode.handle_table`, and
224 + 32 null entries = the 256 opcode slots. The image agrees: `.rodata` of the
computed-goto build holds 258 words pointing into the image, the switch build 34.

Reach beyond WAMR, from the sources:

- **MicroPython** uses the same construct in `py/vm.c`. `MICROPY_OPT_COMPUTED_GOTO`
  defaults to 0 and we never set it, so our MicroPython worked by accident of a
  default. Turning on the standard performance option would have broken it silently.
- **WAMR's fast interpreter** (`wasm_interp_fast.c`) uses it too; our config
  excludes that file.
- sqlite, lua, jerryscript, beebs, rv8, coremark and musl: no `goto *` at all.

## The fix

`isa<GlobalValue>(C) || isa<BlockAddress>(C)`. `GlobalValue` rather than
`GlobalVariable || Function` also picks up **aliases**, which were skipped for
exactly the same reason and had no test.

Nothing was needed in the backend proper: `lowerBlockAddress` already produces the
same `LGA` node `lowerGlobalAddress` does, so the store lowers to `auipc` +
`cincoffset gp` + `delin` + `stc` like any other. The whole cost was the decision to
emit it.

## And the part that matters more than the fix

The pass used to skip a capability leaf **silently**. That is what let this survive:
the build is clean, the image is well-formed, and the fault lands far away with
nothing pointing back. It now warns, once per leaf, naming the holder. Exempt from
the warning: null (needs no tag) and `inttoptr` (an absolute address cannot carry
one, and MicroPython's `MP_ROM_INT` is a deliberate instance).

A vector of capabilities cannot be reached by the GEP path this pass builds, so it
is reported rather than walked. No configuration here generates one.

## Evidence

- `llvm/test/CodeGen/Capstone/static-cap-global-init-blockaddress.ll` checks the
  store TARGETS (`%pcrel_hi(.Ltmp0)`, `.Ltmp1`, `ali`), not a count, and pins that
  null and `inttoptr` still produce nothing.
- `capstone/tests/probe-cap-init-coverage.sh`: 6 stores for pointer globals, 2 for
  block addresses, 0 for null/absolute/int. It was written the day before with the
  arms inverted, to fail if the gap were ever closed, and it fired on exactly the
  build that closed it.
- WAMR built with the upstream default, 263 stores in `__capstone_cap_init` =
  39 + 224 as predicted, image 264896 -> 282112 bytes (+6.5%).
- The full stage-4 run under QEMU returns `0x5741002A`.
- `llvm/test/CodeGen/Capstone/`: 60 tests, all passing. Full `llvm/test/CodeGen`
  plus `MC`: 36194 tests, 6 failures, all the `emutls`/`tls-android` set measured
  pre-existing on 2026-08-24. `createCapstoneCapGlobalInitPass` has exactly one
  caller, `CapstoneTargetMachine.cpp`, so it cannot run in an X86 or RISC-V
  pipeline at all.
- A build with no block addresses and no aliases is byte-identical to the same
  build before the change (same md5 over 264896 bytes): a strict addition.

## Consequence

`WASM_ENABLE_LABELS_AS_VALUES` is back to upstream's default of 1 in
`build-wamr-silicon.sh`. The knob stays as the A/B arm and as a fallback. The 2x2
that attributed the alignment fault gains a row rather than losing one:

| | computed goto | switch |
|---|---|---|
| no pad | cause 4 | cause 4 |
| pad, before this fix | cause 1 | 42 |
| pad, after this fix | **42** | 42 |

## QEMU corpora

Run over this commit, serialized, one QEMU at a time.

| | |
|---|---|
| lit Capstone (CodeGen + clang) | 71 / 71 |
| lit RISCV + Generic | 2458 / 2458 |
| smoke, coremark, borrow-cost, shared-region, revoke-matrix, tree-cost-O2 | PASS |
| static-cap-globals | PASS, and it is the probe most directly on this change |
| wamr | PASS, retval `0x5741002A` |
| sqlite-memory | PASS |
| sqlite-slt | `records=1031 stmt_pass=31 stmt_fail=0 query_pass=1000 query_fail=0` |
| revoke-on-free / hier-revoke | 9 / 9 each, O0 / O1 / O2 |
| rv8 | 7 / 7 |
| beebs | 82 / 82 |
| intra-domain-mrev | UNRESOLVED, three attempts exhausted their retries on a different arm each time; every arm that DID complete reported the correct cause and retval |
| authority | UNRESOLVED, incomplete at ~30 of 32 domains when the run was cut off |
| linear-uninit-corpus | not run |

**The two unresolved suites were already flaky before this branch.** The nightly of
2026-08-25, one day before any of this work, records `authority | FLAKE` and
`linear-uninit-corpus | FLAKE` in exactly the same way. Checked directly rather than
taken on report.

**Zero occurrences of the new warning** across 231 suite logs, and the grep pattern
was positive-controlled against a synthetic line so the zero is a measurement. That
is consistent with the census: beebs, rv8, coremark and sqlite have no block
addresses or aliases at all, and WAMR's 224 are materialized rather than reported.

## Open, and NOT caused by this change

`run-domain-smoke.py` deliberately omits `infra_phase=` for the workload command,
so a domain that produces no output at all is reported as a hard `exit=1` instead of
the soft flake every other phase gets. Two unrelated domains hit that today (WAMR and
beebs `newlib-exp`), both passing immediately on retry, and a caller without its own
retry loop sees a FAIL. Worth deciding on separately; it is a harness question, not a
codegen one.
