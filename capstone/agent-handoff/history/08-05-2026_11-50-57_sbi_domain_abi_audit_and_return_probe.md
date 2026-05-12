# sbi.dom ABI audit and DOM_RETURN probe

Timestamp: 2026-05-08T11-50-57Z

## Purpose

Follow up on the failed shared-region sentinel probe with the narrowest next check:

> audit the `sbi.dom + .smode` ABI and test whether the failure was caused by using raw `ecall` instead of the full `SBI_EXT_CAPSTONE_DOM_RETURN` calling convention.

## Source-backed audit result

The audit strongly suggests the following:

1. `sbi.dom` installs `_cap_trap_entry` from `capstone-sbi-domain/capstone-sbi/sbi_capstone.S`.
2. That trap handler restores `a0..a7` and dispatches `Supervisor ecall` through `handle_trap_ecall(...)` in `capstone-sbi-domain/capstone-sbi/sbi_capstone.c`.
3. Therefore, in the `sbi.dom` path, a custom `.smode` payload should follow the full SBI calling convention (`a6 = fid`, `a7 = ext`) rather than relying on a raw `ecall` with only `a0` set.
4. This differs from the separate `smode.dom + smode.smode` path, where the custom trap wrapper is different and raw `ecall` is used by the sample.

Most likely narrow misunderstanding before the test:
- raw `ecall` was probably the wrong return mechanism for the `sbi.dom`-based probe.

## Concrete change tested

The custom `.smode` probe was changed from:
- raw `ecall` after writing the sentinel

to:
- explicit `sbi_ecall(SBI_EXT_CAPSTONE, SBI_EXT_CAPSTONE_DOM_RETURN, ...)`

That is, the probe now returns through the same high-level calling convention used by `capstone-test-domains/src/sbi.smode.c`.

## Runtime result

The probe still failed.

Observed key output:

- `shared-region-probe: created domain ID = 0`
- `shared-region-probe: created shared region ID = 0`
- `shared-region-probe: initial word = 0x0000000000000000`
- `shared-region-probe: region shared`
- `shared-region-probe: call 1 retval = 0`
- `shared-region-probe: word after call 1 = 0x0000000000000000`
- `shared-region-probe: stage 1 sentinel mismatch (observed=0x0000000000000000)`

Logs:
- build: `/tmp/capstone/build-shared-region-probe-abi-fix.txt`
- wrapper: `/tmp/capstone/run-shared-region-probe-wrapper-abi-fix.txt`
- QEMU: `/tmp/capstone/capstone-runtime-qemu-shared-region-probe.log`

## Conclusion

The raw-`ecall` vs `DOM_RETURN` ABI mismatch was a plausible hypothesis, but switching to explicit `SBI_EXT_CAPSTONE_DOM_RETURN` did **not** make the sentinel visible.

So the blocker is now more likely to be in one of these areas:

1. the custom `.smode` payload is still not executing as expected,
2. `share_region()` into `sbi.dom` does not populate the state later used by `SBI_EXT_CAPSTONE_REGION_QUERY` in the way we assumed,
3. the queried region capability/base is not usable or not the same memory view seen by the guest userspace mapping,
4. there is a runtime / wrapper-system bug in the `capstone-sbi-domain` path.

## Practical recommendation

The next technical step should now target runtime/wrapper plumbing, not another HostCall ABI design iteration.

