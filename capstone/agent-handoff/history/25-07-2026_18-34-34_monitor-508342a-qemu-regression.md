# QEMU regression of the 508342a-built monitor (adoption check)

**Date:** 2026-07-25 · **Lane:** B · **Board-free.** Follows
`25-07-2026_16-51-09_monitor-regen-not-broken-in-b-tree.md`, which established that the
monitor rebuilds and boots with either compiler. This is the broader pass needed before
**adopting** the `508342a` build as the default (it is strictly smaller: identical codegen
minus ~192 KiB of spurious 64 KiB `.gct` padding).

## Method

Install `fw_jump.elf` built with `capstone-c` `508342a` (`a56492c4`), run a
monitor-focused probe subset serially (shared `rootfs.ext2` write-lock), then **always**
restore the status-quo image (`2311c0b9`) via an `EXIT` trap. Verified restored afterwards.

Exit-code classification matters here: the harness uses **rc 75 = `__CAPSTONE_INFRA_FLAKE__`**
(e.g. a 9p `mount` failing during guest setup). Treating 75 as a failure produces phantom
regressions — it did once during the `-fno-jump-tables` rung sweep before I separated it.

## Result: 6/8 pass, 2 investigated

| probe | 508342a | status-quo control | verdict |
|---|---|---|---|
| run-smoke | PASS | — | |
| run-shared-region-probe | PASS | — | |
| run-revoke-matrix-probe | PASS | — | |
| run-hier-revoke-probe | PASS | — | |
| run-hostcall-fileread-probe | PASS | — | |
| run-linear-uninit-corpus-probe | PASS | — | |
| **run-borrow-revoke-uaf-probe** | FAIL rc=1 | **FAIL rc=1** | **pre-existing, NOT a regression** |
| **run-intra-domain-mrev-revoke-probe** | FAIL rc=1 | PASS | **under investigation** |

### `run-borrow-revoke-uaf-probe` — not a regression

It fails identically on the status-quo monitor, so `508342a` is not implicated. Its
required marker is literally

```
borrow-revoke-uaf-probe: NO-TRAP-GAP use-after-revoke store landed
```

i.e. the probe **asserts a known trap gap** — it demands that a use-after-revoke store
*still lands*. It therefore fails whenever that store does not land, which is arguably the
*better* behaviour. Worth revisiting on its own merits; unrelated to the compiler.

### `run-intra-domain-mrev-revoke-probe` — regression vs flake, being separated

This is the only candidate for a genuine `508342a` regression, and n=1 is not enough to
call it. Its own log shows retry churn on the 508342a run —
`...no boot/fault for held_revoke_fault (attempt 1), retrying` — and every sub-probe it
printed passed (`held_protected_value_lifecycle`, `held_revoke_fault`,
`held_mem_alias_fault`, `held_ambient_miss`), with the failure in an unprinted one. That is
the signature of a flaky probe, not a clean functional break.

**RESOLVED: it is a FLAKE, not a regression.** Re-ran 3× on `508342a` → **3/3 PASS**, and
2× on status-quo → 2/2 PASS. The single failure in the batch was not reproducible.

## Verdict: `508342a` ADOPTED

With both failures accounted for (one pre-existing and unrelated, one non-reproducible),
`508342a` is clean and strictly better, so it is now the resident monitor in B's tree:

- `capstone/caplifive-buildroot/build/images/fw_jump.elf` = **`a56492c4`** (508342a build,
  1,517,952 B — down from 1,714,600 B, i.e. ~192 KiB of pure `.gct` alignment padding gone).
- Re-verified after the swap: `run-smoke.sh` passes (boot → create domain → call →
  `retval = 42`).
- Previous status-quo image kept at `/tmp/capstone-b/monitor-regen/fw_jump.elf.b-jul19.bak`
  (`2311c0b9`) if a revert is ever needed.

**Build monitors with `caplifive-system`'s pinned `capstone-c`:**
`CAPSTONE_CC_PATH=$(realpath capstone/caplifive-system/sw/capstone-c)` (branch `bugfix`,
`508342a`), and `CARGO_TARGET_DIR=/tmp/capstone-b/...` because that submodule's `target/`
is root-owned.

## Reusable notes

- Classify **rc 75 as a flake, not a failure**, in any batch runner over these probes.
- Always drive the comparison with a **status-quo control run of the same probe**. Both of
  the "regressions" here looked real until controlled; one dissolved immediately.
- The `EXIT`-trap restore pattern is worth keeping: a crash mid-suite otherwise leaves B's
  `fw_jump.elf` swapped, which silently changes every later QEMU result.
