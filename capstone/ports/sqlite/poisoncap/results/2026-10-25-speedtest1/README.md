# PoisonCap's own SQLite benchmark (speedtest1), reproduced on their QEMU (2026-10-24/25)

Their code, their toolchain and their benchmark, without edits: what runs and what does not. QEMU only;
their published numbers come from their FPGA, which is not available here.

## What was built

- **Source:** their fork CTSRD-CHERI/sqlite at a879a48e (the commit the artifact's own `manifest` names)
  with the artifact's two uncommitted files laid over it (`src/mem5.c`, `test/speedtest1.c`). After the
  build the tree equals the artifact's `sqlite/` file for file (`speedtest1/build-speedtest1.sh` checks it), except
  in `build-memsys5-all32`, whose `test/speedtest1.c` is a879a48e's by design.
- **Build:** their pinned cheribuild (07d5e034), `sqlite-riscv64-purecap --sqlite/build-speedtest`, their
  SDK and CheriBSD sysroot (`build-*/`).
  - `build-cheribuild`: exactly that recipe. It does not enable memsys5.
  - `build-memsys5`: the same plus `--sqlite/configure-options=--enable-memsys5` (configure's own switch);
    20 main tests, as published.
  - `build-memsys5-all32`: as `build-memsys5`, with `test/speedtest1.c` from their commit a879a48e, in
    which all 32 main tests are active. Same `main()` set-up otherwise.
- **Kernels:**
  - their `CHERI-PURECAP-QEMU-POISON`, the platform's kernel: a debug kernel (WITNESS, INVARIANTS);
  - `CHERI-PURECAP-QEMU-POISON-NODEBUG` (`kernel-poison-nodebug/`): their FPGA kernel's options on the
    QEMU board. It is their `CHERI-PURECAP-QEMU-NODEBUG` plus the three option lines of their
    `CHERI-PURECAP-GFE-POISON-NODEBUG`; a new config file, no file of theirs changed
    (`platform/build-nodebug-kernel.sh`).
- **Arguments:** none. Their `main()` configures memsys5 itself: a 256 MiB heap from `malloc`, minimum 16.
  speedtest1's defaults are then `--size 100 --testset main` with a temporary database.

## Two revocation layers

- **libc:** `security.cheri.runtime_revocation_default`. It applies to every process exec'd while it is 1.
- **memsys5:** their `mem5.c` calls `cheri_revoke` from its quarantine. Their `main()` also calls it once
  at start-up. This happens whatever the default says.

"Default on/off" below is the libc layer only. No run here has memsys5's revocation off: that would need
an edit to `mem5.c`.

## Results

Runner: `speedtest1/speedtest1-guest.py`, single-user boot (their image's multi-user boot panics). One guest per run.
Each `*.json` is one run's record, written by the runner as it was at the time (`speedtest1/collect-speedtest1.py`
scrubbed paths and user@host strings and added `console_from_serial` and `runner_log_tail` where the
record lacked them). These records predate the runner's current fields: they have no binary or kernel
hash (see `build-*/SHA256SUMS` and `kernel-poison-nodebug/SHA256SUMS`), and `running_at_end` means the
last test seen started in the stdout the record holds. `seconds` is how long the run's ssh call lasted,
not the time of a panic.

| run | binary | kernel | libc default | outcome |
|---|---|---|---|---|
| `run-cheribuild-noargs-revoff` | cheribuild | debug | off | exit 1 at once: "Failed to configure memsys3 1" (no memsys5) |
| `run-cheribuild-noargs`, `run-memsys5-noargs` | both | debug | on from boot | the copy of the binary into the guest fails (`runner_log_tail`) and the kernel panics `share->excl` (WITNESS: an exclusive lock of the vm map at vm_map.c:6103 while it is share-locked from 5978); speedtest1 never runs. The faulting process is not identified |
| `run-E3-nodebug-revon-boot` | memsys5 | NODEBUG | on from boot | the copy of the binary times out after 120 s (`runner_log_tail`); no panic printed; speedtest1 never runs |
| `run-memsys5-noargs-revoff` | memsys5 | debug | off | tests 100-145 complete; 4 h cap reached inside test 160 |
| `run-E1-revon-witness`, `run-E2-revon-nowitness` | memsys5 | debug (E2: WITNESS off at run time) | on from the run | tests 100-145 complete; 1 h cap reached inside test 160 |
| `run-E3b-nodebug-revon-run` | memsys5 | NODEBUG | on from the run | tests 100-145 complete; 1 h cap reached inside test 160 |
| **`run-E3c-nodebug-revon-run-long`** | **memsys5 (20 tests)** | **NODEBUG** | **on from the run** | **all 20 tests complete, exit 0, 15,261 s (test 160: 14,476 s)** |
| **`R6-20-revoff`** | **memsys5 (20 tests)** | **NODEBUG** | **off throughout** | **all 20 tests complete, exit 0, 16,903 s (test 160: 15,958 s)** |
| **`run-E4b-…`, `R4-…`, `R4b-…`** | **memsys5-all32** | **NODEBUG** | **on from the run** | **kernel panic `Poison probe missing page 0x4fef8000` in `vm_cheri_revoke_probe_poison`; the run's ssh returned after 1,155-1,162 s. R4/R4b: in the last copy of the output (at 362 s) tests 100-145 are complete and test 150, "CREATE INDEX five times", has started. E4b kept no output** |
| **`R5-all32-revoff`** | **memsys5-all32** | **NODEBUG** | **off throughout** | **the same panic and page; ssh returned after 1,161 s; the last copy (362 s) shows the same: 100-145 complete, 150 started** |
| `R1-…`, `R1b-…` | memsys5-all32 | NODEBUG | on from the run | exit 162 (SIGPROT) at start-up, before any output: poison fault, write to the poisoned 4 KiB block at 0x40c0e000. These two ran with stdout piped (`tee` to the console); see below |

"On from the run": the runner sets the default to 0 from the single-user shell before `sshd` starts, so
`sshd` starts and the binary is copied in; a `--pre` command then sets it to 1, and processes exec'd from
then on (the run's shell, `timeout`, speedtest1) run with libc revocation. R4 and R4b also set it back to
0 two seconds later (`--after-start`), so that the output copies (`--poll`, every 30 s) are exec'd without
it; whether that took effect was not recorded, since the guest died. R5 had no `--pre`: the default was 0
throughout. In R4, R4b and R5 the last successful copy is from 362 s; nothing was copied after that.

## What follows

- Their published benchmark (20 tests) completes with both revocation layers active, on their FPGA
  kernel's options (E3c).
- It also completes with the libc default off throughout (R6). Test 160 dominates both runs (14,476 s on,
  15,958 s off). The two ran at different times on a shared host, so the times are not a comparison.
- Their own 32-test driver, with stdout to a file, ended in the same kernel panic in 4 of 4 runs. In the
  three whose output was copied, test 150 ("CREATE INDEX five times", the first of the 12 tests the
  artifact comments out) was the last test started 800 s before the panic was seen; that it was still
  running at the panic is likely but not observed.
- R5 shows the panic does not need libc revocation: the default was 0 throughout. Their memsys5 calls
  `cheri_revoke` itself (their `mem5.c`), the plausible trigger; the records do not identify the caller.
- The other 11 commented-out tests were not exercised: speedtest1 runs its tests in one sequence, and the
  run ends with the kernel.
- With the libc default on from boot, the copy of the binary into the guest fails: the debug kernel
  panics, the NODEBUG one does not answer within 120 s. The faulting process is not identified.
- Their paper v1 states that 17 of 32 tests "were either hanging or triggering a kernel panic", from
  "recursive page fault handling as a result of poison probes in the revoker". Their v3 keeps that
  limitation in its design section but shows SQLite results for all 32 tests.
- Not explained: with `--console-echo` (stdout and stderr through `tee` to the console), their speedtest1
  exited on a poison fault before printing anything (R1, R1b; 2 of 2); with stdout to a file it does not.
  What causes it (the pipe, `tee`, the console I/O) is not isolated. `--console-echo` is not used for
  results for that reason.

## Not here

- R2 and R3, runs with `--console-echo` and the libc default off, were stopped by hand after R1; E4 was
  stopped by hand for a longer cap. None is a result.
