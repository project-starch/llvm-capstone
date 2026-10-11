# memcached on silicon — what the platform has to provide, measured (2026-10-04)

The goal is a delegated application on the FPGA, not only in QEMU, with memcached as the target. It needs the FPGA
monitor (supervised CALL), and a delegated runtime that can run on silicon at all. What follows is what is MEASURED,
what is BLOCKED, and what is still to build. It is written for whoever builds the silicon runtime.

## The finding that sizes this
Every SDK image on dev (memcached, FFmpeg, tshark, ...) links `ports/musl-capstone/runtime/start-musl.S` with
`my_first_domain/link.ld`, and relies on QEMU fabricating `gp`.
- memcached.dom has 7,958 `delin gp` call sites.
- FFmpeg with `CAPSTONE_GP_FABRICATE=0` never runs.
- No gp-captable delegated runtime exists yet (contexts, seal, offer, recovery block, TLS).
  **2026-10-05: one now exists for a single context.** B0's hello-world runs byte-exact on silicon
  (`docs/plans/b0-silicon-delegated-runtime.md`): full LTO, the yield, HELLO, delegated syscalls and exit.
  Contexts, threads, a non-empty TLS data image and init arrays are still to build (B1).
- **musl's memcpy loses data on silicon (ISSUES R-29; found by B0, reproduced in simulation 2026-10-05).**
  - R-29's general statement: a 128-bit load of a granule returns a WRONG HIGH HALF whenever a plain store to that
    granule is still in the write buffer.
    - It reads 0 when the fresh store is to the low word.
    - It reads the OLD value when the fresh store is to the high word.
    - The low word and plain `ld` are always correct, and a fence clears both faces.
  - musl-capstone's memcpy granule loop does exactly that load, for any freshly written buffer or struct, in both
    directions.
  - B0 avoided it only in the delegate runtime (`dl_bytes`).
  - **The runtime memcpy's guard exists and holds on silicon (B0.8, 2026-10-05).**
    - `CAPSTONE_MEMCPY_PLAIN_GUARD` in `string_bounds_safe.c` checks the type of what the 128-bit load returned and
      copies plain data with `ld`/`sd`.
    - The unguarded control miscopied 94/96 across R-29's three faces; the guarded copy 0/96.
    - It is on in the silicon build (build-b0-hello.sh).
  - **Still needed for memcached:** compiler-emitted aggregate copies (struct assignment) take the same granule
    load. The SQLite silicon build guards those with its W-12 pass; a memcached silicon build needs the same pass, or
    the RTL fix (R-29's fix fork is back with the lead).
That runtime is the bulk of the work. It is unassigned; the runtime lane and the compiler lane (I-8, one-unit LTO)
are the owners on record.

## Measured on `caplifive_supcall_36a641e0b.bit`
- **Atomics work through a capability in capability mode: 22/22 exact** (`tests/rtl-smoke/cap-atomics-2026-10-03/`).
  - Covered: lr/sc .w/.d, amoadd/amoswap .w/.d (32-bit wrap included), SC failure with no reservation and on another
    granule, a musl `a_cas` loop.
  - memcached links 171 atomics of exactly these forms. Not covered: AMO bounds enforcement, the I4 tag residual,
    multi-hart contention.
- **There is no `time` CSR on silicon.**
  - `csr_regfile.sv` at 36a641e0b has no `CSR_TIME` read case, so `rdtime` (0xC01) falls to
    `default: read_access_exception`. Linux user code still works, because OpenSBI emulates it.
  - In a capability domain, though, `rdtime` is an unhandled illegal instruction: no trap vector unsupervised, a fault
    event supervised.
  - The musl runtime's `dl_clock` (`delegate.c:58-81`) executes `rdtime` whenever the host's launch record carries a
    non-zero `ticks_per_second`.
  - `exec.c`'s `timebase_frequency()` reads `/proc/device-tree/cpus/timebase-frequency`, which the board's DTS
    (`caplifive.dts`) sets to 12,500,000.
  - **So on the board every `clock_gettime` in a domain would fault, and libevent calls it on every loop.**
  - Fix, when a silicon runtime exists to test it: `ticks_per_second = 0` on this CPU (cpu compatible `"eth, ariane"`).
    **Done in B0** (`runtime/linux/exec.c`, `cpu_without_time_csr`). B0's application ran on the board with it and
    made no clock call; a clock-calling application on silicon is still to come.
    `dl_clock` then returns 0 and the libc delegates the call.
- **A supervised domain may not touch any plain CSR with addr[9:8] != 0**, by design (`csr_regfile.sv:2879-2886`).
  - The my_first_domain-family glue (`start.S`, `start-fpga-gpseed.S`, the gp-captable copies) saves and restores
    mcause/mtval around domreturn. Every `mcycle`/`minstret` cycle bracket is also affected.
  - `start-musl.S` has none. Its CCSRRW to cscratch/ctvec/cepc is exempt: the gate covers only CIH and CPMP under
    `ccsr_en`.
  - The static check is `tests/rtl-smoke/supmon-2026-10-03/sup-static-audit.py` (controls both ways). memcached
    mc19-level0 audits clean apart from the next item.
- **CALL is illegal inside a supervised domain** (`decoder.sv:1289`).
  - The musl runtime's `__capstone_context_call` is "a nested, unsupervised entry from this context".
  - On dev, nothing in memcached or musl calls it directly. Only `context-probe.c` does (checked by the compiler lane;
    direct calls only).
  - A silicon runtime whose contexts run supervised must have the monitor step them instead.

## Blocked
- **S-16: RESOLVED 2026-10-04 on `caplifive_supcall_715bdd1fe.bit`** (the S-16 entry in ISSUES).
  - The supervised SQLite speedtest now preempts 1,277 times at +0.69 % with no fence.
  - The history below is kept: a supervised CALL's switch could stall forever on silicon
    (`tests/fpga-repros/S16-...`).
  - The FPGA monitor's resume path hit it after 212 good resumes of the SQLite speedtest.
  - A bare image hits it within 16 resumes.
  - Until there is an RTL fix, supervised contexts (preemptive threads) on silicon stall after a few hundred quanta.
  - **Withdrawn (2026-10-04):** "a memcached run with UNSUPERVISED contexts does not depend on it".
    - A PLAIN CALL is exposed too: bare on silicon, plain CALLs after 4, 24 and 32 stores hang at the exchange,
      idx 4.
    - The FPGA monitor's plain domcalls have not shown it.
    - Until the RTL fix, a silicon runtime without quantum preemption should drain the store buffer (`fence`)
      immediately before every CALL and every RETURN it emits. That completed every bare twin that hung.
- **S-17:** after a domain switch, an LDC right behind `ccsrrw sp <- cscratch` can hang (LSU not ready),
  supervised or not (`tests/fpga-repros/S17-...`).
  - Generated code in the FPGA monitor's `__domcallsaves` does load through the just-restored sp. A runtime's post-call
    glue must not put an INDEPENDENT capability load there.

## To build, in order
1. **4b, the gp-captable delegated runtime.** A silicon `start-musl.S` with contexts, seal, offer, the recovery block
   and TLS; linker-script symbol and code-capability fixes; a gp-captable musl. Milestone B0: a hello-world delegated
   app through `capstone-exec` on the board.
2. **The `ticks_per_second = 0` change above**, tested by B0's first `clock_gettime`.
3. **4c, memcached as one unit** (full LTO, I-8), then the carve-count check: about 435 globals.
4. **4d, the milestones.**
   - B1: `mc-threads-probe` over loopback.
   - B2: memcached `-t 1 ... -m 8 -U 0 -l 127.0.0.1` answering `version`/`set`/`get`, then exiting 0 on SIGTERM.
   - B3: the full oracle by hash.
