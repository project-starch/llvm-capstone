# B0 — a delegated application on silicon: hello-world through capstone-exec with a real gp

Part 4b, milestone B0 of `memcached-on-silicon.md`, was assigned to the board lane by the lead on 2026-10-04.

**B0 is done when** a hello-world application (`puts` + `exit`, musl runtime) is launched by `capstone-exec` on
the FPGA board under `caplifive_supcall_715bdd1fe.bit` and prints its line. It must use the delegated runtime, not
the monitor's classic `call_domain`, with gp built from the image's capability table.

That run becomes the control for every later step: memcached, and the moved monitor checkpoints (delegate-bench,
context-probe, pthread-probe).

## What is true today (verified 2026-10-04 unless marked)
- **QEMU fabricates gp for every SDK image; silicon does not.**
  - `start-musl.S` reaches all globals and code through gp (26 references) and never establishes gp.
  - QEMU makes gp the domain's PCC at cursor 0 unless `CAPSTONE_GP_FABRICATE=0` (capstone-qemu op_helper.c,
    default ON).
  - On silicon there is no such gp. FFmpeg with `CAPSTONE_GP_FABRICATE=0` never runs (its results README).
- **The silicon ABI exists for classic domains.** The gp-captable glue is
  `tests/runtime-qemu/silicon-ladder/start-gp-captable-interp.S`:
  - its carve reads `.capstone_gp_initdesc` from the blob the monitor copies to the front of dom_data;
  - the monitor splits `dom_gp` off the image;
  - the SQLite speedtest, k800 and the ladder run on the board with it.
- **The monitor refuses exactly the combination B0 needs.** capstone-sbi `monitor/supcall-fpga` 78151e4,
  `sbi_capstone.c`:
  - `:1612-1615`: a MANAGED (process-ABI) domain with `gpoff != 0` is refused.
  - The reason: a managed domain loses the top `CONTEXT_DESC_AREA` (1024 B, `process-abi.h:53`) of its data region to
    context descriptors (`:1766-1774`), and the gp park writes at `base + tot_size - 16` (`:1887-1889`), inside that
    area.
  - The blob-fits guard (`:1827-1830`) also ignores the descriptor area.
  - The QEMU-pinned monitor has the same refusal (4674ab6a:1403, per the prior-art sweep; not re-read).
- **Two runtime files reference linker-script symbols:** `hostcall.c:229-230` (`__init_array_*`,
  `__fini_array_*`) and `tls.c:47` (`__capstone_tls_image`, `__capstone_tdata_end`, `__capstone_tls_end`).
  - Under gp-captable a declaration with no definition is reached by deriving from gp (`cincoffset gp` + `delin`).
  - On silicon a `delin` of a non-linear capability faults (C-13).
- **I-8:** in a multi-TU image, cap-table slot indices are TU-local, so they collide.
  - The remedy is full LTO, one module (`tests/runtime-qemu/gp-free-domain/multi-tu-slot-collision.sh`). The compiler
    lane re-ran it on 2026-10-04: non-LTO [0,1,2]/[0,1,2]; LTO [0,1,2]/[3,4,5].
  - `-mllvm` options do not survive into LTO. `-capstone-gp-captable` and the silicon shrink flags must be re-passed
    with `--plugin-opt` (c128 line, 5f9fd5e92283; not re-read).
- **Prior art.** The c128 line (origin/rebased/c128-3-musl, dadde4dd1bc7) ran musl + mruby with gp-captable + full
  LTO + the interp glue, with ZERO gp fabrications, on QEMU only.
  - It was the old HostCall v0 runtime, not the current process ABI.
  - Its LTO musl archive build carries a per-member codegen verifier that was never landed.
- **There is no `time` CSR on silicon.** `dl_clock` executes `rdtime` (delegate.c:58-81) when `ticks_per_second` is
  non-zero.
- **The process ABI** (`SBI_CAPSTONE_PROCESS_*`, `capstone_step`) has never executed on silicon.

## Constraints to design against (the compiler lane, 2026-10-04)
- **(a) Variable aliases under gp-captable are unsupported** (the C-75 residual). musl `fork.c` has 11 weak data
  aliases of `dummy_lockptr`. A hello-world must not link `fork.o`; check the link map. memcached will need this
  settled.
- **(b) Under full LTO one un-selectable libc member fails the whole link.** Use a per-member verifier, and keep the
  member set minimal.
- **(c) Slot measurements must defend against LTO folding:** volatile, noinline, and a positive control.
- **(d) C-74:** an 8/16-bit atomic on a self-bounded small object faults. Audit the runtime for sub-word atomics on
  lone globals.

## The compiler lane's review of this plan (2026-10-04; their record is i8-b0-review.md in their scratch)
- **The silicon flag set is four `-mllvm` options**, the ones dev's silicon builds (r1, micropython) pass.
  - The set:
    - `-capstone-gp-captable`;
    - `-capstone-shrink-stack=false`;
    - `-capstone-shrink-globals=false`;
    - `-capstone-merge-string-constants=true`;
    - plus `-DCAPSTONE_GP_CAPTABLE_ABI=1`, which is compile-time.
  - Under LTO all four need `--plugin-opt=` forms. c128's 5f9fd5e92283 re-passed only the first.
  - Dropping merge-string-constants is not cosmetic: micropython records 232 carves with it, 633 without.
- **Jump tables:** `-fno-jump-tables` survives LTO as an IR attribute.
  - But on this target no jump table appears even under the default ABI (a 24-case switch gives 0 `.LJTI` either
    way), so it is not established what suppresses them.
  - Do not count on that flag as the protection. B0.5 counts `.LJTI` and indirect jumps in the image.
- **Accessors over linker-script-filled globals:** the latter is the C-13 shape again. An accessor must return a
  VALUE, not a pointer into a glue-owned object whose bounds C then re-derives.
- **A third site: `context.c:38` `extern char __capstone_context_entry[];`.** A code label declared as data gets
  DATA bounds under gp-captable (`scc a0, gp, a0; delin a0`).
  - The fix is one line: declare it as a function.
  - It is in the same remediation pass as tls.c and hostcall.c, even though B0 itself mints no contexts.
  - The codegen shape is shown two-sided. A runtime fault is not shown.

## Steps
Each step has a gate that can fail, and is shown to fire before it is trusted.

**B0.1 Monitor: admit a managed domain that declares globals.** In both the FPGA monitor (monitor/supcall-fpga)
and the QEMU-pinned one:
- park gp at the top of the data region the domain actually receives, `base + tot_size - CONTEXT_DESC_AREA - 16`,
  when managed;
- count the area in the blob-fits guard;
- drop the refusal.
- **The glue's side of that contract:** find gp at its data capability's END minus 16, not at a fixed offset. That
  is checked in B0.2.
- **Gate:** the QEMU process-ABI suites are unchanged, since managed images have gpoff = 0. A managed gpoff != 0
  image boots, with a must-fail control: the old monitor refuses it.

**B0.1 status (2026-10-04):** written as capstone-sbi `monitor/b0-managed-gp` 3be6737.
- The refusal is dropped. `data_top = tot_size - CONTEXT_DESC_AREA` for a managed domain is used by both blob guards
  and the gp park.
- It compiles for FPGA with supervision, FPGA plain and QEMU, and the firmware passes the dom_stack gate.
- Not yet run: it needs the B0 image. A park move is enough, because the interp glue never reads the park: it carves
  its own table from its data capability's END (`start-gp-captable-interp.S:451-455`), which after the descriptor
  split is `data_top`.

**B0.2 Glue: `start-musl-gpct.S`.** The interp glue's carve, prologue and reentry, plus start-musl's recovery block,
`__capstone_yield`, contexts, seal/offer and `domain_main` call.
- **What in today's `start-musl.S` cannot carry over (read 2026-10-04).**
  - It treats gp as the domain's CODE capability at cursor 0. It loads data through it (`cincoffset t3, gp, t3;
    ld t4, 0(t3)` for `__capstone_context_arena_bytes`) and forms every code pointer from it (`domain_main`, the
    cap-init initializers, the fault vector).
  - On silicon a code-authority capability cannot load data (the 2026-07-22 root cause in the monitor's comment).
- **Data the glue defines in assembly, which C references:**
  - `__capstone_context_arena` (.data);
  - `__cp_begin/__cp_end/__cp_cancel` (.rodata, compared by address in musl's `pthread_cancel.c`);
  - `__capstone_gct_start`.
  - Under full LTO an assembly definition is outside the module, so C's `extern` is an undefined declaration: the same
    gp-derivation plus `delin` as a linker-script symbol, fatal on silicon.
  - These move into a C file that LTO sees.
- **B0 needs no minted contexts** (CONTEXT_BYTES = 0). The context entry/exit/seal/offer/call exports become stubs
  that fault deterministically if reached.
- **The gp-captable yield already exists, so B0.2 is a port, not a new glue.** The c128 line
  (`origin/rebased/c128-3-musl`, plan `20-08-2026_mruby-gp-captable.md`) put `__capstone_yield` into the interp glue:
  - under `CAPSTONE_GLUE_YIELD`, landed on dev as d67a82a0e3a4;
  - it reaches its frame through `cscratch` and the entry return capability at `+48`, the slot `test:` already
    parks;
  - byte-identity controls keep SQLite, micropython and jerryscript untouched;
  - its yield-probe passed on QEMU on the gp-captable glue: resume, not restart, with frame and cap table intact.
- **The c128 build path to port:**
  - `build-yield-probe.sh YIELD_PROBE_GPCT=1` and `build-mruby-probe.sh MRUBY_GPCT=1` on that branch;
  - the musl archive built with `MUSL_CAPSTONE_EXTRA_CFLAGS`, with its compile census (1321/1361, as the default
    ABI);
  - full LTO, its step 4b;
  - stack_reserve.o in BOTH links (a new global shifts the descriptor).
- **What B0 adds is the CURRENT runtime.** c128 ran the old HostCall v0 runtime. Today's runtime is `hostcall.c`
  domain_main with 3 region shares, delegate.c over the process ABI, and the recovery-block layout in context.h.
  - The interp glue's frame and result-slot convention differs from start-musl's recovery block (slots 0/16/32/48/64/
    80). B0.2 reconciles the two for exactly what the runtime reads: the result slot on a fault, and the yield.
- **No plain M-CSR access.** The interp glue's mcause/mtval saves (:719/:721/:942/:944) are illegal under supervision.
- **The init/fini and TLS template bounds come from the glue** (as values computed from gp-relative definitions it
  owns), not from linker-script symbols. The runtime C uses them through two small accessors.
- **The S-17 shape** (an LDC right after `ccsrrw sp, cscratch`) is avoided at `.Lyield_resume`, even though S-17 did
  not reproduce on 715bdd1fe (3/3).
- **Gate:** `sup-static-audit.py` clean; the delin census.

**B0.3 Linker script.** `link-gpfree.ld` plus `.capstone_application` (an indexable global), `.capstone_domreq` and
TLS (`.tdata/.tbss`, PT_TLS). Two-pass link sized by `domdata-budget.py`.
- **Gate:** `application-image.c` parses the image. `gp-initdesc-blocks.py` reports one block.

**B0.4 Runtime C.**
- `hostcall.c` and `tls.c` use the glue accessors.
- `exec.c` passes `ticks_per_second = 0` when the CPU is `"eth, ariane"`.
- **Gate:** the existing QEMU delegated suites are unchanged (the accessors are the same values on the legacy ABI).

**B0.5 Build: full LTO, one module.** The app, the runtime and the musl members it needs, with the gp-captable
and shrink flags re-passed to the LTO plugin. Port the c128 per-member verifier.
- **Gates:**
  - one initdesc block;
  - slots globally unique (a positive control in the image: two volatile noinline readers in different source
    files);
  - no `fork.o`;
  - no sub-word atomics on self-bounded objects.

**B0.6 QEMU, silicon-mimicking.**
- Run with `CAPSTONE_GP_FABRICATE=0` and `CAPSTONE_MOVC_NULL_SCALAR=1` under the patched QEMU-pinned monitor.
- Pass: the hello line, and a gp-fabrication count of 0.
- The count is shown to fire: the same image with fabrication ON counts at least 1, and a legacy SDK image counts
  > 0.

**B0.7 Board.**
- Bake capstone-exec, the process-ABI modcapstone and the image.
- Use the FPGA monitor from B0.1.
- Control first (k800), the hello-world last. Pre-registered.
- This is also the first execution of the process ABI on silicon. If it fails, a classic-path control of the same
  image separates "the ABI" from "the image".

## Toolchain (2026-10-04): the shared build is STALE for B0; use the frozen dev147 build
- **The shared build is stale.** `llvm/cmake-build-debug`'s libLLVMCapstoneCodeGen.so dates from 2026-09-24. dev has
  gp-captable backend fixes since then, among them C-75: e2a1010c6891 and 214373ee2add.
- **B0 uses the compiler lane's frozen copy:**
  - built from 612b3ec514c0, with 0 files differing under llvm/ and clang/ from dev dedab28c4a2c;
  - lib manifest 18c0d1b392dff5d9;
  - read-only, at `<compiler lane scratch>/dev147-toolchain/bin`.
  `clang --version` says 612b3ec5, a configure-time string.
- **Identified by behaviour, two-sided** (the compiler lane's probes, run here):

  | probe | shared build | dev147 |
  |---|---|---|
  | B: gp-captable function alias, `pcrel_hi(fn_alias)` | 0 | 1 |
  | A: default-ABI alias, `stc` in cap-init | 1 | 2 |
  | A's control | 2 | 2 |

**B0.5, first piece: the gp-captable LTO musl archive.**
- `MUSL_CAPSTONE_EXTRA_CFLAGS`, appended to the survey's flags, carries -flto, the four silicon -mllvm options and
  -DCAPSTONE_GP_CAPTABLE_ABI=1. When unset it is a strict no-op: the flag lists are identical with the hook absent and
  unset. `MUSL_SURVEY_JOBS` caps the parallelism.
- With dev147: 1355 of 1361 sources compile, all bitcode. The 6 failures are mallocng (a static `sizeof(void*)`
  assert); the runtime replaces malloc with level0/sublet.

## B0.2-B0.5 status (2026-10-04): `b0-hello.dom` builds and passes every static gate
Build: `capstone/runtime/silicon/build-b0-hello.sh`, with CAPSTONE_LLVM_BIN set to dev147 and B0_MUSL_ARCHIVE set to
the gp-captable LTO archive. Result: sha256 2b2224da711c89c0, .text 38,916 B, globals at 0x10000.

**What it took, each found by a gate firing.**
1. **lld keeps every bitcode LIBCALL definition** (`<internal>: reference to acosl`), and the fp128 family cannot
   be selected under gp-captable (C-43).
   - Fix: a per-member codegen verifier, ported from c128 into build-musl-capstone.sh. It runs llc with all four
     -mllvm options and has a self-test that refuses a result in which no bitcode was recognised.
   - It drops 41 musl members, acosl among them. The archive keeps 1314.
   - The same verifier (`runtime/silicon/lto-codegen-verify.py`) drops the 4 fp128 soft-float builtins.
2. **`__cp_begin/__cp_end/__cp_cancel` were start-musl.S assembly data.**
   - They are now C definitions in `glue-data.c`, in name-sorted sections, emitted SORTed by link-gpfree-app.ld.
   - The image has begin < end < cancel (0x205f0/1/2), as musl's handler needs.
3. **2 `cjalr` in altstack.S's `__capstone_call_on_stack`.** Under gp-captable a function pointer and ra are
   integers, and compiled code uses jalr/ret and `sd ra`.
   - Fix: a `CAPSTONE_GP_CAPTABLE_ABI` form. The default ABI's .text is byte-identical.
4. **9 gp-derived `scc rX, gp, rY; delin` sites** (C-13), all linker-script symbols:
   - tls.c (inlined into domain_main);
   - hostcall.c's init array;
   - hostcall.c's `__libc_exit_fini`.
   - Fix: `glue-accessors.S` returns the extents as integers (`lla` differences). tls.c and hostcall.c use the
     accessors under `CAPSTONE_GP_CAPTABLE_ABI` and refuse a non-empty .tdata or init/fini array, which B0 cannot
     read yet.
   - In this image .tdata is empty, .tbss is 0xe4 B, and both arrays are empty.
   - Default-ABI disassembly of tls.c and hostcall.c is identical; the define changes it (control).
5. **4 mcause/mtval words in the interp glue**, illegal under supervision.
   - Fix: a new opt-in `CAPSTONE_GLUE_NO_MCSR`.
   - The glue disassembles identically in the default, YIELD and INTERP_DOMAIN_MTVEC configurations against HEAD.

**Gates on the image:**

| gate | reading |
|---|---|
| `cjalr` | 0 |
| gp-derived `scc` | 0 (was 9) |
| `delin` | 1 (the glue's `delin sp`) |
| descriptor blocks | one, 66 globals |
| supervision audit | 0 forbidden of 10,094 instructions |
| fork.o, dummy_lockptr | absent |

**Not yet:**
- **Jump tables:** the image has 14 `jr`. They are not yet classified.
- **context.c:38:** in the module but not reached. Its fix lands when contexts are needed.
- **B0.6, the QEMU run with CAPSTONE_GP_FABRICATE=0.**

## B0.6 PASSED in QEMU (2026-10-04): the delegated hello-world runs with NO fabricated gp
`capstone-vm run` printed `B0: hello from a gp-captable delegated application`, and the result was exit 0.

**The image:** b0-hello.dom, sha256 777ec140369fb0f6, b0-silicon-runtime 8e3ecc0565f3.

**The platform** (private, frozen copies, in /tmp/capstone/b0/platform):
- QEMU 4940d3fde12b3efa: the compiler lane's idle qemu-dpr, src 440b922c90. Its strings carry GP_FABRICATE,
  MOVC_NULL_SCALAR and GP_NONLIN.
- Firmware faed82b38f33d052: wrapper 3514060, monitor capstone-sbi monitor/b0-managed-gp 3be6737,
  `-DCAPSTONE_TARGET_QEMU -DCAPSTONE_DEBUG_ENABLE -DCAPSTONE_SUPERVISED_CALL`. Built with mon-c0/build-fw.sh.
- Kernel and rootfs: the shared buildroot QEMU Image (e58613598c897103) and rootfs.ext2 (9903242c37a72425), the
  rootfs under -snapshot.
- launcher 0752aa7c, job helper 5a160efa and module a88ed215, copied from the memcached lane's
  pinned-platform/run-level0 (2026-10-03). ssh_server 385fcf91. The shared rootfs has no capstone-exec or dropbear.
- The QEMU process's own /proc environ: `CAPSTONE_GP_FABRICATE=0 CAPSTONE_MOVC_NULL_SCALAR=1 CAPSTONE_GP_NONLIN=1
  CAPSTONE_REV_NODES=65536`. MOVC_NULL_SCALAR's one-shot report went to the monitor (pc 0x80021d80, priv 3), as
  warned.

**One fault before the pass.** cause 24 at `lcc` on `domain_main`'s address. HELLO read the capability bounds of a
function pointer, which is an integer under gp-captable. It now reports the code range through two more accessors.

**Controls, each firing as predicted** (/tmp/capstone/b0/vm-controls.sh):

| arm | firmware | gp fabrication | b0-hello | legacy SDK image (ffapp_fx24, 61b707bd) |
|---|---|---|---|---|
| the run | B0.1 | OFF | PASS | faults at once (cause 0, pc 0, before HELLO) |
| A | B0.1 | ON | PASS | runs (exit 91, its FFAPP output, as on its owners' platform) |
| B | without B0.1 (78151e4, fw 71965bfc) | OFF | REFUSED ("cannot create domain") | - |

- **The legacy image runs or fails on fabrication alone.** B0 passing with fabrication OFF therefore means B0
  does not depend on it.
- **Arm B** shows that the B0.1 monitor change is what admits a managed image with a globals region.

## B0.7 preparation (2026-10-04)
**The board image had none of the process ABI.**
- The FPGA buildroot (d04bd83) has no capstone-exec in its target.
- Its capstone.ko has no process-ABI symbols: 0 `process` strings, against the QEMU module's `process_cache_bytes`.
- The FPGA kernel is 6.4.14. The QEMU guest's is 6.1.

**Built for the board** (the build trees are private, in /tmp/capstone/b0):
- **Module (sha 57cb9a9b6821e979):** the process-ABI modcapstone (the memcached lane's pinned package source), built
  against build-fpga's linux-6.4.14 with the FPGA toolchain. vermagic is `6.4.14 SMP riscv`.
  - It needed one port: Linux 6.3 made `vma->vm_flags` read-only, so `process.c` now uses `vm_flags_set()` behind
    a `LINUX_VERSION_CODE` guard. That change belongs in the modcapstone package's process-ABI branch.
- **Launcher:** capstone-exec and capstone-job from `capstone/runtime/exec`, cross-built with the FPGA buildroot's
  toolchain (`linux-guest.cmake`). They carry the time-CSR change below.
- **`exec.c`:** the launcher reports `ticks_per_second = 0` when the device-tree CPU compatible list has the exact
  entry `eth, ariane`, i.e. the Capstone FPGA core, which has no `time` CSR. The libc then delegates every clock
  read. The matcher was unit-tested on 6 cases, a missing file included.
- **`runtime/silicon/b0run.sh`:** the board wrapper for the baked-rung driver.
  - It runs the k800 classic control first, with the stock module.
  - Then: rmmod/insmod of the process-ABI module, `capstone-exec --stats`, then the image.
  - Distinct retvals: 901 module swap, 902 --stats, 903 no hello line, 904 non-zero exit.

**insmod works on the board.** The stages driver loads /capstone.ko with `[ -c /dev/capstone ] || insmod` on every
boot. The "insmod of ANY module hangs this board" comment in run_rtl_smoke.py is stale.

## B0.7 pre-registered (2026-10-04 21:12, before the boot)
**The boot.** One boot on `caplifive_supcall_715bdd1fe.bit`, driven by `run_baked_rungs_fpga.py` with
`BAKED_CTL=/test-domains/b0run.sh` and `BAKED_RUNGS="k800r b0-hello"`. The oracles are k800r 4 and b0-hello 0.

**Firmware abb829785a79.**
- Monitor capstone-sbi monitor/b0-managed-gp 3be6737, wrapper 882892f,
  `-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS`, no pre-CALL fence (S-16 is fixed on this bitstream).
- dom_stack gate PASS.

**Image 7b0c02fd6f2e.** The shared FPGA initramfs plus six files, each verified by hash in the cpio:
- b0-hello.dom (777ec140);
- the process-ABI module built for 6.4.14 (57cb9a9b);
- b0run.sh;
- capstone-exec and capstone-job, which carry the eth,ariane time change;
- the k800 relinked at 0x20000 (589ceee3), as k800r.dom. b0-hello enters at 0x10000, which would be an R-3/C15 collision
  with the stock k800.
The shared image was restored and verified afterwards.

**PREFLIGHT=0, for C5f's reason.** The preflight inspects the SHARED overlay, which is stock, not this private
payload. Its BLOCKs were "b0-hello not staged", "unused files", "1 distinct image" and C15. C15 is real and is fixed
by the relinked control. k800r's QEMU-pass record is ~/capstone-artifacts/k800-relinked-0x20000/orc.

**Predictions.**
- k800r: `RESULT k800r retval=4`.
- **b0-hello: retval 0.** "B0: module swapped", then `capstone-exec --stats` answers, then the hello line, then
  capstone-exec rc 0.
- This is the FIRST execution of the process ABI on silicon. A failure is read by its code:
  - 901: the module swap failed;
  - 902: the monitor does not answer the process ABI;
  - 903: no hello line; capstone-exec's own fault line (cause, pc, entry) is printed above it;
  - 904: the line printed, but the exit status was non-zero;
  - no RESULT at all: a hang.
- **The known risks, so a failure can be placed:**
  - the supervised context_step under CSR events has never run;
  - the yield's `.Lyield_resume` is the S-17 shape (`ccsrrw sp <- cscratch` then `ldc`). S-17 did not reproduce on
    715bdd1fe, 3 of 3.
  - a domain `rdtime`, if the time change did not take.

## B0.7 attempt 1 (2026-10-04 21:20-21:24): the control passed; b0-hello = 901, the module swap
- **k800r: RESULT retval=4.** The control is OK.
- **b0-hello: `rmmod: can't unload module 'capstone': Function not implemented`, then RESULT 901.**
- **Why.** The board's kernel has no module unload: vermagic is `6.4.14 SMP riscv`, without the QEMU kernel's
  `mod_unload`. Once the control had loaded the stock module, nothing could replace it.
- **Attempt 2: the wrapper loads the process-ABI module FIRST, and that module serves both rungs.**
  - The classic control (lpc + k800r) now also checks the new module's classic path. A failing control says "the
    module"; a failing b0-hello says "the process ABI".
  - Predictions as before: k800r 4, b0-hello 0.
  - A new failure point: **k800r 901 or a wrong value means the process-ABI module breaks the classic path on the
    board.**

## B0.7 attempt 2 (21:26-21:29): both rungs 901 -- the DRIVER loads the stock module first
- Every board driver runs `[ -e /dev/capstone ] || insmod /capstone.ko` after boot and before any rung
  (run_ladder_perf_fpga.py:193-202). The stock module was therefore in place before the wrapper ran, and it cannot
  be unloaded. Attempt 1 had the same cause.
- **Attempt 3:** in the private image, `/capstone.ko` IS the process-ABI module, so the driver's own insmod loads it.
  - image 705722685b549e93: the cpio check shows /capstone.ko = the new module.
  - The shared target's stock module (ed807a292aaad3f6) was backed up and restored by the bake's EXIT trap, and the
    restored shared cpio carries it again (hash checked).
- Predictions unchanged: k800r 4 (the classic path on the process-ABI module), b0-hello 0.

## B0.7 attempt 3 (21:33-21:40): the module's own insmod hangs the board -- a build-flag defect, already on record
- **What the board did.** The driver's `insmod /capstone.ko` never returned. The monitor printed
  `EXCX:0000E002 MCAU:00000004 MEPC:80006B18 MTVL:00F0006E MSTA:00000822`.
  - That is a misaligned load (cause 4), taken in S-mode (MPP=1).
  - The faulting pc is in the FPGA vmlinux's `apply_r_riscv_call_plt_rela`, at `ffffffff80006b18: 4090 lw a2,0(s1)`.
    The kernel's module loader reads the auipc+jalr pair at a relocation site with 32-bit loads.
  - The capstone monitor's `handle_exception` has no cause-4 case, so it prints EXCX and spins. The kernel never gets
    the trap back.
  - Neither rung ran, and there is no verdict about the process ABI.
- **Why.** My module was built out of tree with `make -C linux M=...`, and that drops the package-level
  `EXTRA_CFLAGS`. On the FPGA target, `external.mk` sets `-march=rv64g -mabi=lp64d` for exactly this failure: upstream
  cfa3d49, "Disabled C extension for modcapstone build to avoid misaligned access". Without it the kernel's own
  `-march=rv64imac` applied.
- **Measured** (python over `readelf -rW` and `objdump -d`; the old build is the positive control):

  | build | 2-byte instructions | R_RISCV_CALL* in .text | at a 2-mod-4 offset |
  |---|---|---|---|
  | attempt-3 module 57cb9a9b (rv64imac) | 1602 | 161 | **71** |
  | rebuilt module 0332e2d6 (`EXTRA_CFLAGS="-march=rv64g -mabi=lp64d"`) | 0 | 161 | 0 |
  | stock buildroot module ed807a29 | 0 | 87 | 0 |

  The rebuilt module still carries the process ABI: `process_cache_bytes` is present, and the `vm_flags_set` guard is
  in process.c.
- **A side note on the external.mk comment** ("the FPGA core has no C extension"): it does not match this boot. The
  faulting instruction is itself a 2-byte `c.lw` that the core was executing (the kernel is built rv64imac). The
  defect is the misaligned DATA load that RVC relocation sites cause, not instruction fetch. The flags are right
  either way.

## B0.7 attempt 4 pre-registered (before the boot)
- Same boot, bitstream, firmware recipe, driver command and oracles as attempt 3. The only change is the module in the
  private image: 0332e2d6 (rv64g) replaces 57cb9a9b, as both `/capstone.ko` and `/test-domains/capstone-proc.ko`.
  - Image 96811d0769d064d8. Its cpio check shows all six staged files and the replaced `/capstone.ko` present by hash.
  - The shared image was restored afterwards: its target `/capstone.ko` is ed807a29 again, and the six files are
    absent from the restored cpio.
  - Firmware a63f82b0dfe5: monitor 3be6737, wrapper 882892f,
    `-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS`. The dom_stack gate passes (21552 of 32768 B).
- **Predictions.** The driver's insmod returns, and `/dev/capstone` appears.
  - **k800r: retval 4.** This is the classic path under the process-ABI module. If it fails, the module breaks the
    classic path.
  - **b0-hello: retval 0**, with the codes as pre-registered at 21:12.
- **A new failure point.** If the insmod hangs again, read the EXCX block.
  - Cause 4 again would mean a misaligned access the flags did not remove: data, not RVC relocation sites.
  - Any other cause is a new finding.

## B0.7 attempt 4 (21:55-21:59): the module loads; both rungs fail on one cause -- the control used the legacy API
- **The module loads.** The driver's `insmod /capstone.ko` returned and `/dev/capstone` appeared (DEVOK), with no
  EXCX. The rv64g rebuild fixed attempt 3's hang.
- **k800r: `ladder-perf: create_dom failed`, with no RESULT.** lpc's `struct ioctl_dom_create_args` is 11 words. The
  capstone-bootstrap module's struct added `copy_len`, making it 12. The struct size is part of the ioctl number,
  so lpc's DOM_CREATE is not recognised. lpc's own comment (ladder_perf_ctl.c, at the struct) predicts exactly this.
- **b0-hello: `capstone-exec: device: Device or resource busy`, then 902.** `device_ioctl_locked` sets
  `legacy_api_selected` on any non-PROCESS_ENABLE ioctl, before its switch, so lpc's unrecognised one counts too.
  After that, PROCESS_ENABLE returns EBUSY for the life of the module, and the board cannot unload it.
- **So 902 here carries no verdict about the process ABI or the monitor.** It is collateral of the control.
- **Design consequence.** A classic control and a process-ABI image cannot share a module load, and on this board a
  module load is a boot.

## B0.7 attempt 5 pre-registered (before the boot)
- **Rungs, in order:**
  1. `b0-stats`: `capstone-exec --stats` alone. The monitor answers the process ABI's census, so 0 is expected.
  2. `b0-hello`: the unknown, last of the informative ones. 0 is expected, with the codes as pre-registered at 21:12.
  3. `b0-stats2`: the census again. It runs only if b0-hello returned. 0 is expected, and its counts are reported,
     not predicted.
- **This boot has no classic control.** That is a deviation from the board-run rule, recorded deliberately: the
  classic path cannot run on a process-ABI module load (attempt 4).
  - Board and boot health come from the shell login and from DEVOK after the driver's insmod.
  - The k800r classic control passed on this monitor (3be6737) with the stock module in attempt 1.
- **Changes from attempt 4:**
  - b0run.sh gains the b0-stats cases. A stub test prints RESULT 0 for a stats rc of 0, and 902 for rc 3.
  - k800r.dom leaves the manifest.
  - Image ba6facd1d096372e. Its cpio shows 5 files plus the replaced `/capstone.ko` (0332e2d6). The shared image
    was restored (`/capstone.ko` is ed807a29, the 5 files are absent).
  - Firmware 343ad544162a: monitor 3be6737, wrapper 882892f, the same defines. dom_stack gate PASS.
- **Reading b0-stats:**
  - 902 means the monitor does not answer the process ABI's census on silicon, and b0-hello will 902 as well;
  - 901 means the module did not come up with the process ABI.

## B0.7 attempt 5 (22:04-22:10): the census answers; b0-hello's first step RETURNS, then the board goes silent
- **b0-stats: RESULT 0.** `capstone-exec --stats` answered with all counters 0. **The monitor answers the process
  ABI on silicon.**
- **b0-hello: no RESULT within 300 s.** The trace, in order:
  - `DBAS:AC080000 DENT:0`: the domain was created.
  - Three region shares: RGID 0x0D (0x8000 bytes), 0x0F and 0x10 (0x10000 each). Each ran SHA0..SHA5, then
    `SUPA 0` (armed), `SUPK 0` (returned), then `ECSZ 0`. The managed share path enters the domain through
    supervised_invoke, so **the gp-captable domain was entered and returned three times on silicon.**
  - A lone `SUPA 0 / SUPK 0`. RESUME_SHARE runs only after a preemption, so this is the first PROCESS_STEP
    (context_step): **the application ran and came back with kind 0, a yield or its exit.**
  - Then nothing for 300 s: no EXCX, and no next SUPA.
- **Where the hang can be.** capstone-exec serves the yielded request and steps again, which must print SUPA. So the
  hang lies between that SUPK and the next SUPA, in one of:
  - context_step's loan_end, in M-mode;
  - the return to Linux;
  - capstone-exec's serving;
  - the next step's nodes_reserve, loan_begin or arm.
- **The console cannot separate these:** b0run.sh sent capstone-exec's stdout and stderr to files.
- b0-stats2 is collateral. The driver's "ENTRY STALL (R-16)" label keys on the ladder's SHA6 and does not apply to
  this path.

## B0.7 attempt 6 pre-registered (before the boot): instruments that separate the hypotheses
- **Monitor ee1dd50:** 3be6737 plus two traces, STPB at context_step entry and STPE after loan_end. The generated
  assembly carries each constant once (SUPK, the positive control, twice).
- **b0run.sh b0-hello:**
  - a bounded heartbeat, `B0: hb N` every 5 s;
  - capstone-exec run with `CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_DELEGATE_STATS=1`, its stderr live on the console;
  - a 90 s watchdog: SIGTERM, then SIGKILL, and **retval 905**.
  - A busybox-sh stub test scored pass 0, hang 905, no line 903 and bad rc 904, and left no process behind.
- **Build:** Image 88ad30cafd090956, firmware d58595da63a8. Same rungs, oracles and driver command as attempt 5.
- **Predicted readings, written before the boot:**
  - **H1, loan_end wedges in M-mode:** `STPB SUPA0 SUPK0`, no STPE; heartbeats stop.
  - **H2, capstone-exec stuck with Linux alive:** STPE printed, heartbeats continue, and the watchdog gives 905 plus
    capstone-exec's stderr.
  - **H3, the kernel hangs after the step ecall:** STPE printed; heartbeats stop.
  - **H4, the next step's entry wedges:** STPE, a second STPB, no SUPA.
  - **H5:** b0-hello returns 0. That would make attempt 5's hang timing-dependent, since the traces add UART time.
- Every reading is informative, so this boot is not spent confirming what is already known.

## B0.7 attempt 6 (22:17-22:24): H1 -- the core wedges in M-mode inside loan_end
- b0-stats 0 again.
- **b0-hello:** the same three shares, each `SUPA 0 / SUPK 0`, then `STPB 0, SUPA 0, SUPK 0`, and nothing after:
  **no STPE, and not one `B0: hb` line** in 300 s.
  - The step returned kind 0, and context_step never reached the trace after loan_end. The only work between SUPK
    and STPE is `result = loan_end(k)` and four stores to the trap frame.
  - Linux never ran again: no heartbeat, with the first due 5 s after the launch began.
  - This is the pre-registered H1 reading.
- **loan_end and managed_reclaim have never run on silicon before.** C5's supervised runs drove classic domains,
  which are UNMANAGED slots, and those never take context_step's loan path. The managed shares do not loan.

## B0.7 attempt 7 pre-registered (before the boot): bisect loan_end, and read the wedged core
- **Monitor b6d74c3:** ee1dd50 plus stage traces.
  - LNDE 1: the two descriptor loads done.
  - LNDE 2: the `ldc` of the offered seal at offset 32.
  - LNDE 3: the `stc x0` that clears it.
  - LNDE 4: the offer bookkeeping.
  - Inside managed_reclaim: MRCL 1, the `__revoke` done; MRCL 2, the stc loop and C_INIT done.
  - LNDE 5 and LNDE 6: managed_reclaim returned, desc_put done.
  - The generated assembly carries LNDE 6 times and MRCL twice.
- **Build:** firmware 596924b260b3 on the same Image 88ad30cafd090956.
- **Driver:** `run_baked_rungs_fpga.py` gains `BAKED_WEDGE_APERTURES=1`. After a wedged rung, it reads the bare
  wedge harness's apertures before release: trap log, 224..229, 192..195, 219..222, commit pc and trap mepc.
  - A fake-console test assembled pc and mepc correctly and parked the switches at 0.
  - Off by default.
- **Predicted readings.** The last stage printed names the statement:
  - no LNDE 1: the plain loads through `view` of memory the domain wrote;
  - LNDE 1 only: the `ldc`;
  - LNDE 2: the `stc`;
  - LNDE 3: cap_type or offer_valid;
  - LNDE 4: the revoke;
  - MRCL 1: the reinitialising loop;
  - MRCL 2 or later: desc_put or the return.
- The apertures say whether the LSU is stuck: lsu_rdy (224), the load and store states (194/195), the bypass head
  (219/220), the commit queue (221) and the tag unit (222).
- **Current lean, low confidence:** the revoke or the first access to the loaned block. The loan's sequence (a
  `__mrev`'d block, delinearised and lent, then revoked) has no silicon precedent. The traces decide it, not this
  lean.

## B0.7 attempt 7 (22:34-22:43): the wedge follows loan_end's offer stage; a nested trap in the monitor's trap entry
- **Trace:** `STPB 0, SUPA 0, SUPK 0, LNDE 1, 2, 3, 4`, then nothing. MRCL 1 never printed.
  - The descriptor loads, the seal's `ldc` and `stc x0`, and the offer bookkeeping all completed.
  - The wedge lies in loan_end's call into managed_reclaim, its prologue `stc`s, the `ldc` of the rev capability,
    `revoke`, `movc`, or the `stc` before the trace call.
- **Apertures**, all stable on a double read:
  - trap log 0x98: seen, mcause 24 (UNEXP_OP_TYPE), mepc 0x80020064 = `_cap_trap_entry`+4.
  - commit pc 0x8002007c = `_cap_trap_entry`+0x1c, a SAVE_REG store.
  - 224 = 0x0d: lsu_rdy 0.
  - commit queue FULL: 221 = 0xf8 (valid 1111, st_data_req 1, gnt 0), 193 count 4, 194 store_state 3.
  - bypass head a STORE (220 = 0x82, 219 = 0x26). Tag FSM IDLE (222 = 0). 228 = 0xc2.
- **Reading.** cva6.sv at 715bdd1fe latches the LATEST non-interrupt, non-illegal trap, ecalls included.
  - `_cap_trap_entry` starts `ccsrrw sp <- cscratch`. Inside the monitor, cscratch holds Linux's integer sp. So any
    trap taken IN M-mode makes the next instruction, `cincoffsetimm sp`, fault with cause 24 (+4).
  - The second entry swaps the monitor's sp back and starts saving registers. Its stores then never drain: the
    commit queue is full and the store port is not granted.
  - **So the original fault, somewhere after LNDE 4, is a trap inside the monitor whose cause and pc the nested
    trap overwrote.** The final LSU wedge may be a consequence rather than the cause. UNRESOLVED.
- Side finding: a trap raised by the monitor's own code cannot be reported on silicon. It always becomes this
  nested cause-24 trap.

## B0.7 attempt 8 pre-registered (before the boot)
- **Monitor 8a5004e:** b6d74c3 plus `MRCL 0xf` at managed_reclaim's entry (prologue done) and
  `MRCL 0x10 + cap_type(root)` right before the revoke. The generated code is prologue, trace, `ldc root`,
  `lcc type`, trace, `ldc`, `revoke`, `movc`, `stc`, trace (MRCL 1).
- **Build:** firmware 104f5798c558 on the same Image 88ad30cafd09. dom_stack gate PASS.
- **Driver:** the wedge read adds the latched tval (210/211/213..218) and the rev-node state:
  rev_node_debug_ex (241..248), head (249/250), serving index (251..254). A fake-console test assembles each.
- **Predicted readings:**
  - no MRCL 0xf: the call or the prologue's stores fault;
  - 0xf but no 0x1t: root reloads as a NON-capability, so `lcc` faults;
  - 0x1t but no MRCL 1: `revoke` (or `movc`/`stc` behind it) faults or blocks on a capability of type t;
  - MRCL 1 printed: attempt 7's wedge did not reproduce with these traces.
- **The tval of the nested trap** should read Linux's sp, a kernel stack address, if the nested-trap reading is
  right. Anything else refutes it.

## B0.7 attempt 8 (22:48-22:57): the value handed to REVOKE is NOT a capability
- **Trace:** `... LNDE 4, MRCL 0xF, MRCL 0x17`, then nothing.
- **Correction to the pre-registration.** It said an lcc on an integer faults. On silicon the TYPE query (selector
  1) is TOTAL by design, the "S-06 enabler" (capstone_dyn_unit.anvil:212-237 at 715bdd1fe). It returns
  `cap_type - 1` in 3 bits (:249), so NOT_CAP (0) reads 7. **So 0x17 is the "root reloaded as a non-capability"
  reading**, and REVOKE on plain data then raises.
- **tval of the nested trap: 0xffffffc80412bca0**, a Linux kernel stack address, as predicted for
  `cincoffsetimm sp` on Linux's integer sp. The nested-trap reading holds.
- **Rev-node state:** rev_node_debug_ex 0, head 0x009e, serving index 0. All other apertures match attempt 7.
- **Not Q-11.** On silicon, LDC forwards a loaded capability verbatim (ISSUES Q-11: a revoked handle reads 2 there).
  So silicon's 7 means the stored value had no tag. It is not a revoked rev capability.
- **Not S-07.** That defect is sporadic. This one repeated identically in attempts 5-8.

## B0.7 attempt 9 pre-registered (before the boot): which hop loses the tag
- **Monitor 5c984b2:** 8a5004e plus type traces in post-shift numbering (LIN 0, NONLIN 1, REV 2, UNINIT 3,
  NOT_CAP 7).
  - In loan_begin: `LNBG 0x10+` the block desc_take returns; `0x20+` what MREV returns; `0x30+` the block after
    MREV; `0x40+` desc_rev[k] reloaded right after the store; `0x50+` the DELINed view.
  - In loan_end, before the call: `LNDE 0x40+` desc_rev[k].
- **Build:** firmware 43018fb5871e on Image 88ad30cafd09.
- **A healthy sequence reads** LNBG 0x10, 0x22, 0x30, 0x42, 0x51, then LNDE 0x42 and MRCL 0x12. The first 7 names
  the hop:
  - LNBG 0x27: MREV returned a non-capability;
  - 0x47: the stc into the global array lost the tag;
  - LNDE 0x47 after LNBG 0x42: the tag disappeared while the domain ran;
  - MRCL 0x17 after LNDE 0x42: the stack round trip lost it.
- Also informative: a 0x20 + t with t not 2, i.e. MREV returning some other type.

## B0.7 attempt 9 (23:01-23:10): tagged after the store, untagged at loan_end -- and the cause, from the codegen
- **Trace:** `LNBG 0x10, 0x22, 0x30, 0x42, 0x51`. The block is LIN, MREV returned a REV, the block stays LIN, the
  global reloaded REV right after the store, and the view is NONLIN. Then the step, LNDE 1..4, **`LNDE 0x47`**,
  MRCL 0xF and 0x17.
- **Caveat on attempt 9 itself.** On silicon, LDC moves a linear-family value out of its slot (load_unit.sv:214-218;
  ISSUES Q-12). The 0x40 trace's `ldc` of desc_rev[k] was spilled, not written back, so in that boot the trace
  emptied the global. LNDE 0x47 there is self-inflicted. Attempts 5-8 had no such reader.
- **The cause, in the generated code of the ORIGINAL monitor (3be6737):** `ldc(s1, desc_rev[k])`,
  `movc(a0, s1)`, **`stc(a0, sp, 112)`**, `call managed_reclaim`.
  - capstone-c treats `__rev` as copyable, so it caller-saves the live argument with `stc` before the call.
  - On silicon, STC writes cnull back to its register source for every type but NONLIN
    (capstone_dyn_unit.anvil:544-548 at 715bdd1fe, read in source). capstone-qemu's `trans_csstc` copies (Q-12).
  - So managed_reclaim received NOT_CAP. All four managed_reclaim call sites had this shape: loan_end, domain
    cache reuse, domain destroy, region reset.
- The rtl-oracle confirmed, with quotes, that domain CALL/RETURN/SAVE/RESTORE never touch rev nodes, and that MREV
  cannot write an untagged rd. That rules out the other hops.
- This is a NEW instance of the KNOWN divergence Q-12, through a compiler that predates silicon's linear STC. It is
  the same class as the LLVM RegAllocFast spill in history/13-08-2026 (STC with empty `(outs)`).

## B0.7 attempt 10 pre-registered (before the boot): the fix
- **Monitor d5459e1:** each caller does `x = __revoke(slot); x = managed_reinit(x)`, so no `__rev` value crosses a
  call.
  - The generated code is `ldc` of the global, `revoke`, then the LINEAR local, then the call; nothing saves a0.
  - A scan for "argument register passed exactly as stc-saved" finds the 4 old sites in 3be6737 (the positive
    control) and none in d5459e1. The other saved-argument hits are integers, NONLIN values, or temporaries that
    are not the callee's arguments.
  - The trace reloading desc_rev[k] in loan_begin is removed.
  - Firmware 228a0194e64f on Image 88ad30cafd09, byte-identical to the compile that was inspected.
- **Predicted reading:**
  - b0-stats 0.
  - b0-hello: each step reads `STPB, LNBG..., SUPA 0, SUPK 0, LNDE 1..4, MRCL 0x10` (silicon returns LINEAR after
    DELIN, per the oracle; 0x13 would mean UNINIT), then `MRCL 2, LNDE 5, 6, STPE 0`.
  - capstone-exec serves the request and steps again, through to the exit. The hello line appears, then
    **RESULT b0-hello 0**.
  - b0-stats2 0, with live_domains 0 after the exit if destroy works. It now also exercises the domain-destroy
    revoke.
- **A failure is read the same way as before.** The wedge apertures are on, and the last trace names the stage.

## B0.7 attempt 10 (23:12-23:21): THE FIX WORKS -- steps complete; a new livelock follows
- **Every step now runs to the end:** `STPB, LNBG 0x10/0x22/0x30/0x51, SUPA 0, SUPK 0, LNDE 1..4, MRCL 0x10,
  MRCL 2, LNDE 5, 6, STPE 0`.
  - **MRCL 0x10:** silicon's REVOKE returns LINEAR for the DELINed block, as the oracle predicted from
    rev_node.anvil:71-73 and dyn_unit.anvil:92-95. The UNINIT reinitialisation does not run there. (QEMU returns
    UNINIT; both end LINEAR.)
- **But b0-hello did not finish in 300 s.** There were 6,638 STPB and 6,636 STPE (counted over this boot's UART
  joined and scoped after its own `load_image`; the driver's per-rung transcript holds fewer, 6,205/6,204), every SUPK 0 (returned), about 22
  steps per second, and the core was not wedged.
- **No `B0: hb` and no watchdog line came through.** The monitor's ~1.6 MB of step traces at 57,600 baud starved and
  garbled Linux's console (binary fragments between trace lines), so the console says nothing about Linux.
- **QEMU regression of monitor d5459e1** (fw d7d769b1, /tmp/capstone/b0/vm-fix.sh):
  - b0-hello passes with fabrication OFF and ON;
  - the legacy SDK image exits 91, as on its owners' platform.

## B0.7 attempt 11 pre-registered (before the boot): read the request stream
- **Monitor 1855cb4:** d5459e1 with the step traces opt-in (`CAPSTONE_LOAN_TRACE`, not set), built with
  `-DCAPSTONE_SUPERVISE_QUIET`. A step prints nothing; SUPA and SUPK print only on a refused arm or a missing event.
  Firmware 12cd4c35aea7.
- **capstone-exec 35c045ff:** `CAPSTONE_DELEGATE_TRACE=N` prints the first N requests, then every 256th: round,
  number, the first three arguments and the result. b0run.sh sets N = 64.
- **Image 14a0ec7ac8168c80**, with the same module and domain.
- **Predicted readings:**
  - **R1, the application restarts from the top at every step:** the same opening requests (HELLO first) repeat
    with the round count, the watchdog fires at 90 s, and the result is 905.
    - The suspected mechanism: a kind-0 RETURN leaves `slot_paused` at 0, so the next arm is a first entry
      (csupctl 0). If silicon then enters at the domain's original entry and not at the yield's `.Lyield_resume`
      (RETURN rs2), domain_main restarts.
  - **R2, a libc retry loop:** one request (e.g. a write) repeats with a result that makes musl retry, then 905.
  - **R3:** b0-hello prints its line and returns 0. That would mean the trace volume was what kept it from
    finishing in 300 s.
- The heartbeat and watchdog are readable again, so a 905 says Linux was alive.

## B0.7 attempt 11 (23:24-23:28): the gp-captable application runs to its exit on silicon; its OUTPUT is not byte-exact
- **Rungs:** b0-stats 0, **b0-hello 0**, b0-stats2 0.
  - capstone-exec exited 0, after 121 delegated rounds.
  - The census after the exit reads `live_domains 0` and `cached_bytes 688128`. Domain teardown ran through the
    fixed revoke, and the module cached the memory.
- **This is the first time a delegated application has run to completion on silicon.** It ran HELLO, an ioctl and a
  writev through the process ABI, with the monitor's supervised steps under CSR events, and a real gp: no
  fabrication exists on silicon.
- **But the oracle only greps for the line, and the output stream is WRONG.** `/tmp/b0.out` holds:
  - the correct 51-byte line (`B0: hello ... application\n`);
  - then `0: hello from a gp-captable delegated application` (the line again, from offset 2);
  - then 14 NUL bytes, a newline and a NUL.
  - QEMU, on the same image and monitor source, writes exactly the 51 bytes (vm-fix-off/on).
- **The request stream:** round 1 HELLO; round 2 `ioctl(1, TCGETS)` -> -25 (ENOTTY: stdout is a file, so stdio is
  fully buffered); **round 3 `writev(1, iov, 2)` -> 50**, short by one byte of the 51; then **rounds 4..~119
  `writev(1, iov, 2)` -> 0**; then the exit. bytes_out=64, bytes_in=180.
  - musl's `__stdio_write` retries the remainder while writev returns less than requested, so a run of zeros is a
    retry loop. Its length varies: about 116 here, while attempt 10 spent 6,600 rounds in it without finishing.
- **So the write path loses data on silicon.** The marshaled iovec or its data is wrong after the first round. That
  is UNRESOLVED.
  - The leading suspect is the same family as the monitor bug: an `ldc` that moves a linear-family pointer out of
    the domain's iovec or stdio state (Q-12). It is not yet checked against the runtime's marshaling code.
- **Status of B0:** the milestone's control flow is proven on silicon; its output contract is not. B0 is not
  closed until the stream is byte-exact.

## The silicon-hazard census in QEMU: no LDC-move site in the application (2026-10-04 23:30)
- **The tool:** capstone-qemu's S-12 slot tracker (`CAPSTONE_SLOT_LOG`). It lists every granule stored with a
  clear-set capability (LIN/REV/UNINIT/SEALED/SEALEDRET) and loaded again: on silicon the first LDC moved that value
  out, so the next load reads cnull.
  - Run on b0-hello: fabrication OFF, firmware d7d769b1 (monitor d5459e1). b0-hello's output was the exact 51 bytes.
  - Positive control: `CAPSTONE_SLOT_CLEARSET_INCLUDES_NONLIN=1` gives 749 lines, against 96.
- **The application has ZERO hits.** pcc_base 0xe0200000 has only 7 clear-set stores, the glue's SEALEDRET stashes.
  **The writev corruption is therefore not an LDC move in the application.**
- **One hit, in the MONITOR's create_domain.** `dom_gp = __split(...)` (LINEAR) is spilled to a stack slot. The test
  `if (dom_gp != 0)` LOADS it, which on silicon moves it out, and `*(__linear void **)dom_data = dom_gp` loads it
  again. So silicon delivers cnull in the gp slot at data_top-16. The FPGA build has the identical sequence.
  - **This is latent:** the interp glue builds gp from its own cap-table carve and never reads the delivered slot,
    which is why gp-captable domains run.
  - A glue that reads it would get cnull on silicon. It is the same compiler pattern as the managed_reclaim bug:
    a linear value tested and then used.

## B0.7 attempt 12 pre-registered (before the boot): what the host receives in each writev
- **capstone-exec 1dca5b19:** for each traced writev round, every iovec's `{offset, length}` pair as it lies in the
  exchange, plus up to 64 data bytes. b0run.sh traces 200 rounds, so all ~121 print.
- **Build:** Image a1bb71fd4a879f80, firmware e867ddc286c2, the same quiet monitor 1855cb4.
- **Predicted readings:**
  - round 3 with len 51 and the full line as data, but result 50: the host side loses a byte (unlikely: a regular
    file);
  - **len 50: the application computed 50, so its stdio state is wrong on silicon;**
  - rounds 4+ with len 1 and data "\n" but result 0: the host refuses (it should not);
  - rounds 4+ with len 0, or offset/len garbage: the application's marshaling or iovec state is wrong.
- The data bytes show directly whether `"0: hello..."` came from a wrong offset.

## B0.7 attempt 12 (23:46-23:51): the corruption is in the runtime's wire copy (mechanism UNRESOLVED; see the retraction below)
- **The descriptors the host received:**
  - round 3: `iov[0] {0x30, 50}` = "B0: hello ... application" and **`iov[1] {0x70, len 0}`**;
  - every retry: `iov[0] {0x30, 0}`, `iov[1] {0x30, len 0}`.
- **The application's side is correct.** musl's `putc('\n')` -> `__overflow` -> `write(&c, 1)` makes iov[1]
  `{&c, 1}`. The 1 is lost on the way: musl retried a 1-byte write against a 0-length descriptor.
- **The code** (b0-hello.dom 777ec140, dl_vector): `sd t0, 0x0(s7)` (the wire's offset), `sd a6, 0x8(s7)` (its
  LENGTH, the granule's high word), then ~17 instructions later the inlined memcpy's granule loop
  `ldc t5, 0x0(t2)` with t2 = s7.
- ~~This is ISSUES R-29~~ **WITHDRAWN 2026-10-05, see "RETRACTION" below.** It is the same copy idiom as R-29 (a
  struct built with `sd`s, then a 128-bit granule copy of it), but R-29's recorded mechanism does not predict the
  value read.
- **R-29's README says W-04 (the memcpy fixup) is "unaffected, the memcpy loop does not have this shape".** Under
  full LTO the shape is formed ACROSS the inlined call: a struct built with `sd`s, then a granule copy. A
  same-register scan for the shape finds the R-29 reproducer (1 site, distance 1) and nothing here, because the
  copy reads through a different register. The check that matters is by address. This holds whatever the
  sub-mechanism is.

## B0.7 attempt 13 pre-registered (before the boot): the wire-copy workaround in the delegate runtime
- **delegate.c:** every copy of freshly built plain data now goes through `dl_bytes` (8-byte integer moves; an
  8-byte `ld` of the word is forwarded correctly), not libc memcpy. That covers the entry, the iovec wire pairs, the
  msghdr pairs and header, the thread name and the ioctl buffer.
  - b0-hello.dom **cdd82e56**: dl_vector goes from 799 to 528 instructions, `ldc` 43 -> 23. (The first count, "462
    to 350, ldc 9 -> 6", stopped at a local label and is corrected here; the claim-auditor caught it.) The
    remaining `ldc`s that were inspected are capability spill reloads and the iov_base pointer load.
  - QEMU (fw d7d769b1, fabrication off): exactly 51 bytes.
- **b0run.sh now checks BYTE-EXACT** (`cmp` against the expected line). The new **906** means the line is there but
  the stream is wrong; attempt 11 scored 0 on a grep. A stub test scores pass 0, dirty 906, hang 905, no line 903
  and bad rc 904.
- **Build:** Image d5ab7fad7520e2a5, firmware e411395bde2c (quiet monitor 1855cb4), capstone-exec 1dca5b19
  (descriptor trace on).
- **Predicted reading:**
  - round 1 HELLO, round 2 ioctl -25;
  - **round 3 `writev` -> 51**, with `iov[0] {.., 50}` (the line) and `iov[1] {.., 1}` ("\x0a");
  - then the exit, about 4 rounds in all;
  - **RESULT b0-hello 0, byte-exact.**
- If 906 or a retry loop returns: another such copy in the write path, which the descriptor trace will name.

## B0.7 attempt 13 (2026-10-04 23:58 - 10-05 00:03): B0 PASSES ON SILICON, byte-exact, as pre-registered
- **Rungs:** b0-stats 0, **b0-hello 0** (the stream is byte-exact under the new `cmp` oracle), b0-stats2 0.
- **The request stream:**
  - round 1 HELLO;
  - round 2 `ioctl(1, TIOCGWINSZ)` -> -25;
  - **round 3 `writev` -> 51**, with `iov[0] {0x30, 50}` (the line) and `iov[1] {0x70, 1}` ("\x0a");
  - round 4 `exit_group(0)`.
  - That is 4 rounds and 2 syscalls; capstone-exec exited 0.
- **What this run carries, on silicon (caplifive_supcall_715bdd1fe.bit), with no fabricated gp anywhere:**
  - a gp-captable, full-LTO musl application, launched by capstone-exec through the process-ABI module;
  - entered and stepped by the FPGA monitor's supervised CALL under CSR events;
  - delegating its system calls through the yield;
  - exiting with its domain torn down.
- **Images and firmware:**
  - b0-hello.dom cdd82e56884e4dd2; module 0332e2d6 (rv64g); capstone-exec 1dca5b19;
  - monitor caplifive-sbi 1855cb4 (`managed_reinit`, d5459e1, plus quiet step traces), wrapper 882892f,
    firmware e411395bde2c, Image d5ab7fad7520e2a5.
- **The two defects that stood between attempt 4 and this one, both silicon-only and both QEMU-silent:**
  1. capstone-c caller-saves a live `__rev` argument with `stc`, and silicon's STC MOVES it (Q-12's register half).
     managed_reclaim then received cnull, and its REVOKE wedged the core through a nested M-mode trap.
  2. In the delegate runtime, a freshly built `{offset, length}` pair was copied by the LTO-inlined memcpy's
     128-bit granule loop, and the length (the high word) arrived as 0 with the offset intact. The idiom is R-29's;
     the sub-mechanism is UNRESOLVED (see the retraction below).
- **N = 2 (repeat, 2026-10-05 00:05-00:09, same image and firmware):** identical. All three rungs 0, byte-exact,
  the same four rounds, the same descriptors (`{0x30, 50}`, `{0x70, 1}`).

## RETRACTION (2026-10-05): "the stdout corruption is ISSUES R-29" is withdrawn; its sub-mechanism is UNRESOLVED
The claim-auditor weakened it, and the reasons hold up. **What stands:**
- On 715bdd1fe, the LTO-inlined memcpy's 128-bit `ldc`/`stc` granule copy of a freshly `sd`-built `{offset, len}`
  pair delivered the length (the high word) as 0, with the offset intact.
- It is deterministic within a boot (197 retries, every one len 0), across 2 boots.
- The 8-byte-copy workaround (72b8a662e7e9) is byte-exact on 2 boots.

**What is withdrawn:** the attribution to R-29's mechanism.
- R-29's recorded verdict (waveform, 2026-09-10) is a MISS whose refill serves a STALE high word from memory. Here
  the same granule had just been read by iov[0]'s own `ldc`, so iov[1]'s `ldc` most likely HIT. The stale-refill
  account also predicts the previous value; round 3's slot had held 50.
- The 17-instruction distance lies far outside R-29's measured 1-nop window. That window was measured on another
  bitstream and another shape, so it is not a refutation.
- The account that predicts exactly 0 with the low word intact is a word-0 write-buffer hit overlaying `.user = 0`
  (wt_dcache_mem.sv:397, store_unit.sv:370 at 715bdd1fe). ISSUES records that one as REFUTED in simulation
  (`r29-sep-userzero-miss`).
- So B0 is either evidence against that refutation or a third mechanism.
- **Discriminator, not yet run:** word 1 holding a known nonzero value, then plain `sd` to both words with the
  write buffer busy and the line cached, then `ldc`. Stale-refill predicts the old value; the `.user = 0` overlay
  predicts 0.

**Also corrected:**
- 72b8a662e7e9's subject said the defect "zeroed every iovec length". iov[0]'s 50 arrived intact.
- Its body and the delegate.c comment named the word-gated refill overlay as the mechanism. The comment is
  corrected; the lane commits stay as they are.

**What would have caught it:** checking that the known defect's recorded mechanism predicts the observed VALUE, not
just that the code has its SHAPE.

**Resolved the same morning, by the RTL lane's simulation on 715bdd1fe** (r29-wbuf-busy-hit.S, memory delays 12
and 40, identical).
- A 128-bit `ldc` of a granule whose two words were just written by plain `sd`s returns the low word correctly and
  the HIGH word as **0**. That holds with nothing else in flight, and with 17 nops when stores are queued ahead of
  the pair. A plain `ld` of the high word reads correctly.
- B0's exact two-pair shape gives the board's readings: the first pair intact, because stores issued behind it do
  not hold it, and the second pair high 0.
- The path is the pair's residency in the write buffer at the wide load. The load takes its high half from the
  `user` lanes, and a plain store forwards `.user = 0`.
- So it IS R-29's family, by the path R-29's earlier hit arm never created. The retraction stands for the
  mechanism first named, stale-refill. The `.user = 0` account is reproduced.
- The R-29 registry entry and its folder are the RTL lane's to update. The 8-byte-copy workaround stands.

## B0.8 pre-registered (2026-10-05, before the boot): the runtime memcpy's R-29 guard on silicon
**Why.** R-29 (the RTL lane's statement): a 128-bit load of a granule returns a WRONG HIGH HALF while a plain store
to that granule is still in the write buffer. It reads 0 after a fresh low-word store, and the OLD value after a fresh
high-word store. The runtime's memcpy (`string_bounds_safe.c`) copies aligned granules with exactly that load, so
any real application on silicon is exposed. B0 avoided it only in the delegate runtime.

**The guard.** `CAPSTONE_MEMCPY_PLAIN_GUARD`: memcpy and memmove ask LCC's type query about what the 128-bit load
returned.
- Type 7 (NOT_CAP) is copied with two `ld` and two `sd`; a tagged granule still moves by `stc`.
- The query is total on 715bdd1fe and on capstone-qemu.
- Off, the object's .text is byte-identical to before.
- build-b0-hello.sh turns it on for silicon builds (`B0_MEMCPY_GUARD=0` for an A/B) and now takes `B0_APP`.

**The test, b0-memcpy.c (image 73b7cb9a14edd170).** A matched pair after the same fresh stores, with six stores to
another line queued ahead (the RTL lane's busy arm):
- `copy_unguarded`, the old loop. LTO reduced it to one granule `ldc`/`stc`, reached by a call ~10 instructions
  after the stores.
- `memcpy`, guarded. LTO inlined it: `ldc`, `lcc`, `bne 7`, `ld`/`ld`/`sd`/`sd`, ONE instruction after the stores.
  So the guarded arm is MORE exposed than the control, which makes a pass conservative.
- Three faces (fresh low, fresh high, both), 32 reps each.
- Exit 0: the control miscopied and the guard never did. 1: the guard miscopied. 2: the control never miscopied
  (void).
- QEMU (fw d7d769b1, fabrication off): exit 2 with 0/0 on every face, as it must be, because QEMU has no R-29.
  b0-hello still passes there.

**Board run.** One application per boot (R-3: b0-memcpy and b0-hello both enter at 0x10000): b0-stats, b0-memcpy,
b0-stats2.
- Image 41f949e4a4414238; firmware 18d47385a38e (quiet monitor 1855cb4); capstone-exec 1dca5b19; module 0332e2d6.
- b0run.sh's new rung reports the application's exit status as the result.

**Predicted readings:**
- b0-memcpy exit 0, with unguarded_bad > 0 on each face (likely near 32: the RTL simulation was deterministic, and
  the stores sit behind six others), and guarded_bad 0 on every face.
- Exit 2 would mean the control's ~10-instruction distance let the pair drain. That would say nothing about the
  guard, and the next arm would bring the control's stores closer.
- Exit 1 would mean the guard does not hold.

## B0.8 result (2026-10-05 01:37-01:42): THE GUARD HOLDS ON SILICON; R-29's three faces are confirmed on the board
- **Rungs:** b0-stats 0, **b0-memcpy 0**, b0-stats2 0. That is the predicted reading.
- **The unguarded control miscopied:** fresh low **31/32**, fresh high **31/32**, both **32/32**. R-29 fires on
  715bdd1fe in all three of its faces at a ~10-instruction distance, with six stores queued ahead.
- **The guarded memcpy miscopied 0 of 96**, although LTO placed its load ONE instruction after the stores.
- The plain-data guard is therefore the runtime's answer to R-29 for silicon builds (`CAPSTONE_MEMCPY_PLAIN_GUARD`,
  on in build-b0-hello.sh). Compiler-emitted aggregate copies are a separate path: the SQLite silicon build's W-12
  pass covers those. A full application build needs it too.

## Pre-registered for the R-29/S-10b fix bitstream (2026-10-05, from the RTL lane; nothing to run until it exists)
The fix is capstone-ariane sup-call 776d9d859 (in simulation; docs on dev efef06cda618). A read whose granule has a
conflicting store in flight now waits in the dcache read controller until the store drains. The RTL lane's
correction to the account above: at zero distance the torn read was the STORE buffer's word-granular check letting
the load pass; the write-buffer phase begins 2-4 instructions later. Both are closed by the fix.
- **Synthesis first.** The flash needs the lead's own word.
- **The guards stay in force until the fix is measured on silicon:** CAPSTONE_MEMCPY_PLAIN_GUARD, the delegate
  runtime's dl_bytes, and the W-12 pass.

**Acceptance boots, controls first.**
- **Unchanged** (they are controls):
  - the S-16 bare image;
  - R-43 a1..a10;
  - the guarded b0-memcpy build's guarded arm (0/96);
  - C5u without a fence (1,277-ish preemptions; its quiet overhead is the first number to report if it moves beyond
    noise).
- **Changed:**
  - b0-memcpy's UNGUARDED control: 94/96 becomes **0/96**, and b0-memcpy then exits 2. That "void" now means the
    hazard is gone, and the 0 is only meaningful because the same image read 94/96 on 715bdd1fe.
  - The R29 repro rung: 31/32 becomes 0/32.
  - s10b-storebuf-primed as a bare .dom, if the bare harness builds it: 0 of 8 legs trap becomes 8 of 8.
    - Source: capstone-ariane sup-call 776d9d859, `verif/tests/custom/capstone/s10b-storebuf-primed.S`.
      `s10b-storebuf-residual.S` sits beside it.
    - **The verdict is the TRAP COUNT, not the exit code.** It passes either way: 9 exceptions (the control plus 8
      legs) on the fix, 1 on 715bdd1fe.
    - Its header (lines 31-38) is stale boilerplate saying the test does not create its condition. The primed
      variant does; read past line 120.
    - For the bare harness, replace its `tohost` exit with `CAPPRINT(gp)` and `CAP_PASS(s11)`, as the RTL lane did
      for r29-s06agg-shape, and have it report the trap count.
    - Build: testlist_sup.yaml's gcc_opts, with riscv-tests' `isa/macros/scalar` and `env/p` includes.
- **Cost:** a true hazard now waits for the store's AXI write, the cost of a fence; a same-set false candidate costs
  ~3 cycles.

**Measured, 2026-10-05, after the flash (on the lead's word, 16:58-17:00).**
- **b0-memcpy: as pre-registered.** Same image 73b7cb9a, same firmware 18d47385a38e.
  - The UNGUARDED control now miscopies **0/96**: low 0, high 0, both 0 of 32 each. It read 94/96 (31, 31, 32) on
    715bdd1fe.
  - The guarded copy stays 0/96. The image therefore exits 2, which means the hazard is gone.
  - Census rungs b0-stats and b0-stats2: 0 and 0. Boot 19:11-19:16; raw lines `/tmp/capstone/b0/board-b08-776.txt`.
- **R-43 a1..a10: unchanged.** Identity readings are exact and a10's trap and refusal bytes are as on 715bdd1fe.
  Every row was re-derived from the primary logs by a forensics pass. Result lines: board-supmon,
  `fpga-repros/R43-revocation-cache-false-deny/results/board-776d9d859.result-lines.txt`.
- **The S-16 bare set: 18/18 with accept715's readings** (accept776, board-supmon cfa9b34b9adc).
- **C5u: unchanged.** All six tests pass, and quiet supervision costs +0.680 % (+0.686 % on 715); board-supmon
  2a1d03e4a3c1.
- **Corrected label:** the item "the R29 repro rung: 31/32 becomes 0/32" above is b0-memcpy's per-face numbers. R-29's
  own reproducer is s06agg, whose acceptance is 66 -> 64. It and s10b-storebuf-primed are pre-registered on
  board-supmon (b4efce6eabd2) and run after this.
- **The guards stay in the build.** The memcpy guard costs one LCC per granule, and dl_bytes is the clearer code
  either way. Retiring W-12 for memcached is a separate decision once s06agg reads 64.

## B1.0 (2026-10-05): printf on the silicon build, in QEMU
- **The gap.** musl's `vfprintf.o` is DROPPED from the gp-captable archive by the per-member verifier: its
  `long double` needs fp128 constant pools, refused under gp-captable (ISSUES C-43; slot-allocated pools are a lead
  design item). So the silicon build had no printf; B0.8 had to format by hand.
- **The fix is prior art from the c128 line**, ported:
  - `capstone/runtime/silicon/gen-vfprintf-double.py` (origin/rebased/c128-3-musl libc-ext) generates musl's own
    vfprintf with `long double` narrowed to `double`.
  - musl's float formatter needs no 128-bit arithmetic, only the value's type, one `frexpl` and the `LDBL_`
    constants. Every rule must fire and nothing may survive, or generation fails.
  - On musl 1.2.5 all five rules fired (`long double` x7, `frexpl` x1, `LDBL_` x14, signbit, isfinite).
  - build-b0-hello.sh generates it per build and links it ahead of the archive.
- **The test.** `b0-printf.c`, generated by `gen-b0-printf.py`. Its 20 snprintf cases take their expected strings
  from the HOST printf: integers, strings, widths and padding, and doubles in %f/%e/%g/%a.
  - Native: 20/20.
  - **QEMU, fabrication OFF (fw d7d769b1): 20/20, rc 0.** Image 0751dc622b9b78df; it has no fp128 helper symbols.
- **Still open:** `%Lf` now reads a `double`, which matters to no program here (C-20: nothing can produce a long
  double). `floatscan` (strtod/scanf) is dropped the same way, and memcached needs strtod. The generator's `--source`
  mode is the route, applied to floatscan.c and its header and callers together.
- **Board run, pre-registered 2026-10-05 before the bake.**
  - The image is the QEMU-passed `b0-printf.dom` (0751dc622b9b78df), baked into a private Linux image with the B0
    files (`/tmp/capstone/b0/b0-bake-printf.sh`, a copy of b0-bake.sh with this image and its own output dir). The
    firmware has the same monitor and defines as B0.8's (`-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS
    -DCAPSTONE_SUPERVISE_QUIET`).
  - Bitstream 776d9d859. Rungs in order: `b0-stats` (control), `b0-printf`, `b0-stats2`. b0run.sh's new
    `b0-printf` rung returns the application's own exit status: the number of cases that differ.
  - **Predicted:** `B1.0 printf: 20 of 20 cases match`, `RESULT b0-printf retval=0`, no MISMATCH line, and both
    census rungs 0.
  - A MISMATCH on a `double` case only, with the integer and string cases intact, would put the defect in the
    narrowed float path on silicon, since QEMU passed the same image.
- **Board result, 2026-10-05 19:37-19:42, 776d9d859: as pre-registered.** `B1.0 printf: 20 of 20 cases match`,
  `RESULT b0-printf retval=0`, no MISMATCH line, census rungs 0 and 0. Firmware fw_8068f626e0d7, image
  0751dc622b9b78df (manifest of the private image), raw lines `/tmp/capstone/b0/board-b10-776.txt`. So printf,
  doubles included (%f/%e/%g/%a), works in a gp-captable delegated application on silicon.

## B1.0b (2026-10-05): strtod, atof and scanf's %f on the silicon build, for memcached
- **The gap.** The archive drops musl's `floatscan.o` for the same reason as vfprintf (C-43). `strtod.o` and
  `vfscanf.o` survive but call `__floatscan`, so any program that parses a float fails to link. memcached does
  (`safe_strtod` in util.c, `atof` in its option parsing).
- **The fix: `gen-floatscan-double.py`, vfprintf's sibling.** It generates musl's own floatscan, strtod and vfscanf
  with `long double` narrowed to `double`.
  - The substitution selects musl's own `LDBL_MANT_DIG == 53` configuration, which arm and mips use, so the parser
    is still musl's correctly rounded one.
  - The entry is renamed `__floatscan_d`, so an archive member still expecting the fp128 `__floatscan` fails to
    link, not mis-reads.
  - `strtold` is not generated, and `%Lf` stores a double (C-20).
  - Every rule must fire, and nothing long-double may survive.
- **The generator's own gate had a blind spot, caught downstream.**
  - Its first leftover check looked for `long double`, LDBL_ and the l-suffixed libm calls. It passed while
    `1000000000.0L`, a long double LITERAL and so an fp128 constant, survived in floatscan.c.
  - The backend's gp-captable verifier then refused the object (constant-pool data, C-43).
  - Both generators now also reject decimal, exponent and hex literals with an `L` suffix. musl's vfprintf has
    none, and its generated output is unchanged.
- **The test, `b0-strtod.c` (generated by `gen-b0-strtod.py`).** It covers 21 strtod cases, plus atof and
  `sscanf("%lf")`. Each expected value is the host's correctly rounded double, compared bit for bit, with its end
  offset and errno.
  - Native glibc: 23/23. The negative control (one expected bit flipped) reads 22/23 and exits 1.
  - **QEMU, fabrication off and on:** every value and end offset matched on the first run. The one mismatch was errno
    on the smallest subnormal: glibc sets ERANGE there, musl does not. C leaves that implementation-defined, so the
    case now checks value and end only.
- **Board run, pre-registered before the boot (776d9d859):** b0-strtod (e7d30ad72f104eed) in a private image whose
  b0run.sh has the b0-strtod rung (0a00ffab4e23d16c; the first bake took the previous wrapper and was redone).
  Rungs b0-stats, b0-strtod, b0-stats2.
  - **Predicted: `B1.0b strtod: 23 of 23 cases match`, retval 0,** since the parser is pure integer and double
    arithmetic, which silicon already runs in vfprintf.
  - A mismatch confined to subnormal or rounding cases would point at the soft-float/FPU path on silicon, not at
    the narrowing, which QEMU passed bit for bit.
- **Board result, 2026-10-05 20:22-20:27: as pre-registered.** `B1.0b strtod: 23 of 23 cases match`, retval 0,
  census rungs 0 and 0. Firmware fw_b15d76e38a54, image e7d30ad72f104eed; raw lines
  `/tmp/capstone/b1/board-strtod.txt`. memcached's float parsing (strtod, atof, sscanf %lf) therefore works on
  silicon, bit for bit against the host.

## Open, to settle before B0.7
- Does the board's buildroot carry the process-ABI modcapstone and a capstone-exec? Not checked.
- B0.1 changes the monitor every lane boots. The first boot of it is announced, and the previous firmware stays the
  fallback.
- Untagged function pointers under gp-captable are acceptable for the hello path. This is the prior-art sweep's
  assumption; check it in B0.5's disassembly.
