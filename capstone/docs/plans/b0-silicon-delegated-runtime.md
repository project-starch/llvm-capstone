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

## Open, to settle before B0.7
- Does the board's buildroot carry the process-ABI modcapstone and a capstone-exec? Not checked.
- B0.1 changes the monitor every lane boots. The first boot of it is announced, and the previous firmware stays the
  fallback.
- Untagged function pointers under gp-captable are acceptable for the hello path. This is the prior-art sweep's
  assumption; check it in B0.5's disassembly.
