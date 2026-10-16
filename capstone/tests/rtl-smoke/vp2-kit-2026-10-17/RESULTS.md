# vp2's Tier 1 and Tier 2 kit: the pipeline proof on dev's runtime (2026-10-17 03:14-03:40)

This proves the pipeline pre-registered in README.md. It is not vp2's kit. That kit is built from the ABI v6 runtime
revision when the compiler lane hands it over, and no hash here is final.

## Inputs

- **Runtime:** dev 4798dcab2d52, the revision whose SDK failed on the October 10 compiler (README, "Status of the
  pipeline proof"). dev's `runtime/`, `ports/musl-capstone`, `ports/sqlite` and `ports/mruby/app` have not changed
  since.
- **Compiler:** the compiler lane's dev-current build, `/tmp/capstone/llvm-dev-current-20261017`. It is clang
  85b1daddf230 (`clang` sha256 `ba603a574ce08ea4`): a lane commit whose `llvm/`, `clang/` and `lld/` equal origin/dev's
  in both directions. It is Release with assertions, and carries C-80's fix (7ce213020f87).
- **Reference QEMU:** capstone-qemu `virtual-capstone` d35af47b163c (`/mnt/hdd/<user>/capstone/qemu-pin-d35af47b`).
- **Guest images:** `/tmp/capstone/pinned-platform/images-root/build/images`.
- **One local change in the source worktree:** `ports/sqlite/app/prepare-sources.sh`, the fix landed on dev as
  e012dd105f00. Without it no build of the app-port SQLite can run on dev (below). The kit now logs any such change
  (`SRC-LOCAL-CHANGES`).

## What the build produced

| piece | sha256 (16) | gate |
|---|---|---|
| the virtual SDK (musl survey) | — | **1,361 of 1,361 files compile, 0 fail.** The October 10 compiler failed 101 on the same runtime, so the compilers are told apart two-sided. |
| `capstone_vm.ko`, QEMU guest (linux-custom 6.1.0) | `58cb6b529956867c` | — |
| `capstone_vm.ko`, board (linux-6.4.14) | `76c7740dd655e044` | **0** 16-bit instructions; `vermagic=6.4.14`, the board kernel's release |
| `capstone-vexec`, board | `796f073f7778f944` | a static RISC-V executable |
| `capstone-vexec`, QEMU | `ea816da94dff1439` | — |
| `contract.dom` | `f3af2896838cf968` | — |
| `sqlite3.dom` (Tier 2) | `674a347559e1755e` | — |
| mruby 4.0.0-rc2 (Tier 2) | `754d11a9353584ff` | 124 capstone objects, 0 failed |

**Reproducibility, from two builds of the same inputs three minutes apart:**
- The board module and `contract.dom` are byte-identical.
- **Both launchers are not.** Each build differs in 6 bytes of `.strtab`: one FILE symbol carries the temporary
  object name gcc gives `runtime/linux/stub.S` (`ccpyFFc5.o`, then `ccMjCO8h.o`). With the symbol table stripped,
  the two builds are identical (board `dc92a2e343c52635`, QEMU `e6cdc8cddb5412db`). So a launcher's hash pins the
  file that was staged, and a rebuild does not reproduce it. `stub.S` has no `.file` directive. Adding one would
  fix this; that change is the runtime lane's.

## The run.py gate (the QEMU prediction)

**PASS: 41 of 41 checks, 22 steps.** The record is `out/result.json` (`1f91ffe4f1d2edf4`, virtual VM ABI 5). The
run is at this revision's own setting, exact bounds on (`x-capstone-exact-bounds=true`). Each step's exit status
is run.py's table:
- normal, threads, vm, nested, heap-threads and sparse-churn: 0;
- nested-stale, stale, bounds, retired, vm-ro-write, vm-none-read, vm-guard, vm-padding and sparse-stale: 139;
- preempt: 143;
- parallel_one, parallel_two, sqlite_write, sqlite_read, mruby and cleanup: 0.

**The first attempt timed out.** It ran at run.py's default `--timeout 300` and stopped after vm-padding, with
status FAIL and every later test false. The two 200,000-allocation recycling cases do not finish in 300 s while
this host also runs another lane's QEMU. That record is kept as `out/result.timeout300.json`. The kit now passes
`--timeout 1500`.

**The board arm (`make-board-arm.py`):**
- It is generated from this run.py's own guest script (SCRIPT sha256 `fb77127eab03b2f9`), with the staging root
  `/mnt/vm` replaced by `/test-domains/vm`.
- It lists 25 steps. Perl, its smoke test and legacy run only when their files are staged, which leaves these 22.
- The arm ends with `BASELINE-PROBE vm_steps = <n>`, so the predicted board reading is 22 steps with the statuses
  above.

## The same pipeline on the ABI v6 revision (a first reading, not vp2's prediction)

The kit also built from the compiler lane's `lane/compiler-exact-bounds-iii` 85b1daddf230 with the same toolchain:
- the module, `capstone_vm.ko` with `node_max_records`, compiled for the first time for both kernels: QEMU
  `97334f0fadd8ffd4`, board `dc0a0de0ea408a7b`, 0 16-bit instructions, vermagic 6.4.14;
- the SDK's 89 built files and its libc are byte-identical to the compiler lane's own build of that revision;
- that branch predates e012dd105f00, so its SQLite was prepared with the fix applied locally.

**run.py on it (exact bounds off, option (iii)'s default there) FAILS:**
- `normal` exits 1 with `VM_CONTRACT_FAIL:heap growth/zero`;
- `heap-threads` exits 1 with `VM_CONTRACT_FAIL:child malloc` and `VM_CONTRACT_FAIL:parent malloc`;
- the run then reached the 300 s timeout after vm-padding.

The compiler lane states this SDK "builds and produces v6 domains; not yet verified to work". This reading went to
that lane. It does not separate the v6 runtime from the bounds profile, since both differ from the passing run
above; the lane's own on/off control does.

**The cause, as far as it is established (corrected 2026-10-17 05:00; the first version of this paragraph,
dev ee0fa6d0fe55, is retracted below).**

What stands, from the compiler lane's audit (lane commit 811840072b8f):
- Each measured refusal is the module's `add()` at `runtime/virtual/module/capstone_vm.c:466`:
  `!c->exact_bounds && (!is_power_of_2(r->bytes) || (r->address & (r->bytes - 1)))` → -EINVAL.
- `exact_bounds` comes from CSR 0x5c0 bit 0x100, which option (iii) turns off.
- In the lane's instrumented run, all five refusals read errno 22. The mapping sizes were 28672, 200704, 1126400
  and 6307840, none a power of two.
- For each, every disjunct of `add()` before line 466 is false, and the launcher's own EINVAL paths do not fire. So
  the launcher is excluded.
- The gate bites only on the v6 local-malloc heap. There the launcher grants the exact visible size
  (`exec.c:266`, `bytes = visible`). Its other paths grant a naturally aligned power of two and pass.

What is NOT established:
- **Which predicate the gate is, empirically.** Every measured size is both non-power-of-two and above 16384, so a
  size threshold fits the data equally well. Only the source decides between them so far. The pilot's 20480 and
  36864 are modelled group sizes, never measured.
- **run.py's heap-growth failure.** No heap growth completed in any arm, exact bounds on or off. The lane's
  `CAPSTONE_VM_REMAP_OK` reads 0 in all three arms, and that instrument is proven live. So growth fails for a
  further reason, still unidentified. `remap_begin()`'s refusal of every remap with exact bounds off is in the
  source, but nothing shows it is the reason.

> **RETRACTED (2026-10-17 05:00), from this paragraph's first version (dev ee0fa6d0fe55):**
> - **"The compiler lane confirmed it two-sided: its `--exact-bounds` arm clears all four failing contract
>   markers."** Those four strings are absent from the exact-bounds-on arm, but that arm FAILS too:
>   - 5 of its 8 failures survive;
>   - two of the four markers are replaced by different failures, a capability out-of-bounds fault and a
>     "denial survived" safety failure;
>   - turning the profile on also changes the encoder, so the arm is not a single-variable control.
>   It confirms nothing about the gate. I relayed the claim without reading the arm's own log.
> - **`remap_begin()` as part of the cause of run.py's heap-growth failure.** See "What is NOT established" above.
> - **"Every NULL ... needs such a mapping (20480, 28672, 36864 B and up)"** mixed measured and modelled sizes.
>   The measured ones are listed above.

The fix to `add()` belongs in the module, and the board module carries the same file, so it applies to the board
too. The compiler lane is running the two arms that separate the predicates, before any revision: a request whose
group needs 32768, and one whose group needs 12288.

## Defects found on the way, and where they went

- **dev's app-port SQLite could not be prepared** after 423d13f5b81f, which moved the sorter rewrite into
  `adapt-sqlite-322.sh`. Fixed on dev, e012dd105f00. The fix was gated byte-identical against the pipeline before
  423d (`sqlite3.c` 7fc92e19a3af…), with the failure reproduced on dev's script and a negative control.
- **The kit's own two:**
  - it did not export `CAPSTONE_QEMU_LOCK`, which run.py takes itself;
  - a manual rerun lacked `CAPSTONE_LLVM_BIN`, which run.py reads for `llvm-nm` after the guest has run.
  Both are fixed in the kit script.
