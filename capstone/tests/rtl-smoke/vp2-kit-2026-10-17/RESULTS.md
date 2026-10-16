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

## The pre-fix "before" readings on the instrumented launcher (lane/compiler-exact-bounds-iii fe1a12fe7c74, 2026-10-17 05:11-05:40)

**Kit:**
- QEMU module `5a3b57ec3873c273`, board module `dc0a0de0ea408a7b`;
- module headers `cap_rev_table_abi.h` `257fbd0ea0ca8844`, `vm-abi.h` `67b18ae168ec2178`, `wire.h` `55debb3c71f3c92b`.

**Positive control for "ADD_REFUSED absent", the pre-registration's run 1:** bundle 5's cases 00 and 10 on this kit.
- Without `VIRTUALVM_TRACE` the serial held 0 refusal lines while case 00 was refused, so that run proved nothing.
- With it: **10 `CAPSTONE_VM_ADD_REFUSED` lines**, all from case 00.
  - The sizes are 36864 and 49152. 36864 is the group size the compiler lane modelled.
  - All have `pow2=0`, so the size half refuses them. Their `naturally_aligned` field carries no information:
    `address & (bytes - 1)` is not an alignment test when bytes is not a power of two.
- So the line fires through this harness.
- **Case 10's malloc(6144)** was admitted in this boot. That makes 2 of 5 boots on pre-fix launchers, and every
  other boot read NULL.

**The positive control repeated untraced** on a kit from 561b8b2cd692, where the compiler lane made the refusal
line unconditional:
- gate.sh exports no trace;
- case 00 prints **10 lines** (49152 B, all `pow2=0`);
- case 10 is admitted, making 3 of 6 boots on pre-fix launchers.
So the post-fix absence check runs on the unmodified harness, and the trace switch is not needed for this line.

**run.py at `--timeout 1500`:** completes, with status FAIL.
- `normal` exits 1 on `heap growth/zero`.
- `heap-threads` exits 1 on `child malloc` (twice).
- `parallel_one` and `parallel_two` exit 1, each on `heap growth/zero`.
- `sqlite_write` exits 7 (SQLite's NOMEM), `sqlite_read` 1, and `mruby` 1.
- Every step that does not depend on heap allocation passes as on dev's runtime: threads, vm, nested, the 139
  denials, preempt 143 and cleanup 0.
- The record still says `virtual_vm_abi` 5. That field was hard-coded at this revision, and the compiler lane has
  fixed it since.

## run.py on b61f56b10c06 without the `add()` change: sqlite_write's NOMEM is `add()` refusals, not the refused growth (2026-10-17 05:59-06:10)

Pre-registered at the lane's README ("run.py on b61f56b10c06 WITHOUT the add() change"). Kit: QEMU module
`6dce3015bfe8566d`, headers as before, exact bounds off, `--timeout 1500`. Status FAIL.
- **Computed row, held:** `normal`, `parallel_one` and `parallel_two` exit 1 on `heap growth/zero`. Each step's own
  output carries a 135168-byte `CAPSTONE_VM_ADD_REFUSED` (`pow2=0`).
- **The question, answered:** `sqlite_write` still exits 7. So its NOMEM is not the growth that b61f56b10c06
  fixes.
- **Every failing step's own output carries `add()` refusals.** The guest script prints each step's `VM_EXIT`
  line before that step's captured output, so these are the lines after each exit line:

  | step | refusals |
  |---|---|
  | normal | 49152 and 135168 |
  | heap-threads | 1 |
  | sqlite_write | 49152, 28672, 122880, 94208, and 8192 with `pow2=1 naturally_aligned=0` (the alignment half) |
  | sqlite_read | 5, all `pow2=0` |
  | mruby | 49152 and 40960 |

- So, before the `add()` change, every remaining run.py failure has `add()` refusals in its own output. The
  pre-registered inferred rows (heap-threads, sqlite_write, sqlite_read and mruby clear) are now each backed by an
  observed refusal in their step. Whether they clear is still the fix run's question.

## The v6 `add()` fix (lane/compiler-exact-bounds-iii 89ee98be99e5): every registered board-lane row holds (2026-10-17 07:34-07:47)

The change is "Option A". In `add()`'s exact-bounds-off branch, `!capstone_repr_ok(r->address, r->bytes)` replaces
the power-of-two test. `representable.h` is shared into the module build. The project lead delegated the choice to
the compiler lane, after an audit.
- Its runtime differs from b61f56b10c06 in `capstone_vm.c` and `build-adapter.sh` only. So the b61 run above is a
  single-variable "before".
- The runs are as pre-registered in README.md: the unmodified corpus harness, QEMU `40264f5de09ef153`, the same guest
  images, exact bounds off.

**The kit; the module's first build:**
- QEMU module `d24d036ec8c4edc3`, board module `07ba2c4f7425310f`;
- 0 16-bit instructions, vermagic 6.4.14;
- MODULE-HEADERS now lists `representable.h` (`7c41a409acb7a897`) beside `cap_rev_table_abi.h` and the module's own
  headers.

| registered row | reading | held |
|---|---|---|
| run.py `normal` and both parallel steps: `heap growth/zero` clears (computed) | `normal` 0, `parallel_one` 0, `parallel_two` 0 | yes |
| heap-threads, sqlite_write, sqlite_read, mruby clear (inferred) | 0, 0, 0, 0 | yes |
| run.py overall | **PASS, 41 of 41** (record `c2e0bfa7901e1c76`). Every step's status is the dev-runtime table's. | — |
| `CAPSTONE_VM_ADD_REFUSED` absent | **0** in run.py's serial, against 35 in the b61 "before". The untraced positive control on 561b8b2cd692 showed the line fires. | yes |
| Tier 3 bundle 5 in full: 13/13 equal to the record; cases 00 and 10 CAUGHT at cause 28 | 13 of 13 equal, cell by cell (verdict, cause, faulting symbol, attribution, reached); 0 refusal lines | yes |
| cases 00 and 10 CAUGHT in every one of ≥ 3 more fresh boots | CAUGHT in all 3, and in the full run: 4 of 4 boots. Pre-fix, case 10 was admitted in 3 of 6. | yes |
| bundles 13 and 14 do not move | 9/9 and 3/3 equal to the record, cell by cell | yes |

**No new out-of-bounds access.** In run.py's serial the domain-fault causes are the same before and after:
- 13 ×2, 15, 24, 25 ×3 and 28 ×2, every one an expected denial step;
- QEMU's capability-violation lines are the same 6 classes: 3 revoked, 2 out-of-bounds, 1 "requires capability".

**The compiler lane's run-malloc rows** (its own run on this kit's QEMU adapter; its lane eeb083159e2e):
- these held: the five allocation markers, the align-probe at **600 of 600** (413 failed before), 0 refusal lines,
  `REMAP_OK` 0, and both predicate arms ALLOCATED;
- the aggregate did not hold: **18 / 5** where 21 / 2 was predicted.
  - The three extra failures are one event. `basic` now allocates, runs further, and dies on a previously unreachable
    fault: cause 29 at address 0, after a kind 2 boundary.
  - That lane reports no cause 28 and no out-of-bounds line in that arm. It is investigating the fault as a separate
    thread, not as a defect of the change.

## Defects found on the way, and where they went

- **dev's app-port SQLite could not be prepared** after 423d13f5b81f, which moved the sorter rewrite into
  `adapt-sqlite-322.sh`. Fixed on dev, e012dd105f00. The fix was gated byte-identical against the pipeline before
  423d (`sqlite3.c` 7fc92e19a3af…), with the failure reproduced on dev's script and a negative control.
- **The kit's own two:**
  - it did not export `CAPSTONE_QEMU_LOCK`, which run.py takes itself;
  - a manual rerun lacked `CAPSTONE_LLVM_BIN`, which run.py reads for `llvm-nm` after the guest has run.
  Both are fixed in the kit script.
