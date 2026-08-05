# A-lane task: finish the large-`.rodata` delivery path (monitor copy → QEMU proof → two-pass)

**Owner: A lane (in-lane — do not delegate).** Touches the OpenSBI monitor,
capability-ABI-adjacent glue, and the SQLite critical path; A holds the hot context.
The B lane is explicitly fenced off these files (see
`plans/ladder-breadth-task-B.md`).

Canonical design + contract: `plans/sqlite-on-silicon-scoping.md` (the large-RO
implementation contract + the "STATUS 2026-07-24 — toolchain side DONE" block). This
doc is the *execution checklist* for the remaining half.

## Where we are
**Toolchain side DONE (uncommitted):**
- `link-gpfree.ld` — `__gpfree_globals_base` anchor at the globals base
  (`base + 0x1000`), linkable so glue can `lla` it.
- `gen-gp-captable-glue.py` — large-RO **copy path**: for an initialized global
  `> 256 B` with an 8-multiple size and a **file-scope** (non-`.L`) symbol, emit a
  copy loop `lla sym - __gpfree_globals_base` → `cincoffset(sp, off)` (src in
  dom_data) → cap-table storage (dst), instead of an `li`/`sd` immediate storm.
- Rung `beebs_crc32big` (upstream 256-entry `const crc_32_tab`, **2048 B**, external
  linkage; oracle **1703161001**). Builds, assembles, links; static gate `cjalr=0
  ldc-gp=2`; emitted glue shows the 11-instr copy loop, no `li`/`sd` storm.
- **Cannot run yet** — needs the monitor to place the initializer blob in dom_data.

## Remaining work (in order)

### 1. Monitor copy in `create_domain` (the gated step)
File: `capstone/caplifive-buildroot/package/capstone-sbi-domain/capstone-sbi/sbi_capstone.c`
(canonical edit copy per `project_fpga_fw_payload_build_recipe`).

`create_domain` (`:291`) layout is already understood:
- `:299` `dom_code = split_out_cap(base, tot_size, 1)` — full-authority image cap.
- `:301` `dom_seal = __split(dom_code, base+code_size)` → `dom_code` = `[base,
  base+code_size)` (still holds **read** authority, derived from `mem_l`).
- `:302` `dom_data = __split(dom_seal, base+code_size+DOMAIN_DATA_SIZE)` — fresh RAM,
  delivered to the domain via cscratch (`dom_seal[2]`).
- `:316` `__seal(dom_seal)` — after this `dom_code` is consumed.

**Insert between `:302` and `:309`** (while `dom_code` is still a usable linear cap):
memcpy the initialized-globals byte range **`[base + GPFREE_GLOBALS_OFFSET,
base + code_size)`** (GPFREE_GLOBALS_OFFSET = `0x1000`, matching `link-gpfree.ld`)
from `dom_code + 0x1000` into the **front of `dom_data`** (`dom_data[k] ==
image[base + 0x1000 + k]`). Then the glue's `lla sym - __gpfree_globals_base` offset
(`= sym_off`) indexes `dom_data` correctly via `sp`.

Watch-outs (resolve at implementation time with the hot context):
- Read must go through `dom_code` (full-authority, spans the globals) **before**
  `__seal`. Write into `dom_data` (fresh RAM) — both linear, both M-mode-usable
  (`split_out_cap` derives from `mem_l`, so the monitor can read the image).
- **Verify `code_size` semantics** against the controller that computes it and
  against the linked `.dom` ELF: confirm `[base+0x1000, base+code_size)` actually
  covers `.rodata` (where `crc_32_tab` lives). `.bss` is NOLOAD (zeroed by glue, not
  copied); `.gct` sits after `.bss` in the script — check whether the copied range
  should stop at the end of `.data`/`.capstone_cap_init` rather than the whole
  `code_size`. For `beebs_crc32big` only `.rodata` matters, so a conservative
  "copy `[0x1000, code_size)`" is fine to first-light; tighten later if needed.
- Blob (front of dom_data) vs. glue-carved storage (from dom_data's END downward) do
  not collide for these sizes; the front blob is later reused as stack scratch —
  harmless (the copy into cap-table storage already happened).

### 1-STATUS (2026-07-24 v3, DEFINITIVE): monitor REGEN path is broken — large-RO blocked on it
Ran the isolation to the end. **The monitor-regen path cannot reproduce the working monitor in
this environment — every regeneration boot-hangs; only the checked-in prebuilt boots.** Details:
- Good monitor `fw_jump.elf` md5 `6724bcb3` (checked-in **prebuilt**) — boots, all rungs pass.
- **Every** regen → md5 `788f8a1a`, **hangs at boot (zero serial)**. Confirmed with BOTH
  capstone-c `8cda52c` (drifted) AND the pinned `4899cf9` — so it is **NOT the commit drift**;
  both compilers produce the same broken monitor. (H1 = my copy code is moot; the reverted,
  no-change source also boot-hangs.)
- Root of the divergence (diff of good vs regen `.c.S`): the current compiler allocates **more
  callee-saved regs / bigger frames** (`s0–s11`, frame −464) than whatever built the good
  prebuilt (`s0–s6`, frame −368). I.e. the working monitor was built by a **different compiler
  state not reproducible from the current capstone-c** — a real toolchain gap.
- **Consequence:** large-RO on QEMU is **blocked** — I can't rebuild the monitor to add the
  copy without boot-breaking it. This is NOT a quick fix; fastest path = ask whoever produced
  the prebuilt which exact capstone-c commit + flags built `fw_jump` (6724bcb3), or debug why
  the current compiler's monitor output hangs (register-alloc / linear-dyn-offset codegen).
- **Safe state left:** good `fw_jump` restored (B unblocked); good-`.c.S` backup at
  `/tmp/claude-.../llvm-capstone-b/.../sbi_capstone_dom.c.S.orig`; broken regen saved at
  `/tmp/capstone/fw_jump.elf.largero-broken`; capstone-c back at session-start `8cda52c`;
  component source reverted. **DO NOT rebuild the QEMU monitor** until the toolchain gap is
  resolved (it silently boot-breaks the shared `fw_jump` for both lanes).
- **Decoupled:** the FPGA handoff for the non-large-RO rungs (matmult, coremark_matrix,
  rv8_primes, beebs_crc32/insertsort/prime/recursion) needs NO monitor regen — proceed there.

### 1-STATUS (2026-07-24 v2, superseded): monitor rebuild BROKE BOOT — reverted + restored
**Outcome: the large-RO monitor copy is BLOCKED at the rebuild step, not the code.** The copy
loop compiled cleanly under capstone-cc (`.c.S` regenerated, `fw_jump` relinked), but the
**rebuilt `fw_jump` hangs at boot with ZERO serial** (no OpenSBI banner) — and `create_domain`
isn't even called at boot, so this is a **whole-monitor** breakage, not the copy logic.
- **Restored:** the known-good `fw_jump.elf` from `/tmp/capstone/fw_jump.elf.orig.bak2` (==
  `.bak`, md5 `6724bcb3…`, the Jul-22 20:04 monitor). `matmult_int` boots + PASSES again — B
  unblocked. The broken rebuild is saved at `/tmp/capstone/fw_jump.elf.largero-broken`.
- **Suspected cause = the monitor REGEN itself, not the copy.** capstone-c advanced to
  `8cda52c`; its recent commits are codegen fixes for *"loading from dyn addresses"* /
  *"linear addr dyn offset branch"* — i.e. regenerating the monitor with current capstone-cc
  may miscompile it (the good `fw_jump` was a **prebuilt**, likely from an older capstone-cc).
- **To isolate H1 (my dyn-offset copy code) vs H2 (regen breaks the monitor regardless):**
  revert the copy (DONE — component source reverted), `rm sbi_capstone_dom.c.S`, rebuild, boot
  test. If it STILL hangs → H2 (regen path broken; large-RO on QEMU needs a fixed capstone-cc
  or a different validation route). If it boots → H1 (the `dom_gp[wi]=…` dyn-offset codegen is
  the specific trigger; rework the copy, e.g. avoid variable-offset linear-cap access).
- **DO NOT rebuild the monitor** until this is diagnosed — a rebuild silently re-breaks the
  shared `fw_jump` (affects both lanes). The component source edit is reverted; the package
  copy still carries an (inert, non-fw_jump) copy edit + `GPFREE_GLOBALS_OFFSET` define.
- **Decoupling:** the FPGA handoff does NOT need this — the non-large-RO rungs (matmult,
  coremark_matrix, rv8_primes, beebs_crc32/insertsort/prime/recursion) are silicon-ready and
  boot on the restored monitor. Large-RO (crc32big → SQLite) is a separate, blocked track.

### 1-STATUS (2026-07-24 v1, superseded): monitor copy IMPLEMENTED (pending rebuild)
The copy is written. **Critical finding — the monitor has three divergent copies; the QEMU
`fw_jump` builds from the OPENSBI COMPONENT copy, not the package copy:**
- `build/local.mk`: `OPENSBI_OVERRIDE_SRCDIR = components/opensbi` → `fw_jump` compiles
  `components/opensbi/lib/sbi/capstone-sbi/sbi_capstone.c` (the **`__split` gp-delivery**
  variant: `dom_gp = __split(dom_code, base+0x1000)` holds the globals; `dom_code` is only
  `[base, base+0x1000)`). The large-RO copy is added THERE, reading from **`dom_gp[wi]`**
  (not `dom_code`), placed right after the `dom_gp` split while both cursors are fresh and
  before `dom_gp` is consumed by the cscratch store. Loads through `dom_gp` don't consume it.
- The `capstone-sbi-domain` **package** copy (cap-table variant) also got a copy edit (reads
  `dom_code`) — harmless but NOT what QEMU builds; the component copy is authoritative for QEMU.
- Copy uses the `void*[i]` word-copy idiom (capstone-c `enclave_code[i]=code_base[..]`); const
  tables carry no cap tags so a plain word copy is exact.

### 2. OpenSBI monitor rebuild
Mechanism found 2026-07-24:
- Wrapper `components/opensbi/lib/sbi/sbi_capstone_dom.c` `#include`s the edited
  `capstone-sbi/sbi_capstone.c`. capstone-cc compiles the wrapper → `sbi_capstone_dom.c.S`,
  checked into the srcdir and generated **out-of-band** (NO `%.c.S: %.c` make rule). Editing
  the `#include`d `.c` doesn't change the wrapper's mtime → the `.c.S` is stale.
- So: (a) build capstone-cc (`cd capstone/capstone-c && cargo build --release`, or
  `./local_build.sh`); (b) **force-regen** `sbi_capstone_dom.c.S` (+ `capstone_int_handler.c.S`)
  from their wrappers via capstone-cc with the ABI/cpp flags matching the existing `.c.S`
  (gp-free README documents `make ... A=opensbi-rebuild CAPSTONE_CC_PATH=$(realpath
  ../capstone-c)` — VERIFY that target/invocation before running); (c) rebuild `fw_jump` via
  buildroot; new `fw_jump.elf` → `buildroot/build/images`.
- OPEN: exact regen make-target / capstone-cc flags unverified — confirm before building so
  the monitor isn't subtly miscompiled. Submodule *source* stays uncommitted (rule 7).

> **WITHDRAWN 2026-08-05:** the "submodule source stays uncommitted" rule above no longer
> applies. Submodule work is now COMMITTED on a branch (see CLAUDE.md). Keeping the live
> monitor uncommitted nearly cost the trace markers every board verdict depends on.


### 3. End-to-end QEMU run — first light
Run `beebs_crc32big` in a domain on QEMU (silicon config). **Expect retval
`1703161001`** (== the runtime-table `beebs_crc32` rung's oracle — both fold the same
table). Green here = the large-RO delivery path works end-to-end for a file-scope
`const` table. Serialize with any B-lane QEMU run (shared `rootfs.ext2` lock).

### 4. Two-pass baked-offset variant (SQLite prerequisite — follow-up)
`static` / function-local (`.L…`) large consts have **no linkable symbol**, so the
`lla`-based copy path can't reference them — and SQLite's tables are exactly this.
Add a two-pass build: link once, read the resolved global addresses from the linked
`.dom` symtab, regenerate the glue with **baked `li <offset>` constants** (offset =
`resolved_addr - globals_base`), relink. This unblocks SQLite's `static const`
tables. Scope/prototype it on a `static`-table rung before wiring SQLite.

## After this
SQLite Stage 3/4 on the ladder, then batched silicon board runs (both in-lane).

## Constraints
All permanent repo rules (`CLAUDE.md` / `DELEGATION.md`). Commit only when asked; no
submodule-source commits; no real-person names; bug-fix/root-cause notes →
`history/` dated. A-lane branch `capstone-bootstrap`.
