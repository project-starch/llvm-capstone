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

### 2. OpenSBI monitor rebuild
Apply the `.c.S`-regen gotcha (`project_opensbi_monitor_rebuild_include_wrapper`):
force `.c.S` regeneration or `fw_jump`/`fw_payload` relinks stale. Two monitor copies
exist — rebuild the one QEMU actually loads (the prebuilt `fw_jump.elf` in
`buildroot/build/images`). Submodule *source* stays uncommitted (hard rule 7).

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
