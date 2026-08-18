# C-19 — the cap-init block split raises CAPABLITY_OUT_OF_BOUND on silicon

**This is a COMPILER regression, not a silicon defect.** It is filed here rather than in
`tests/fpga-repros/` because that folder is for suspected RTL/silicon defects handed to the
hardware side; this one belongs to whoever owns the Capstone backend. Sibling issues that a
reader may have arrived looking for: S-06 (untagged 128-bit `ldc`/`stc` high half) and S-08
(dom-switch CSR clobber) are FIXED in silicon; S-07 (a capability read back untagged) is the
open silicon issue and lives in `tests/fpga-repros/S07-capability-untagged-on-reload/`.

## Verdict

`llvm/lib/Target/Capstone/CapstoneCapGlobalInit.cpp`, as merged in `64ee0bb8b37a` (merge of
`5c30555f4399`), makes the SQLite silicon domain fault **deterministically** inside
`__capstone_cap_init`, before any workload runs. Bitstream
`caplifive_s06s08fix_s07tag2_618f4ce.bit`.

| build | domain `S7T` | result |
|---|---|---|
| as merged | `c85d4bfe4f22` | **wedged 2 of 2**, identical `mepc` both times |
| that ONE file reverted, nothing else | `5ad08b1dab3d` | **passed 3 of 3**, `obs=1460078339` |

One file, matched pair, deterministic on both sides. The only functional change in that file is

```c
static constexpr unsigned CapInitStoresPerBlock = 32;
```

which splits the generated initializer into a new basic block every 32 stores.

## The fault

```
mcause 0x9d  ->  seen, cause 29 = CAPABLITY_OUT_OF_BOUND   (not 25)
mepc   0x80939e3c ,  DBAS 0x80800000  ->  image VA 0x149e3c  =  __capstone_cap_init + 0x17a0
last domain marker: SQ: E/share   (dies during initialisation)
```

```
  149e1c: lui           a0, 0x1
  149e24: addi          a2, a0, -0x7b0     ; a2 = 0x850
  149e28: addi          a1, a0, -0x5e0     ; a1 = 0xa20
  149e2c: cincoffset    a1, gp, a1         ; &captable[162]   (0xa20 > 2047, so computed)
  149e30: ldc           a1, 0x0(a1)        ; the HOLDER capability
  149e34: cincoffset    a2, a1, a2         ; holder + 0x850   (also computed)
  149e38: ldc           a1, 0x6e0(gp)      ; the TARGET capability
  149e3c: stc           a1, 0x0(a2)        ; <== CAPABLITY_OUT_OF_BOUND
```

## What the change actually does to the code

Same source, two compilers, `__capstone_cap_init` diffed (`evidence/cap_init.diff`):

* reverted: **130 instructions, 0 branches** — one straight-line block.
* merged: **137 instructions, 1 branch** — plus a stack frame, two `stc` spills, and after the
  split the holder capabilities are **re-materialised from the cap table**
  (`ldc a0, 0x0(gp)`, `ldc a1, 0x10(gp)`) instead of staying live in registers.

That re-materialisation is the same shape as the faulting SQLite sequence, where the holder is
re-loaded from the table in a later block.

## What is NOT the cause — checked, so it is not re-checked

* **Not wrong sizes or offsets.** The faulting store fits its holder: descriptor slot 162 is
  2160 bytes and the store needs 0x850+16 = 2144. Computed by hand from
  `.capstone_gp_initdesc`, independent of any tool.
* **Not a tag-dropping spill.** The change's own comment predicts a spill emitted as `sd` and
  reloaded as `ldc`, which would give garbage bounds and explain cause 29. There are **no
  `sd`/`ld` at all** in `__capstone_cap_init` — 594 `stc`, 101 `ldc`. That part works.
* **Not carve exhaustion.** Globals total 63,920 bytes against 2,818,152 bytes of `dom_data`.
* **Not a descriptor/table numbering shift.** Record count and sizes are identical between the
  two builds.
* **Not visible under QEMU.** Every arm tried passes under emulation, including the SQLite
  domain. Emulation would not have caught this, so a lit test alone will not either.

## Two minimization attempts, both NEGATIVE — please do not repeat them

`src/` holds a standalone rung (no SQLite) with pointer-valued global initialisers.

| attempt | shape | silicon |
|---|---|---|
| v1 | 64 tightly packed pointers; offsets < 1024, all stores use a plain immediate | **both arms PASS** |
| v3 | 40 pointers padded 2064 bytes apart, so 40 stores use the register-materialised `cincoffset` form, 1 block split | **both arms PASS** |

Both were control-validated (`k800` = 4) and both arms were built from identical source,
differing only in that one compiler file. Confirmed in the disassembly rather than assumed: v3's
`cap_init` contains 40 register-form `cincoffset` and 1 branch.

**So the block split and the register-materialised store offset are, on their own, NOT
sufficient.** The one ingredient present in SQLite and absent from both attempts is a **cap-table
offset beyond the 12-bit immediate** — SQLite reaches the holder via `cincoffset a1, gp, a1` with
`0xa20`, which needs more than 128 globals. A rung cannot easily carry that: its `.text` must fit
in 4 KiB and ~130 initialisers plus the entry glue overflow it. That is the next thing to try, and
it needs a build path with a larger code window.

## How to reproduce today

The SQLite domain is the working reproducer.

```
cd capstone/benchmarks/sqlite
bash bake-sqlite-doms.sh S7T:SQLITE_S07_CURSOR_SELFTEST=1
cd ../../tests/rtl-smoke
export FPGA_URL=<FPGA-CONSOLE-URL>  FPGA_BITSTREAM=caplifive_s06s08fix_s07tag2_618f4ce.bit
export FPGA_FW=<...>/opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin
SQLITE_TRAPLOG_CLEAR=first \
SQLITE_STAGE_DOMS="/test-domains/S7T.dom,/test-domains/S7T.dom,/test-domains/S7T.dom" \
  python3 -m fpga_driver.run_sqlite_stages_fpga
```

Merged: no domain returns. Reverted: `SQ: obs=1460078339` three times.

## Suggested shape of a fix

Making the split **opt-in** — a flag whose default reproduces the pre-merge single-block
behaviour — restores silicon while keeping the code path available for the module that needed it.
This was deliberately NOT implemented here: a default-off flag *should* be codegen-identical to
the revert, and "should be equivalent" is the assumption that produced this report. Whoever fixes
it should verify against both silicon and the module the change was merged for.

## State of the branch

`CapstoneCapGlobalInit.cpp` is **reverted to its pre-merge content** on `capstone-bootstrap`, in
its own commit, so silicon work can continue. The other seven files from that merge are
untouched. **The revert removes the register-pressure bound the change added, so the module it
was merged for may regress** — that is the trade being made, and it is why this folder exists.
