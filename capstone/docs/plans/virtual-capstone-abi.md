# Virtual Capstone prototype ABI (M0)

Status: **M0 ABI draft**, 2026-10-06. This fixes the concrete interfaces that
the [prototype contract](virtual-capstone-prototype.md) left open: mode
control, the guest node table, privileged operations, the PCC and context
frame, delegation, exhaustion and the protected-U instruction set. It applies
to QEMU only. Nothing here is implemented unless a section says so. The
findings marked "today" cite the QEMU candidate at `2e6d0ff145`.

There is one version of this ABI. It has no version field, feature probe or
compatibility path; a change replaces it.

## Mode and root CSRs

| CSR | Number | Access | Contract |
|---|---|---|---|
| `scapctl` | `0x5C0` | S/M read-write | Bit 0 `PU`: protected U. All other bits read zero and ignore writes. `PU` applies only while the hart executes in U. It persists across traps; S code always runs scalar. Changing it takes effect at the next xRET into U. |
| `srevroot` | `0x5C1` | S/M read-write | Physical address of the selected node table, 4 KiB aligned; bits 11:0 read zero. Zero selects no table: every node ID is dead and every allocation raises cause 30. |
| `urevavail` | `0xCC0` | read-only, all modes | `capacity - next` of the selected table, zero without a table. Lets libc refuse an allocation before SPLIT or MREV traps. |

Both writable CSRs live in the standard custom supervisor read-write range,
and `urevavail` in the custom user read-only range. None of the three numbers
is used in the candidate. They replace the debug selectors 2 and 3 of
`csdebugoncapmem`, which become illegal outside M once this lands.

Linux installs `satp`, `srevroot` and `scapctl` for the next task before the
xRET into U. QEMU keeps no cached node state, so a root change needs no
explicit flush in the emulator.

**Finding, today:** the CSRs `cis`, `cid`, `cic` and `offsetmmu` at
`0x800`–`0x803` use the `any` predicate (`csr.c:4803-4809`), so U code can
access them. Protected U must receive illegal instruction for all four.

## Node table in guest memory

The table is pinned, physically contiguous guest RAM, reserved from Linux's
allocator and never mapped into U. All fields are little-endian.

```text
srevroot + 0      u64 capacity   number of 16-byte slots, header included
srevroot + 8      u64 next       next ID to issue, starts at 1
srevroot + 16*id  node record, 1 <= id < next

record + 0        u32 prev       0 = none
record + 4        u32 next       0 = none
record + 8        u32 depth
record + 12       u32 flags      bit 0 VALID, bit 1 LINEAR, others zero
```

Slot 0 is the header, so ID 0 is never a node and 0 serves as the list
terminator. A node is live only if `1 <= id < next`, `next <= capacity`,
`capacity <= 2^31` and `VALID` is set. A header violating these bounds makes
every ID dead and every allocation fail with cause 30. The capability's
existing 31-bit node field holds the ID; the encoding does not change.

Allocation reads `next`, requires `next < capacity`, writes the record, then
increments `next`. IDs are never reused within one table. There is no
reference count, free list or collector for this table; QEMU's host tree and
supervisor collector remain only on the legacy C-mode path.

Hardware writes to the table are ordinary physical stores; they clear any
physical tag on the touched granules. A table access outside RAM treats the
node as dead, and an allocation as exhausted.

## Privileged operations

All four use opcode `0x5b`, funct3 `001`, R-type, with funct7 values that
the candidate leaves free. They execute in S and M and are illegal in U.
Memory operands are kernel virtual addresses translated by the current
`satp`. Every operation completes its address translation and checks before
it changes the table, a slot or a register.

| Mnemonic | funct7 | Operands | Effect |
|---|---|---|---|
| `CSMINT` | `0x50` | `rd` ancestor ID, `rs1` slot address, `rs2` descriptor address | Descriptor `{u64 base; u64 end; u64 perms}`. Requires `base < end`, `perms <= 7` and bounds that survive compression unchanged, else cause 29. Requires two free IDs, else cause 30. Creates kernel ancestor `K` at depth 0 and child `R` at depth 1 below it. Writes a tagged LIN capability `{cursor=base, base, end, perms, node=R}` into the 16-byte-aligned slot. Writes `K` to `rd`. |
| `CSCHECKR` | `0x51` | `rd` result, `rs1` slot address, `rs2` length | Reads the slot without consuming it. Returns 0 when the slot holds a tagged capability whose node is live in the selected table, whose type is LIN or NONLIN, which grants read, and whose `[cursor, cursor+length)` lies in bounds without overflow. Otherwise returns the cause the access would raise: 24, 25, 26, 27 or 28. Length 0 requires `base <= cursor <= end`. |
| `CSCHECKW` | `0x52` | as above | Same check for write permission. |
| `CSRETIRE` | `0x53` | `rd` zero, `rs1` ancestor ID | Requires `rs1` to name a live node, else cause 29. Invalidates every strict descendant with the existing REVOKE walk, then clears the ancestor's `VALID`. Its result in `rd` is 0. |

`CSMINT` is the only way protected authority is created. The kernel calls it
for registered arenas and for the bootstrap code, data, stack and TLS roots,
never for an address supplied by U. It stores the ancestor ID in the arena's
kernel record as a scalar.

## PCC

| Event | Rule |
|---|---|
| Fetch in protected U | Requires a tagged PCC of type LIN or NONLIN with execute permission, a live node, and the whole instruction within bounds. Otherwise instruction access fault, cause 1, `tval` = pc. |
| Trap from protected U | The PCC moves into `cepc` with its cursor set to the trapping pc. `sepc` holds the same pc as a scalar. |
| xRET into U with `PU` set | The PCC is taken from `cepc` with its cursor replaced by `sepc` or `mepc`, and `cepc` becomes null. If `cepc` holds no capability, the first fetch faults with cause 1. |

The kernel never edits a PCC cursor; it edits `sepc` as usual, for instance
to step over `ecall`. The bounds check at the next fetch rejects a resumed pc
outside the saved authority.

**Finding, today:** QEMU checks no PCC on fetch in protected U.
`capstone_pre_mem_access` consults only CPMP, and the C-mode trap path that
writes `cepc` is skipped for protected U (`cpu_helper.c:1889`). Both rows are
M1 work.

## Context frame

Each protected task owns a kernel-only frame in its thread state, at a fixed
offset from the kernel `tp`, 16-byte aligned:

```c
struct capstone_frame { __uint128_t slot[32]; };   /* 512 bytes */
/* slot[0] = PCC, slot[i] = x_i for i = 1..31 */
```

The frame is reachable from `tp` because Linux saves the user `sp` before a
kernel stack exists. `pt_regs` stays scalar and unchanged.

**Entry from protected U:**

1. `ccsrrw tp, tp, cscratch` swaps the user `tp` into `cscratch` and the
   kernel `tp` out. In U, `cscratch` holds the kernel `tp` as a scalar; in S
   it holds zero, which marks a kernel-mode trap as in Linux today.
2. For `sp`, then every other GPR except `tp`: a scalar store of the register
   into its usual place, then `STC` into its frame slot. The scalar store
   comes first because STC consumes a linear source.
3. `ccsrrw t0, zero, cscratch` then scalar store and `STC` for the user `tp`.
4. `ccsrrw t0, zero, cepc` then `STC` into `slot[0]`.

**Return to protected U:**

1. A C helper compares the saved GPR values in `pt_regs` with the low words
   of slots 1--31. PCC in slot 0 is excluded: `sepc` changes, including the
   increment after `ecall`, are composed with its saved authority at xRET.
   On a mismatch it stores the scalar value into that low word. The scalar
   store clears the slot's tag, so a kernel edit cannot inherit authority.
   The syscall return path also clears the tag of `slot[10]` after storing
   the scalar result, even when `a0` happens to equal the old cursor.
2. `LDC t0, slot[0]` then `ccsrrw zero, t0, cepc`.
3. `mv t0, tp` then `ccsrrw zero, t0, cscratch` stores the kernel `tp` scalar
   for the next entry. Keep `tp` unchanged as the frame address until step 4.
   The candidate's CCSRRW can consume even an untagged scalar source; the
   temporary is disposable after publishing PCC in step 2.
4. `LDC` every GPR from its slot, with `tp` itself last. An untagged slot
   restores the scalar value. LDC in S consumes linear slots, so the frame
   holds no second copy afterwards.

S-mode STC and LDC already take scalar kernel addresses and use the physical
tag table in the candidate. S executes no other capability instruction
except `CCSRRW` and the four operations above. Signal frames are excluded,
so no other kernel path reads the frame.

## Delegation and machine entry

| Item | Contract |
|---|---|
| `medeleg` | Causes 0–7, 8, 12, 13, 15 and 24–30. |
| `mideleg` | SSIP, STIP, SEIP. Sstc supplies the supervisor timer. |
| Other M interrupts | Disabled on the fixed QEMU platform. One hart, so no machine IPIs. |
| Gate | An M-mode trap or interrupt taken while `PU` is set and the hart is in U, or between entry step 1 and 4, or between return step 2 and the xRET, fails the prototype gate. |

**Resolved in the prototype QEMU:** causes 24–30 are delegable. Before,
they were missing from `DELEGABLE_EXCPS`, so writing those `medeleg` bits
had no effect and every capability fault from U went to M. They stay out of
the VS mask. OpenSBI's own mask (`sbi_hart.c:194-205`) still needs the same
bits for the Linux gate in M2.

## Instruction set in protected U

| Class | Instructions | Behavior |
|---|---|---|
| Base | RV64IMAFDC loads, stores, atomics, branches, `ecall`, `ebreak`, `fence`, `fence.i` | Memory forms take a capability base and are checked; there is no scalar address fallback. |
| Capability | `MOVC`, `CINCOFFSET`, `CINCOFFSETIMM`, `SCC`, `LCC`, `SHRINK`, `SHRINKTO`, `SPLIT`, `TIGHTEN`, `DELIN`, `INIT`, `DROP`, `MREV`, `REVOKE`, `LDC`, `STC`, `CJALR`, `CBNZ` | Existing semantics against the selected table. |
| CSR | `urevavail` and the standard unprivileged counters | Readable. |
| Illegal | `SEAL`, `CCSRRW`, `CALL`, `RETURN`, `CAPENTER`, all debug and supervisor-extension encodings, the four privileged operations, vector instructions, and CSRs `0x800`–`0x803` | Illegal instruction, cause 2, delegated to S. |

## Linear transfers (Q-12)

| Instruction | Rule |
|---|---|
| LDC | After all address checks and the load, if the loaded value is tagged and not NONLIN, the slot is consumed: 16 zero bytes and no tag. This depends on the loaded value, not on `rd`, so `LDC` into `x0` destroys a linear slot. Consumption requires write permission from the address capability, else cause 27, and a writable PTE, else store page fault 15 with `tval` at the slot. |
| STC | After the store and any UNINIT cursor advance of `rs1`, if `rs2` is tagged and not NONLIN, `rs2` becomes null. |
| Physical identity | LDC reads bytes and tag from the same physical granule. Its write preflight must select that same granule; a disagreement raises store access fault 7 with `tval` at the virtual slot before any consumption or destination update. Linux must still perform normal `SFENCE.VMA` after PTE changes. |
| Faults | Any fault leaves `rd`, `rs1`, `rs2`, the slot bytes, its tag and any UNINIT cursor exactly as before. A retry after the kernel resolves the fault moves exactly once. |
| Order | Capability checks 24–28, then translation and PTE read or write, then the consumption checks, then commit. Page-table accessed and dirty bits are not rolled back. |

Implemented for protected U and the S-mode context path in the prototype
QEMU, with the gate in `tests/virtual-capstone-m1/`. Legacy C mode keeps the
divergence recorded as Q-12. The original 20-check gate failed 12 checks on
the unchanged base and rejected mutations of the write check and commit
order. The reviewed gate adds frame-base and physical-identity checks.
The combined [Q-12 acceptance runner](../../tests/trusted-linux-feasibility/run-q12.sh)
requires this gate, the existing U-access suite, and a real Linux process
through page-fault retry, syscall and task switch, plus a stripped-tag control.
Its bounded kernel patch saves the scalar `s2` cursor before consuming STC.

## Fault causes and `tval`

| Cause | Raised by | `tval` |
|---|---|---|
| 24–29 | A load, store, atomic, LDC or STC | The effective address: cursor plus offset, or the scalar value plus offset for an untagged base, as on the silicon LSU. |
| 24–29 | Any other instruction | 0 |
| 28 | An access past the bounds in protected U | The effective address. Legacy C mode keeps its access fault 5 or 7. |
| 30 | SPLIT, MREV or CSMINT | 0 |

A Capstone fault records its `tval` through one raise function with an
explicit argument. Every trap delivery consumes that value, so a fault raised
any other way reports 0 and never the address of an earlier trap. The value
does not pass through `badaddr`.

## Representability

A capability is representable when QEMU's compression followed by its
decompression returns base, end and cursor unchanged. The
[measurements](../history/06-10-2026_19-27-18_virtual-capstone-representability.md)
show that QEMU today creates values that fail this test and hides the loss
behind the bounds it stores beside each memory tag.

| Rule | Contract |
|---|---|
| Check at creation | `CSMINT`, `SHRINK`, `SHRINKTO`, `SPLIT` for both results, `CINCOFFSET`, `CINCOFFSETIMM` and `SCC` check every result before any change. A result that is not representable raises cause 29 with `tval` 0 and changes nothing. |
| No rounding | No operation rounds bounds inward or outward to make them fit. |
| Cursor inside the bounds | Moving the cursor within the bounds needs no separate check: STC through UNINIT, INIT, the cursor reset of REVOKE and SHRINK's clamp. This holds once the decoder overflow for regions of 1 GiB and more is fixed. |
| Decoder | `cap_uncompress` computes its window mask in 64 bits. The fix lands with `CSMINT`, gated by the in-bounds cursor measurement, which must then report no failure. |
| Side store | Protected tags drop the stored bounds only after every check above is in place, so a valid program sees no change. |

Consequences for software, measured on QEMU's encoding:

- An object shorter than 4096 bytes is representable at any address.
- An object of length in `[2^k, 2^(k+1))`, `12 <= k <= 29`, needs base and
  end aligned to `2^(k-9)`. libc rounds such allocations; Linux aligns the
  virtual address of a protected arena accordingly.
- The cursor may move about 2 KiB below the base and, for small objects,
  about 14 KiB above it; the span is four times the length class from 4096
  bytes up. Pointer arithmetic beyond that faults at the arithmetic, not at
  the access. M3 measures what this costs C programs.

## Exhaustion

`SPLIT`, `MREV` and `CSMINT` check for enough free IDs before any change and
raise cause 30, `tval` 0, delegated to S. libc reads `urevavail` before
carving and returns `NULL` from `malloc` instead. A cause-30 trap from U is
therefore a bug in libc; Linux terminates the task, since user signal
handlers are excluded.

## Retirement and teardown

`munmap` of a whole registered arena runs `CSRETIRE` on its ancestor, then
removes the mapping and scrubs the frames. Address and frame reuse may follow
only after `CSRETIRE` returns. On exit Linux selects another table or zero in
`srevroot`, after which no execution context can name this table, and frees
it. A freed table's memory is zeroed with scalar stores before reuse.

## M1 work this ABI implies in QEMU

1. Q-12 consumption and fault order for LDC and STC, U and S. **Done.**
2. Causes 24–30 in `DELEGABLE_EXCPS`. **Done.**
3. `scapctl`, `srevroot` and `urevavail`; retire debug selectors 2 and 3.
4. The guest table with the layout above; host tree and collector off for
   protected U.
5. `CSMINT`, `CSCHECKR`, `CSCHECKW` and `CSRETIRE`.
6. PCC fetch check, PCC move into `cepc` on trap and out on xRET.
7. Illegal-instruction policy for the protected-U table above.
8. Representability checks at creation, then bounds decoded from stored
   bits for protected tags.
9. `tval` for causes 24–30 and cause 28 for an access past the bounds. **Done.**

## Constraints for the remaining M1 changes

Agreed in review before the guest-table change.

- **Table choice by operation.** Protected-U accesses and the S context path
  use `srevroot`; legacy C mode and M keep the host tree. `CSMINT`, the two
  checks and `CSRETIRE` use the guest table in every privilege, M included,
  so a bootstrap cannot mint into the wrong tree. An xRET uses the table of
  the context it enters.
- **One list algorithm, two storage bindings.** Allocation, header, null ID,
  reference counts and release belong to the binding, not only record reads
  and writes. The guest binding never recycles an ID and never turns bad
  guest data into a host assertion. The forest model runs against both.
- **Complete preflight.** Before the first write, every range the operation
  will write is writable RAM and no address computation overflows. REVOKE
  and `CSRETIRE` walk the run first, with a visit bound against cycles, and
  only then invalidate. Translation is done once per contiguous chunk: the
  24-byte `CSMINT` descriptor can cross a page into an unrelated frame.
- **PCC.** QEMU reads instruction bytes while translating, before any helper
  it generates runs (`translate.c:1296`). The fetch rule therefore fixes the
  fault order, 16- and 32-bit instructions and page crossings, and requires
  the whole instruction inside the PCC bounds. `pc_cap` gets an explicit tag.
  A control revokes the PCC after its code was translated and runs it again.
- **Representability first.** Measure whether stored bits can widen bounds
  before `CSMINT` relies on them. An operation whose result is not exactly
  representable is rejected before any change; inward rounding would change
  SPLIT's partition and needs its own semantics. The check covers cursor
  changes too, such as CINCOFFSET followed by a store and load.
  `cap_in_bounds` (`cap.h:112`) computes `base + size` without an overflow
  check and must become overflow-safe.
- **Existing gates move with the ABI.** The bounded Linux patch still uses
  the debug selectors and debug mint. It is migrated with the selector
  removal, or its run is documented as a historical control.
- **Instruction policy** keeps the standard FP CSRs while RV64F and D stay
  supported.
- **Completion.** Protected mode is M1-complete only after one joint
  acceptance run of all remaining changes.
