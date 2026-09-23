# Memory safety of the FFmpeg domain, measured: three heap arms on QEMU

**Why this exists.** A reader of the M1–M5 result (`../2026-09-23-qemu-m1-m5/`) asked: do temporal
violations still fault in it? That result could not answer. It shows the decode is **correct**
under capability enforcement; it never tested **safety**. This folder tests it, on QEMU only.

**Verdict.**

1. **The M1–M5 run of record was not heap-safe.** It uses musl-capstone's `level0` allocator:
   - every heap pointer carries the bounds of the whole 1.5 MiB arena;
   - `free` only marks a block free.

   So an overflow into the neighbouring object, a use after free (including after the memory
   was reused), a use after `av_buffer_unref`, and a stale free that makes two live
   allocations alias **all silently succeeded**. Each was shown once, on 64-byte objects, as
   pre-registered.
2. **Global bounds are per merge group, not per object, on this ABI.** At -O1 and above, the
   compiler's GlobalMerge pass packs a translation unit's globals (up to 2,047 B) into one
   `.L_MergedGlobals` block, and `-capstone-shrink-globals` narrows to that block. Two statics
   in one file therefore overflow into each other on every arm (fixture 10), and a global is
   bounded exactly only if it sits last in its group, or unmerged (fixture 8).
   - **The gp-captable (silicon) ABI disables GlobalMerge for exactly this reason**
     (`llvm/lib/Target/Capstone/CapstoneTargetMachine.cpp:507-523`), so the port's M6 removes
     this.
   - **Stack:** a 64-byte stack array whose address escapes is bounded exactly (fixture 9).
3. **Per-object heap bounds (the `shrink` arm) change nothing in FFmpeg's output.** M1–M5 are
   bit-identical to native (30/30), and the control fires. The two heap-overflow fixtures now
   fault: one byte past a 64-byte object, and a write into the neighbour. Temporal errors still
   succeed.
4. **Per-object bounds plus revocation on free (the `sublet` arm):**
   - **Correctness:** M1–M5 are bit-identical to native (30/30), and the control fires.
   - **Temporal:** on QEMU every temporal fixture faults (4–7). That includes a read through a
     stale pointer while a live object occupies its address, and a stale `free`, which is
     caught inside `free` before anything is revoked.
   - **Spatial:** the two heap-overflow fixtures (2, 3) fault; the merged-globals one (10)
     still returns, as item 2 says.
   - **Cost:** the decode spends **1,262 revocation nodes**, 1.9% of the deployed bitstream's
     per-boot pool (65,532, never reclaimed: ISSUES R-12).
5. **The temporal faults are EMULATOR evidence, and the deployed silicon is documented to
   differ.**
   - **Mechanism here:** capstone-qemu's `ldc` untags a capability whose revocation node is
     dead, so the stale pointer faults at its next use. Every temporal fault here is exactly
     that: `cincoffset` or a load on an untagged operand, whose value is the freed object's
     address.
   - **Silicon:** the RTL forwards the loaded capability unchanged (ISSUES **Q-11**). On the
     FPGA, the nginx port's matching fixtures showed that **a data access through a stale
     pointer inside a domain retires**:
     - its use after destroy (s3) read the fill;
     - its reused read (s5) read the new occupant's 0x5B, the case this folder's fixture 5
       faults on under QEMU.

     Each was "unsafe-success" in all three repetitions
     (`docs/ref/fpga-silicon-measurements-for-paper.md` §7r, boots sw78 r1b1–r1b5 and
     repetitions).
   - **Inference, not measured:** on that silicon, `free`'s one-byte probe would retire too, and
     the stale free of fixture 7 would then revoke the **live** new owner's handle.
   - **Nothing in this folder is a claim about the board.** It could not be anyway: this ABI's
     `gp` is fabricated by QEMU (see the M1–M5 result, limit 1).

## Arms

The FFmpeg libraries are one shared build (`host/build-domain.sh`, `FFAPP_HEAP`). What differs
is the heap and its delivery.

| arm | allocator | heap pointer bounds | `free` |
|---|---|---|---|
| `level0` | `runtime/level0.c`, as every earlier run used it | the whole 1.5 MiB arena | marks the block free |
| `shrink` | the same file with `CAPSTONE_LEVEL0_SHRINK` | the requested size | marks the block free |
| `sublet` | `runtime/sublet_heap.c` | the requested size | **scrubs the object, revokes it**, then coalesces |

The `sublet` arm differs from the other two in four objects, all of them the heap's delivery:
- the allocator;
- `hostcall.o`, with `CAPSTONE_PROGRAM_REGIONS` to park the heap region;
- the guest host, which transfers a third, 4 MiB region;
- in the milestone images, the entry's heap-counter print.

**Attributing the temporal faults to revocation** rests on three pieces of evidence. The
level0/`shrink` arms by themselves are not a one-variable pair. Within the `sublet` arm:
1. Fixtures 1–3 keep their pointers tagged across the same print path, and fixtures 2 and 3
   fault as tagged bounds checks.
2. The 730 frees of M5 leave FFmpeg correct, so `free` does not untag unrelated live pointers.
3. In every temporal fault, the untagged value is the freed object's own address.

No arm has a "free a different object, then touch `p`" control.

`sublet_heap.c` in one paragraph:
- **Structure:** a binary buddy allocator over the transferred LINEAR region, following the
  Sublet recipe as SQLite's memsys5 patch applies it (a handle before every split, `take` on
  malloc, `give` on free).
- **Beyond that recipe:**
  - each alias is shrunk to the requested size;
  - `free` reads one byte through the pointer first (the stale-pointer probe), then zeroes the
    object through its own alias, then revokes;
  - the pool is the largest self-aligned power-of-two block inside the grant, 4 MiB here.

The level0 arm's milestone images are the run of record's, byte-identical: 7/7 OK against
`../2026-09-23-qemu-m1-m5/SHA256SUMS`, after every change in this folder.

## Fixtures (`src/capstone-domain/ffapp_safety.c`), and what each arm did

- **Predictions:** registered in `host/safety-expect.txt` before any fixture ran on any arm
  (20:03). Fixture 10's were added at 20:09, after fixture 8's first run and before fixture 10's
  own. No prediction line was edited afterwards. The file's header comment was extended at 20:34
  to describe the classifier change below. The file was not committed before the runs, so this
  history rests on the session record, not on an artifact.
- **Verdict:** `host/safety-verdict.py`, applied to each fixture's own section of the serial log.
- **One image per fixture:** a capability fault ends the emulator, so each boot carries at most
  one fault, and it runs last.
- **What counts as a fault:** one that comes after the fixture's `touch` line and names the
  **target address the fixture printed**. For a bounds fault that is the access address; for a
  temporal fault, the untagged value. A temporal fault with no printed target or no value is
  refused.

| # | fixture | level0 | shrink | sublet |
|---|---|---|---|---|
| 1 | two 64-byte `av_malloc`s; the pointer's capability length | returns; **len 1,572,816** (the arena) | returns; **len 64** | returns; **len 64** |
| 2 | write through `p` at `q`'s first byte, then read it via `q` | returns **0xee**: `q` overwritten | **OOB fault** at `q`'s address; `p`'s bounds are its own 64 B | **OOB fault** at `q`'s address; `p`'s bounds are its own 64 B |
| 3 | read `p[64]`, one past the end | returns **0x60**, the low byte of the next block header's `size` | **OOB fault** at `p+64` | **OOB fault** at `p+64` |
| 4 | read a freed object | returns **0xa0**, its old byte | returns **0xa0** | **temporal fault** at the touch; value = the freed address |
| 5 | free, reallocate the same size (**same address, on every arm**), read the old pointer | returns **0x5b**: the new occupant's byte | returns **0x5b** | **temporal fault** at the touch, while `q` holds `p`'s address |
| 6 | read an `AVBufferRef`'s data after `av_buffer_unref` | returns **0xa0** | returns **0xa0** | **temporal fault** at the touch |
| 7 | free `p`, let `q` take its memory, free `p` again, allocate `r` | returns: **`r` aliases the live `q`**, and `q` reads `r`'s 0x77 | returns: **the same aliasing** | **temporal fault inside `free`**: its probe `lbu zero, 0(a0)`, value = the printed target, before the scrub and any revoke |
| 8 | read one past a 64-byte global | **OOB fault**; bounds 80 B: the global plus a 16-byte static merged below it | **OOB fault**, the same 80 B | **OOB fault**, the same 80 B |
| 9 | read one past a 64-byte stack array | **OOB fault**; bounds exactly 64 B | **OOB fault**, 64 B | **OOB fault**, 64 B |
| 10 | two static 64-byte arrays; read the second through the first | returns **0x5b**: the first array's capability covers both (144 B merge group) | returns **0x5b** | returns **0x5b** |

**Rounds.** Round 1 ran every cell. The audits then changed:
- fixture 7, which now prints its target;
- the `sublet` heap: scrub on free, adaptive pool, and the grant stored before it is tested.

Round 2 re-ran every `sublet` cell and fixture 7 on all arms. The other level0 and `shrink`
cells are round 1's: their images were checked to be byte-identical between the rounds (hashes
against round 1's per-boot sidecars).

**Fault forms on capstone-qemu,** for reading the lines:
- **Bounds faults** print `Cap mem access OOB ... addr = <target>, bounds = (...)`, with cause 5
  for a load and cause 7 for a store.
- **Temporal faults (untagged operand, cause 24)** take one of two forms:
  - `cincoffset with an UNTAGGED rs1 ... val=<target>`, when the touch offsets the reloaded
    pointer first;
  - `Cap mem access requires capability ... value = <target>`, when it loads directly.

## FFmpeg M1–M5 on each arm (`host/run-qemu.sh all`, one boot per arm)

| arm | M1–M5 reached | frames vs native reference | flipped-input control | notes |
|---|---|---|---|---|
| level0 | yes | 30/30 MATCH | 30 frames, 10 changed | the run of record, `../2026-09-23-qemu-m1-m5/` |
| shrink | yes | **30/30 MATCH** | 30 frames, 10 changed | round 1; see the runner note below |
| sublet | yes | **30/30 MATCH** | 30 frames, 10 changed | round 2; M5 heap counters below |

The sublet arm's M5 heap counters (`FFAPP-HEAP`), identical in both rounds, i.e. with and without
the scrub and the adaptive pool:
- **Allocator:** alloc 730, free 730, merge 266, peak live 164.
- **Sublet primitives:** split 266, mrev 996, delin 730, revoke 996, init 266.
- **Revocation nodes:** **1,262**.
- **Node spend:** revocation nodes are split + mrev; silicon allocates nodes only in MREV and
  SPLIT.
- **Native comparison:** during development, an uncommitted native `--wrap` counter over the same
  decode gave 637 aligned allocations, 93 reallocs and a peak of 164 live objects. The heap's
  counts are consistent with that, but it is not an artifact, and the native build reaches
  `posix_memalign` where the domain reaches `malloc`.

## What this does not establish

1. **The board.** See verdict item 5.
2. **FFmpeg's own pools.** The `sublet` arm revokes on `free`. `AVBufferPool` and refstruct pools
   recycle objects **without** calling `free`, so a stale pointer into a pooled buffer that has
   been returned and reissued is not caught by this heap. That lifetime is the buffer-pool
   port's subject (`ports/ffmpeg/buffer-pool/`), not this one's.
3. **Every overflow, and every size.** The fixtures are directed: one byte past a 64-byte object,
   and one into its neighbour. They show the mechanism, not that FFmpeg itself contains no such
   bug.
   - **Objects of 4 KiB or more were not tested.** On silicon a stored capability of that size
     is rounded out to its granule (at most 1/512 of the size). The `sublet` arm's blocks are
     aligned to their size, so that rounding stays inside the block. The `shrink` arm's 16-byte
     bases would let it widen downward too.
   - **capstone-qemu does not round at all.** It keeps full-precision bounds for stored
     capabilities (`cap_mem_map.h`), so none of this rounding could have shown here.
4. **Repeat counts.** N = 1 per cell.

## Instrument notes

- **The classifier was changed twice.**
  1. **After the first `sublet` boot,** it learned capstone-qemu's "cincoffset with an UNTAGGED
     rs1" line. Fixture 4 had faulted in `ffapp_fix_touch` at `cincoffset a0, a0, a1`, with the
     freed object's address as the value, and the classifier called that "other". It counts as
     temporal only when the value equals the printed target; a doctored wrong value is refused.
  2. **After the audit,** it refuses a temporal fault that has no target or no value. Round 1's
     fixture 7 printed no target, so its address check was vacuous: a doctored `value = 0`,
     i.e. a NULL dereference, passed. Fixture 7 now prints its target, and it was re-run.
- **The verdict script was negative-tested before use,** with doctored logs: a fault before the
  touch, a fault at another address, a wrong mark, a wrong length, a missing section, a missing
  prediction, and an unattributed temporal fault.
- **Boot stalls.** In round 1, five boots stalled in the guest kernel before login: shrink
  fixture 9 twice, shrink M1–M5 once, sublet fixtures 8 and 9 once each.
  - **Reporting:** each was reported as "no section, the image never started", never as a
    result, and then retried.
  - **Evidence that no image was involved:** the shrink stalls' logs end at `Run /sbin/init`,
    before any domain image was loaded. The two sublet stalls' logs were overwritten by their
    retries, so that cannot be checked for them.
  - **Fix:** `run-safety.sh` now keeps every attempt. Round 2 had one stall (sublet fixture 9,
    first attempt). Its kept log shows no login and no fixture section, and the retry is the
    counted cell.
- **The shrink M1–M5 boot's runner died after the boot.** `run-qemu.sh` was edited to add the
  `sublet` arm while this job was executing it, and bash reads a script as it runs. So the
  boot's sections and oracle were extracted by hand, with the same code the runner uses.
  - **Image identity:** the M5 image was hash-compared with its booted share copy before that
    copy was overwritten (`b9778be6…`, still the current image). The others are identified by
    the kernel's `code size` line and the segment size.
- **Which allocator was in which image, checked in the instructions:**
  - The shrink arm's M5 image carries the same `malloc` (88 instructions) and `free` (58) as its
    fixture-1 image, which measured len 64. level0's are 84 and 45.
  - The sublet M5 image prints the heap's counters.
  - Two earlier checks of this were void: a `grep` that matched the directory name, and a range
    that ended at a local label.
- **Images:** `SHA256SUMS`. Every boot's copies are hash-checked against the build (per-boot
  `.sha256` sidecars written when the share is staged).
- **Not verified:** that the heap's own diagnostics (`sublet-heap: ...`, written to fd 2)
  reach the serial log. None appeared, and that absence is not evidence of anything.
