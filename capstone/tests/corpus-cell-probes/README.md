# Probes of the corpus cells the batch bitstream does not read

Each item is pre-registered here and pushed before its run. The cells are listed in
docs/plans/2026-10-22-next-bitstream-whole-corpus-coverage.md, section (b).

## lt0 18 (gh-157335) with application mremap (pre-registered 2026-10-22, before its run)

**The blocker:** on the batch image, `mmap.resize` raises `OSError: [Errno 38]`, because the virtual profile has no
public mremap. llvm-capstone runtime/app-mremap c26b60a61f5b adds the MREMAP_MAYMOVE form as map-copy-unmap. The old
mapping is retired through the unmap service, and the new one is read-write.

**The image:** cpython-lt0 from t3-nomovc-r60, relinked through its own recorded link command with one archive member
replaced.
- The member is `mapping.c.obj` 4340ce84702a, rebuilt here from the branch's mapping.c with the batch SDK's own
  compile command. That command, on the SDK's source mapping.c, reproduces the original member ed0680b8081f.
- Every other member of lt0's runtime archive is unchanged. The relink with the ORIGINAL archive reproduces both batch
  images byte for byte: base 4512fdb6f43675a0, and `-dl` 23a2f0cbd80ddfbe.
- New images: base 1fb7dc2b59e09f46, `-dl` 7b871db70b7ec373. The defined-symbol name set equals the batch image's.

**The run:** qkit7-batch2, one boot. The lt0 arm's two block controls, then case 18 on the NEW image, then case 18 on
the batch image as the same-boot control.

**The prediction:**
- **Batch image:** `OSError: [Errno 38]` at `self.m.resize(new_size)`, as on the reference.
- **New image:** the resize succeeds. The 2-page map becomes 1 page, and the defect then writes index 8191, which was
  checked against the old size, through `mmap_ass_subscript`. It faults in `safe_byte_copy` or `mmap_ass_subscript`,
  the case's declared ASan sites:
  - cause 28 if the write goes through the new 1-page mapping's capability;
  - cause 25 if it goes through a pointer to the retired 2-page mapping.
  So the verdict is CAUGHT; never OSError 38, and never completion.
- **Both controls fault.**

**Result** (2026-10-22 15:14, qkit7-batch2, QEMU domain A, one boot):

| call | image | reading |
|---|---|---|
| control-uaf-block | batch 23a2f0cb | `CONTROL uaf-block RETURNED`, exit 0 |
| control-bounds-block | batch 23a2f0cb | `CONTROL bounds-block RETURNED`, exit 0 |
| 18 | mremap 7b871db7 | domain fault cause 28 in `mmap_ass_subscript`, address 0x3f802f5fff: offset 8191 into a page-aligned one-page mapping |
| 18 | batch 23a2f0cb | `OSError: [Errno 38] Function not implemented`, ABR EXIT 1 |

- **HELD:**
  - with application mremap, the resize succeeds and the defect's write faults at a declared site (`mmap_ass_subscript`);
  - it is the cause-28 branch: the write goes through the new mapping's capability, one page short of index 8191;
  - the same-boot batch image raises OSError 38.
  - So lt0 18 reads CAUGHT once mremap is in the image, and its batch MISSED is a not-reached.
- **NOT HELD: "both controls fault."** That line was my error. The lt0 arm's pymalloc-block controls COMPLETE by design
  (the virtual-malloc arm keeps pymalloc's arenas). The batch reference's lt0 verdict rows record both controls as
  `observed: complete`, and the judge accepts it. They read the same here. The case reading does not depend on them.

## lt1 18 (gh-157335) is record capacity (pre-registered 2026-10-22, before its runs)

**The reading that prompted this:** on qkit7-batch2, standalone probes import everything the case imports, from the
case's own directory, and succeed: `select`, `subprocess`, `test.support.script_helper`. Their module high_water
peaks at 59,646. In the same boot, case 18 raises `SystemError` at `import select` with high_water **64,023** against
capacity 65,536, which is the flat table's stop. The probes' run is /mnt/hdd/<user>/capstone/sel18,
readings.json per call. The case also compiles the large test_mmap module before those imports. So this run tests
whether the SystemError is the record limit surfacing as an allocation that fails without setting an exception.

**The runs:** case 18 on each of three kits, through run-gate.sh (capstone-qemu 98b34967b6's runner):
- qkit7-b2twin-q6, the 16-bit table;
- qkit7-b2twin-q8, the 18-bit table;
- qkit7-batch2-paged-lp, uncapped, with the live-peak counters.

**The prediction:**
- **q6:** `SystemError: error return without exception set` at subprocess.py:125, with high_water within 300 of
  64,023, as on the batch kit.
- **q8 and paged-lp:** no SystemError. `import select` succeeds and the test reaches `self.m.resize(new_size)`,
  where this image (no application mremap) raises `OSError: [Errno 38]`. Exit 1.
  - On q8, high_water is above 64,023.
  - paged-lp gives the case's live peak as a bracket. Only "above 64,001" is predicted.
- **If q8 still reads SystemError,** the record limit is not the cause, and this item is withdrawn.

**Result** (2026-10-22 15:14-15:24, one boot per kit):

| kit (module, capacity) | reading | module high_water | QEMU live peak |
|---|---|---|---:|
| b2twin-q6 (69d48252, 65,536) | `SystemError` (2 lines), exit 1 | 64,016 | n/a |
| b2twin-q8 (33254735, 262,144) | no SystemError; `OSError: [Errno 38]` at the resize, exit 1 | 67,935 | n/a |
| batch2-paged-lp (ed856572, paged) | no SystemError; `OSError: [Errno 38]`, exit 1 | (paged count) | 67,875..68,607 |

**Every pre-registered reading holds.** lt1 18 is a record-capacity cell, the tenth: the 64,001-record flat table
refuses an allocation during the test's imports, and CPython reports it as `SystemError: error return without
exception set`, not as MemoryError. Its live peak up to the mremap blocker is 67,875..68,607. With mremap in the
image, the defect runs further, so that is a lower bound on the cell's demand. It is far below lt1 13's 106,272, so
the index sizing is unchanged.

## sqlite/capi-repros on the virtual profile: exploratory (2026-10-22; no verdict rests on it)

This sizes the port before the lead's capi decisions. It is not an arm: there is no pre-registration, no
attribution to declared sites and no judge. The SDK is the batch SDK, t3-freeze-nb-nomovc-cw-r60/sdk
(capstone-cc 7f3077ae), and the kit is qkit7-batch2. Each row's `before.c` is built `-O0 -g` against SQLite
**3.53.3**, the host ASan arm's amalgamation, compiled `-O1 -g -DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION`,
the host arm's flags. The postgres c-repros malloc controls run beside the rows.

1. **The amalgamation as shipped does not run.** 17 of 18 rows fault with cause 6, a misaligned capability store, at
   openDatabase's `db->aDb = db->aDbStatic` (sqlite3.c:190805, `stc a1, 0x40(s2)`).
   - sqlite3MemMalloc prepends an 8-byte size header unless SQLITE_MALLOCSIZE is available. So every SQLite object
     is only 8-aligned.
   - row14 faults first, in its own `main`, with cause 24.
2. **With ports/sqlite/build-sqlite-capstone.sh's six 3.53.3 rewrites** (the 16-byte saveBuf and SZ_VDBECURSOR
   alignment among them) **plus `-DHAVE_MALLOC_H=1 -DHAVE_MALLOC_USABLE_SIZE=1`**, SQLite calls malloc directly with
   no header. The runtime exports malloc_usable_size. Every row then runs:

| oracle (host ASan) | rows | virtual profile, qkit7-batch2 |
|---|---|---|
| heap-use-after-free | 1-10, 16 | all 11 fault with cause 25 (revoked): in `main` 1, 3-5, 7-10; in `read_context` 2; in `stale_udf` 6; in `authorize` 16 |
| bad-free | 11 | cause 2 in `cap_malloc_invalid_free`, the allocator refusing the free |
| null-deref | 12, 13, 14 | cause 24 in `main` for 12 and 14; cause 12 at an unsymbolized pc for 13, a fetch from a non-code address |
| none | 17, 18, 19 | exit 0 |
| (no before.c) | 15 | not built |

Controls: bounds-malloc faults with cause 28 and uaf-malloc with cause 25.

What it settles for the lead's decisions:
- (a) the version: 3.53.3 runs with the existing rewrite pass plus the usable-size defines, so the host arm's version
  can be kept;
- (b) the controls: the generic malloc controls behave;
- (c) the oracles: each one has a distinct fault shape on this profile.

What it does not settle:
- whether each fault is at the row's declared stale access;
- how lookaside (on by default here) changes which rows read 25;
- what a FIXED-twin control would be.
