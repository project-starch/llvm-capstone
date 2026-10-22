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
image, the defect runs further, so that is a lower bound on the cell's demand. (RETRACTED in the next section: it
is not a lower bound. The full run peaks lower, at 66,682..67,583.) It is far below lt1 13's 106,272, so
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

## xlang/repro and xlang/lua-cdp on the virtual profile: exploratory (2026-10-22; no verdict rests on it)

The shims are the CHERI column's hosted programs. Each one is built unchanged, `capstone-cc -O0 -g` on the batch SDK
(t3-freeze-nb-nomovc-cw-r60/sdk): shim plus `cheri/mock-mruby/mock_mruby.c` (repro), or shim plus
`lua-cdp/cheri/mock_report.c` (lua-cdp). The row lists come from each rows.tsv. All 28 build, and they ran in one
boot on qkit7-batch2 with the postgres c-repros malloc controls (bounds 28, uaf 25: both behave).

| set | rows | virtual profile | the classic Capstone column's revoke build (RESULTS.md, expected-results.tsv) |
|---|---|---|---|
| repro, temporal UAF | 12 (libpulse, rlua userdata, secp256k1, nine mruby) | all fault with cause 25 | 12 FAULT by revocation |
| repro, spatial | mruby_bytecode_overflow, mruby_upvar_overflow | cause 28 | FAULT by bounds (both) |
| repro, stack use-after-return | rlua_escaped_handle_uar | exit 0 | MISS (no allocator involved) |
| lua-cdp | 13 | all fault with cause 25 | 13 FAULT by revocation |

Each of the 28 rows reads the classic revoke column's outcome class.

What it does not give:
- the classic column's attribution control. The "control" build is the same program minus the one REVOKE, and
  virtual mallocng has no such switch. That is the lead's decision (a) in the plan;
- site attribution per row.

## lt1 18's full demand, with application mremap (pre-registered 2026-10-22, before its runs)

**The image:** cpython-lt1 relinked as lt0 was (t3-mremap/relink.sh, ARM=lt1), with `mapping.c.obj` 4340ce84 (the
mremap now on dev at b439b7a267c3). The control relink with the original archive reproduces 74f5c610bee6f468 and
916e14f9b3f622c9 byte for byte. New images: base c3891354cf7fafc9, `-dl` 8bcc40cf13731e12.

**The runs:** one boot each on qkit7-b2twin-q8 (the 18-bit table) and qkit7-batch2-paged-lp (uncapped, live-peak
counters). Each boot runs case 18 on the new image, then case 18 on the batch image as the same-boot control.

**The prediction:**
- **Batch image:** `OSError: [Errno 38]` at the resize, as in 588cfd615ff6.
- **New image:** the resize succeeds, and the defect's write faults in `mmap_ass_subscript` or `safe_byte_copy`,
  with cause 28 (as lt0 read) or 25.
  - On q8, high_water is at least 64,016 and no SystemError appears.
  - On paged-lp, the live peak is between 67,875 and 70,000: the work after the resize is one item assignment.
- **The sizing maximum stays lt1 13's 106,272.**

**Result** (2026-10-22 15:37, one boot per kit):

| kit | image | reading | module high_water | QEMU live peak |
|---|---|---|---:|---:|
| b2twin-q8 | mremap 8bcc40cf | cause 28 in `mmap_ass_subscript` | 66,724 | n/a |
| b2twin-q8 | batch 916e14f9 | `OSError: [Errno 38]`, exit 1 | 67,926 | n/a |
| batch2-paged-lp | mremap 8bcc40cf | cause 28 in `mmap_ass_subscript` | (paged count) | 66,682..67,583 |
| batch2-paged-lp | batch 916e14f9 | `OSError: [Errno 38]`, exit 1 | (paged count) | 67,886..68,607 |

- **HELD:** with mremap the defect runs and faults at a declared site with cause 28, on both kits. The batch image
  raises OSError 38. There is no SystemError on the 18-bit table.
- **NOT HELD: the peak band [67,875, 70,000].** The full run peaks at 66,682..67,583, below it.
  - The band rested on my claim that the stopped run's peak was a LOWER bound on the cell's demand. It is not. The
    batch image's failed resize takes CPython's error path (the exception, its traceback, unittest's error report),
    which allocates more than the defect path does. Its 67,886..68,607 is that path's peak, not a part of the cell's.
  - That claim reached dev at edcc82c79912 and is retracted there.
- **lt1 18's demand is 66,682..67,583 live records.** The maximum stays lt1 13's 106,272, so the sizing is unchanged.

## Bitstream 2 predictions: the capacity cells on the RTL-pinned twin (pre-registered 2026-10-22, before the runs)

**What it predicts:** what the corpus's ten record-capacity cells (all lt1) and lt0 18 will read on bitstream 2
(capstone-ariane `vp3-index18` 5d365d9a6, an 18-bit index on a flat table).

**The kit:** qkit7-b2twin2-q8.
- QEMU 708bbff5663f0e86: capstone-qemu `virtual/flat-index-18` 46b69c03a4, pinned to the RTL at d44c9441b.
- Module 3325473537072a3a: capacity 262,144.
- Everything else is qkit7-batch2's.

**The images:** the batch recipe plus application mremap, and nothing else.
- lt1 uses 8bcc40cf13731e12 and lt0 uses 7b871db70b7ec373, both relinked above.
- The recipe is not a build from dev. Dev carries mremap (b439b7a267c3) but not the board recipe:
  - `CAPSTONE_VIRTUAL_MOVC_KEEP=OFF` (8f132305eea1);
  - the integers-first slot offsets (5620badda510);
  - `CAPSTONE_SDK_EXTRA_CODEGEN` (5b3e6b7939af).
  Dev's runtime has also moved past the batch runtime: a 2 GiB region limit, 8,192 arenas and the madvise change.
- So the relink is the batch image with one change.

**The runs:**
- Every call of each cell (the case and its control or probe) gets its own boot, from the batch-refs-r60 session with
  the image swapped. That is 20 boots on QEMU domains A and B.
- Module stats accumulate over a boot, so one call per boot gives each call its own high_water and allocated_total.
- Runner: `b2-pred/run-pred.sh`. Its parser was tested before any boot:
  - it reads the gate's lt1 13 output correctly;
  - it reports NO DATA, exit 2, on a two-call output and on a missing file.

**The prediction, per call.** The reference is the paged kit's row in `capstone/tests/live-peak-sizing/cells.tsv`.
- **Result class and fault site:** the same as the reference, with no cause 30 and no MemoryError in serial.
  capacity reads 262,144.
- **high_water:** inside the reference's live-peak bracket, widened by 300 either side. The gate read 106,271 and
  106,197, one below the brackets' lower ends.
- **allocated_total:** EXACTLY the reference's module count. The gate reproduced 281,516 and 280,614 on the batch image.
  A difference means the call reaches the new mremap or the relinked layout matters; that cell's allocations are then
  read again, and the result class still decides.
- **lt1 18:** cause 28 in `mmap_ass_subscript`, high_water 66,424..67,024 (66,724 on the first twin, ±300). No
  SystemError, and no exact allocation count is predicted.
- **lt0 18:** cause 28 in `mmap_ass_subscript`, as on the batch kit; lt0 is not capacity-bound, so high_water stays
  below 65,536.

| call | result and site | high_water band | allocated_total |
|---|---|---:|---:|
| lt1 02 case | cause 28 `memcpy` | 105,898..106,795 | 280,614 |
| lt1 02 control | exit 0 | 74,466..76,075 | 281,183 |
| lt1 10 case | cause 28 `l_divmod` | 79,783..81,195 | 219,402 |
| lt1 10 control | exit 0 | 79,790..81,195 | 220,865 |
| lt1 13 case | cause 28 `unicode_fromformat_write_str` | 105,972..106,795 | 281,516 |
| lt1 18 case | cause 28 `mmap_ass_subscript` | 66,424..67,024 | n/a |
| lt1 19 case | cause 25 `fnv` | 64,877..65,835 | 174,939 |
| lt1 19 probe | cause 25 `fnv` | 64,303..65,835 | 173,796 |
| lt1 19 control | exit 0 | 64,390..65,835 | 177,214 |
| lt1 23 case | cause 25 `memchr` | 74,672..76,075 | 267,453 |
| lt1 23 control | exit 0 | 72,569..74,027 | 264,737 |
| lt1 27 case | cause 25 `slot_nb_add` | 71,393..73,003 | 278,149 |
| lt1 27 control | exit 0 | 71,465..73,003 | 281,162 |
| lt1 28 case | cause 25 `_PyEval_Vector` | 76,206..77,099 | 313,843 |
| lt1 28 control | exit 0 | 76,262..77,099 | 316,941 |
| lt1 30 case | cause 25 `PyObject_RichCompare` | 76,229..77,099 | 314,758 |
| lt1 30 control | exit 0 | 76,232..77,099 | 316,373 |
| lt1 31 case | cause 25 `odict_copy` | 74,313..75,051 | 233,845 |
| lt1 31 control | exit 0 | 82,064..83,243 | 348,197 |
| lt0 18 case | cause 28 `mmap_ass_subscript` | below 65,536 | n/a |

**What the board will add.** QEMU is the twin, not the silicon. On bitstream 2 a row predicts the board when the
recipe's other gates hold, and the board batch carries the same images. A board row that differs from its twin row is a
twin defect or a silicon defect, not a capacity reading.

## lt0 13 (gh-144169): the MemoryError is the defect's own read, inside a pymalloc arena (pre-registered 2026-10-22, before the guest pair)

**What the coverage file says now:** lt0 13 is NOT capacity; the likely reading was not measured.

**The defect, from the source (CPython 3.13.7):**
- `ast_type_init` (Python/Python-ast.c:5160) formats a non-str key with `%U`.
- `unicode_fromformat_write_str` (Objects/unicodeobject.c:2561) then reads `PyUnicode_GET_LENGTH(str)` at offset 16 of an
  `object()`, which is a 16-byte object. That read lands in the next 16-byte block, and its value becomes the length.
- On lt1 every object has its own bounds, so the read faults at once: cause 28 in `unicode_fromformat_write_str`.

**Host check, already run:** the 3.13.7 native build that the guest build bootstraps from (`build-python/python`), on
the case's statement, five times per allocator. The outcome follows the bytes past the object:
- PYTHONMALLOC=pymalloc: MemoryError 4 times out of 5, and TypeError once, after a warning naming `''`;
- PYTHONMALLOC=malloc: 5 TypeErrors; 4 of the 5 warnings print neighbouring heap bytes as the key;
- the debug allocators: the process dumps core.
So under pymalloc the garbage length is most often huge, and the write of that length raises MemoryError. This is the
defect's out-of-bounds read running uncaught, not an infrastructure stop.

**Guest pair, one boot each on qkit7-b2twin2-q8 (the lt0 batch image 23a2f0cbd80ddfbe, the reference's call):**
- **As recorded:** MemoryError (`ABR EXIT 1`), no domain fault, high_water well below 65,536.
- **With PYTHONMALLOC=malloc added, the one change:** every `object()` is a virtual-malloc allocation with its own
  capability. The read past it then faults with a bounds cause (28) in `unicode_fromformat_write_str`, as lt1 does,
  with no MemoryError.
  - If it does not fault, virtual-malloc's bounds for a 16-byte request cover bytes 16..23. That names the bound width,
    not the defect, as the variable.
  - The run is on the 18-bit table because per-object allocation multiplies records. high_water is recorded and no
    band is predicted.
- **Verdict if both hold:** lt0 13 is a MISSED by the arm's design. lt0 keeps pymalloc's arenas, so a 16-byte read past
  a 16-byte object stays inside the arena's capability. It is not a NO-READING:infra.

## pb13q15 05: the canary, rebuilt to read instead of write (pre-registered 2026-10-22, before its runs)

**The hypothesis (the coverage file's row):** the case's own `memset(next, 0x5a, 1)` writes byte 0 of the NEXT item.
Under patch 0006 that byte is the item's `sublet_parent` capability, so the next `slabs_free` faults: cause 24 in
`capstone_cap_revoke_child`. That is the FIXED run's failure.

**The binaries:**
- Built with the b13 q15 prebuilt's recorded recipe: capstone-cc from t3-freeze-nb-nomovc-cw-r60, the case flags from
  build-cases.sh's capstone-application branch, linked against b13's own `libmemcached-allocators.a` and driver-virtual.c.
- **Control:** the unmodified case.c rebuilt this way equals the prebuilt `defect-05` in .text, .rodata and .data. The
  edited build differs in all three.
- **Edited:** `before` is read from `next` rather than planted, and `damage` compares against it. The source diff is
  three lines; the binary is `defect-05` bd4ec31316baa611 in a hardlinked copy of the b13 prebuilt
  (`pb05-canary/pre-readcanary`).

**The runs:** one boot each, wt-q15cw's run-virtual-cases.py with --only 05, on qkit7-batch2 (the reference's kit and
arguments), for the prebuilt and for the edited copy.

**The prediction:**
- **Prebuilt:** as the reference, NO-READING:control-failed: the FIXED run faults with cause 24 in
  `capstone_cap_revoke_child`, and the buggy run reads cause 28 at the chunk's end.
- **Edited:** the FIXED run prints VERDICT FIXED and exits 0, and the buggy run still faults with cause 28 at the chunk's
  end. The cell reads CAUGHT.
- **Caveat for the corpus owner:** on arms without a Sublet slot, a read canary can be 0 already, so the buggy run's NUL
  write would not show as `damage`. The corpus fix needs a nonzero sentinel that is not the slot, or a separate guard
  byte. This test does not settle that, only the slot clobber.
