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
- The recipe is not a build from dev. Dev carries mremap (b439b7a267c3), but it cannot build the board images: it has
  no `CAPSTONE_VIRTUAL_MOVC_KEEP` switch (8f132305eea1, not on dev; dev's VirtualCodegen.cmake hard-codes
  `+movc-keeps-integer-source`, which the lead's C-32 decision keeps out of board images). Dev's runtime has also moved
  past the batch runtime: a 2 GiB region limit, 8,192 arenas and the madvise change.
- So the relink is the batch image with one change.
- **RETRACTED (2026-10-22): this list first also said dev lacks the integers-first slot offsets (5620badda510) and
  `CAPSTONE_SDK_EXTRA_CODEGEN` (5b3e6b7939af).** Both are on dev, as 232d13b220cc (the same heap.c change) and
  c66e04925d0e (the same builder change, with a corpus-SDK `-fno-builtin` gate on top). The check was commit
  ancestry, not content. The conclusion does not move, because the MOVC switch alone keeps dev from building the
  board images.

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

**Result** (2026-10-22 17:41-18:00, QEMU 708bbff5663f0e86, module 3325473537072a3a, one call per boot):

| call | result and site | high_water | allocated_total |
|---|---|---:|---:|
| lt1 02 case | cause 28 `memcpy` | 106,197 | 280,614 |
| lt1 02 control | exit 0 | 74,766 | 281,183 |
| lt1 10 case | cause 28 `l_divmod` | 80,115 | 219,402 |
| lt1 10 control | exit 0 | 80,104 | 220,865 |
| lt1 13 case | cause 28 `unicode_fromformat_write_str` | 106,274 | 281,516 |
| lt1 18 case | cause 28 `mmap_ass_subscript` | 66,716 | 196,527 |
| lt1 19 case | cause 25 `fnv` | 65,191 | 174,939 |
| lt1 19 probe | cause 25 `fnv` | 64,633 | 173,796 |
| lt1 19 control | exit 0 | 64,692 | 177,214 |
| lt1 23 case | cause 25 `memchr` | 75,000 | 267,453 |
| lt1 23 control | exit 0 | 72,893 | 264,737 |
| lt1 27 case | cause 25 `slot_nb_add` | 71,748 | 278,149 |
| lt1 27 control | exit 0 | 71,792 | 281,162 |
| lt1 28 case | cause 25 `_PyEval_Vector` | 76,490 | 313,843 |
| lt1 28 control | exit 0 | 76,575 | 316,941 |
| lt1 30 case | cause 25 `PyObject_RichCompare` | 76,523 | 314,758 |
| lt1 30 control | exit 0 | 76,539 | **316,375** (predicted 316,373) |
| lt1 31 case | cause 25 `odict_copy` | 74,605 | 233,845 |
| lt1 31 control | exit 0 | 82,356 | **348,190** (predicted 348,197) |
| lt0 18 case | cause 28 `mmap_ass_subscript` | 6,095 | 16,326 |

- **HELD:** every call's result class and fault site, capacity 262,144, no cause 30, no MemoryError, no SystemError,
  and every high_water inside its band. lt1 13 at 106,274 stays the largest demand.
- **HELD for 17 of the 19 exact counts. NOT HELD for two:** lt1 30 control (+2) and lt1 31 control (−7), both calls
  that complete. The pair below separates them.

### Bitstream 2 predictions: two exact counts missed, and the pair that separates the causes (pre-registered before its boots)

Two of the 19 exact allocated_total predictions missed, both on control calls that complete:
- lt1 30 control read 316,375 against 316,373 (+2);
- lt1 31 control read 348,190 against 348,197 (−7).
The paged kit read each of these calls in one boot only, so its count has never been repeated.

**The pair:** each of the two control calls, on domain B (b2twin2-q8), two boots on the batch image 916e14f9 and two on
the mremap image 8bcc40cf, interleaved.

**The reading:**
- The batch image reads the paged counts in both boots, and the mremap image reads today's counts in both: the relink
  moves these calls' allocations.
- Either image disagrees with itself across boots: the count is boot-nondeterministic for these calls, and the exact
  prediction was wrong to assume otherwise.

**Result** (2026-10-22 18:06-18:20, domain B, b2twin2-q8, eight boots interleaved):

| call | batch image 916e14f9 | mremap image 8bcc40cf (incl. the first reading) |
|---|---|---|
| lt1 30 control | 316,373; 316,373 | 316,375; 316,375; 316,375 |
| lt1 31 control | 348,187; 348,169 | 348,190; 348,180; 348,197 |

- **lt1 30: the relink.** Each image repeats its own count, and the batch image's is the paged kit's 316,373 exactly.
  The mremap relink moves this call by +2 allocations, deterministically.
- **lt1 31: boot nondeterminism.** Neither image repeats itself (a spread of 18 on the batch image, 17 on the mremap
  image). The exact prediction assumed a repeatable count that one paged boot never established.
- **For later pre-registrations:** a call that completes gets a band, not an exact count, until two boots agree. The
  17 exact matches stand, and they include every faulting case.

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

**Result** (2026-10-22 18:01-18:02, b2twin2-q8, domain A, one boot each):

| call | reading | high_water | allocated_total |
|---|---|---:|---:|
| as recorded (pymalloc) | `ERROR: test_non_str_kwarg` ... `MemoryError`, exit 1, no fault | 7,753 | 20,810 |
| PYTHONMALLOC=malloc | `ABR RAISED MemoryError` at startup, in `import pkgutil` (`_compile_bytecode`), exit 1, no fault | 20,829 | 36,233 |

- **As recorded: HELD.** It reproduces the reference, and allocated_total equals the paged kit's 20,810.
- **PYTHONMALLOC=malloc: NOT HELD, and VOID as a test of the defect.** The interpreter fails during runpy's import of
  pkgutil, before the test runs: no `test_non_str_kwarg` line appears. The pre-registration had no not-reached branch.
  It should have had one.
- **The leading hypothesis for the startup failure, not tested:** the batch runtime caps registered mappings per
  process at `CV_MAX_ARENAS` 512 (wire.h at 8f132305eea1; exec.c returns ENOSPC at the cap). Per-object malloc
  multiplies mallocng groups, each one a mapping. Dev raised the cap to 8,192 for exactly this case. The table is not
  the limit: 20,829 of 262,144 records.
- **So lt0 13 stays as the coverage file has it:** not capacity, mechanism from the source and the host build, not
  measured on the guest. It is not upgraded to "MISSED by design" on host evidence alone.

**Next instrument (pre-registered here, before its boots): a minimal trigger.** A one-statement script
(`trigger-min.py`: print `MIN BEGIN`, `import ast`, print `MIN IMPORTED`, then `ast.Name(**{object(): 'y'})` with
`MIN RAISED <type>` or `MIN RETURNED`) run directly, without runpy, unittest or test.support. One boot each, as recorded
and with PYTHONMALLOC=malloc, b2twin2-q8, the same image.
- **pymalloc:** `MIN RAISED MemoryError` (or TypeError, when the neighbouring bytes read as length 0), no fault.
- **malloc, if it reaches `MIN IMPORTED`:** cause 28 in `unicode_fromformat_write_str`.
- **malloc, if it dies before `MIN IMPORTED`:** the startup blocker is not the test's import chain, and the next
  instrument is an image whose runtime has the larger mapping cap. That is a second variable, and it would be named
  as one.

**Result** (2026-10-22 18:23, b2twin2-q8, domain A, one boot each):

| call | reading | high_water | allocated_total |
|---|---|---:|---:|
| minimal, pymalloc | `MIN BEGIN`, `MIN IMPORTED`, `MIN RAISED TypeError`; exit 0, no fault | 3,394 | 5,897 |
| minimal, PYTHONMALLOC=malloc | `MIN BEGIN`, then MemoryError inside `import ast` (no `MIN IMPORTED`); exit 1, no fault | 21,327 | 35,130 |

- **pymalloc: HELD, on its TypeError branch.** The statement runs, and the garbage length read past the object was
  small this time (the run 1 host check saw TypeError once in five). The outcome follows the neighbouring bytes,
  as on the host build.
- **malloc: the startup branch.** The interpreter dies importing `ast`, so per-object malloc fails before any test
  code, whatever runs it.
- **The startup cap is not the record table** (21,327 of 262,144). The candidate, `CV_MAX_ARENAS` 512, sits in two kit
  components: the launcher (`exec.c`, built into capstone-vexec) and the module (`capstone_vm.c`).
  - Testing it means a vexec and a module rebuilt with dev's 8,192, which is two kit components owned by the runtime
    lane. That is not done here.
  - **lt0 13 stays not measured on the guest.** Its mechanism comes from the source and the host build.

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

**Result** (2026-10-22 18:00, qkit7-batch2, wt-q15cw's runner 9b5e409c92e1fe66 at e2b26ea2c1e4, `--only 05`):

| build | defect-05 | verdict | reading |
|---|---|---|---|
| prebuilt | 287e4f41484a8572 | NO-READING:control-failed | FIXED run: cause 24 in `capstone_cap_revoke_child`, address 0 |
| read canary | bd4ec31316baa611 | **CAUGHT** | cause 28 at the labelled probe `write_probe`; FIXED run VERDICT FIXED, exit 0; all three controls fault |

- **HELD both ways.** The planted canary is what clobbers the next item's `sublet_parent`. With the byte only read,
  the fixed run is clean and the buggy run's one-byte overflow faults at the chunk's end.
- The edited case is not committed to the corpus: the row belongs to the corpus owner, and this diff
  is the handoff:

  ```diff
  -  memset((unsigned char *)next, 0x5a, 1);
  +  /* Read, not planted: under patch 0006 the next item's first bytes are its sublet_parent slot. */
     unsigned char before = *(volatile unsigned char *)next;
  -  CHECK(before == 0x5a, 756);
  ...
  -  o->damage = *(volatile unsigned char *)next != 0x5a;
  +  o->damage = *(volatile unsigned char *)next != before;
  ```

  The caveat stands: a read canary may already be 0, so on arms without a slot the NUL write would not
  show as damage. The corpus fix needs a sentinel that is not the slot.

### lt0 13: the launcher's mapping cap as the one variable (pre-registered 2026-10-22, before its boots)

**The pair,** built by the runtime lane:
- **Variant:** capstone-vexec `fad465ce4dea7121`, branch `runtime/vexec-arenas-8192` 32e18699a0a1. It is c14933904de4
  plus one line (`CV_MAX_ARENAS 512` → `8192`); the source diff was read here and is that one line.
- **Control:** `b27afefc09c6cae6`, the same build command on the unmodified tree, reproduced byte for byte.
- The module, 3325473537072a3a, already has 8,192, so the cap that bites is the launcher's.

**The kits:**
- qkit7-b2twin2-q8-vxb27 and -vxfad: qkit7-b2twin2-q8 (QEMU 708bbff5, module 33254735) with only
  `adapter/capstone-vexec` replaced.
- The kit's own launcher is a405840d, a third binary. So the control also shows whether the launcher swap alone moves
  anything.

**Already in hand, from the 18:23 minimal-trigger boot on the kit's own launcher:**
- the PYTHONMALLOC=malloc run's highest `CAPSTONE_VM_ADD_OK slot=` is **511**, the 512th slot, over 672 ADD_OK lines;
- the pymalloc run registers 355.

**The runs:** four boots on domain A, each launcher with both malloc instruments: the recorded call with
PYTHONMALLOC=malloc (`lt0-13-malloc`) and the minimal trigger (`lt0-13-min-malloc`). The highest ADD_OK slot and the
last ADD_OK or refusal line are read from each serial.

**Predicted:**
- **Control b27afefc:** both die at startup with MemoryError, as on a405840d. The highest slot is 511, and high_water
  is about 21k.
- **Variant fad465ce, if the cap is the cause:**
  - the minimal trigger reaches `MIN IMPORTED` and faults with cause 28 in `unicode_fromformat_write_str`;
  - the recorded call reaches `test_non_str_kwarg` and faults the same way;
  - the highest slot is above 511.
- **If the variant still dies at startup,** the cap is not the cause, and the last ADD_OK or refusal line is reported.
- **If the variant faults elsewhere first** (per-object bounds catching some other access in CPython's startup or
  unittest), that is read as it is, not as this defect.

**Result** (2026-10-22 18:56-18:58, domain A, one boot each): **every pre-registered reading holds.**

| launcher | call | reading | highest ADD_OK slot (lines) | high_water | allocated_total |
|---|---|---|---:|---:|---:|
| b27afefc (512) | recorded, malloc | `ABR RAISED MemoryError` at startup, exit 1 | 511 (695) | 20,829 | 36,244 |
| b27afefc (512) | minimal, malloc | `MIN BEGIN`, then MemoryError before `MIN IMPORTED` | 511 (672) | 21,332 | 35,107 |
| fad465ce (8192) | recorded, malloc | reaches `test_non_str_kwarg`, then cause 28 in `unicode_fromformat_write_str` | 1,765 (3,264) | 113,423 | 295,593 |
| fad465ce (8192) | minimal, malloc | `MIN IMPORTED`, then cause 28 in `unicode_fromformat_write_str` | 754 (992) | 31,677 | 55,411 |

- **The cap is the startup blocker, by the one variable.** The control stops at the 512th mapping on both
  instruments, as the kit's own launcher did. With 8,192 the interpreter starts and registers past 511.
- **lt0 13's mechanism is now measured on the guest.** Once each object carries its own bounds, the defect's read past
  the 16-byte `object()` faults with cause 28 at the declared site, in `unicode_fromformat_write_str`, as lt1 reads it.
  So the lt0 arm's recorded MemoryError is that same read, running uncaught inside a pymalloc arena's capability.
- **The cell's verdict is not changed here.** As configured (pymalloc), the arm ran the defect and did not catch it,
  which supports reading it MISSED rather than NO-READING:infra. That reclassification is the judge's and the lead's.
- **Reclassified MISSED by the project lead (relayed by the runtime lane, 2026-10-22),** on this pair. The runner's
  bundles keep NO-READING:infra until they are produced under a rule that tells this MemoryError (high_water 7,753 of
  65,536) from a capacity stop (64,0xx); the rule is the cpython runner's `CAPACITY` text match, which cannot.
- **Not a sizing reading:** 113,423 records is the recorded call under PYTHONMALLOC=malloc, an instrument
  configuration rather than any corpus arm's. The corpus maximum stays lt1 13's 106,274.

## Bitstream 2's QEMU kit carries the 8,192-mapping launcher: its regression (pre-registered 2026-10-22, before the runs)

**The decision:** the project lead approved it (relayed by the runtime lane): bitstream 2's board kit carries capstone-vexec
fad465ce4dea7121 (`runtime/vexec-arenas-8192` 32e18699a0a1) in place of the 512-cap launcher. The prediction kit
follows it, so predictions and board share the launcher.

**The kit, qkit7-b2twin3-q8:** qkit7-b2twin2-q8 with only `adapter/capstone-vexec` replaced (a405840d → fad465ce).
- A file-by-file sha256 diff of the two kits shows that one file and nothing else.
- QEMU is 708bbff5663f0e86, module 3325473537072a3a, firmware 6f2b082c, rootfs 9903242c.
- The runner's stamp (`b2-pred/run-pred.sh`) now prints the launcher's hash as well.

**The basis for "nothing moves":** the 20 bitstream-2 calls registered at most 402 mappings (the highest
`CAPSTONE_VM_ADD_OK slot=` across all their serials, lt1 13), against the old cap's 512. No call came near the cap
the launcher changes.

**The runs:**
1. **Tier 1, as a pair:** gen-probe/tier1-r60b.sh's recipe, `run.py` d5f0f399 from wt-mgp 64747600ecec and the batch
   Tier 1 images (contract 65f9f1f1, sqlite3 e63733dd, mruby f8594e11). It runs on qkit7-b2twin2-q8 and on
   qkit7-b2twin3-q8, and the launcher is the only difference. The summary parser was tested on the batch kit's real
   result.json: it reads 33/45, says NO DATA for a missing result, and reports DIFFER on an edited copy.
2. **The 20 calls** (the ten capacity cells, every call, and lt0 18) on qkit7-b2twin3-q8, one per boot, as at
   75b73a8c8862.

**Predicted:**
- **Tier 1:** the pair reads IDENTICAL test sets. Against the batch kit's 33/45 (12 failures listed in
  gen-probe/TIER1-r60), any difference belongs to the 18-bit table, not to the launcher.
- **The 20 calls,** as on qkit7-b2twin2-q8 call for call:
  - every result class, fault site and high_water band as before;
  - allocated_total exact for every call that faults, and for the lt1 02, 10, 19, 23, 27 and 28 controls, which
    matched exactly before;
  - lt1 30's control exact at 316,375 (it repeats on this image);
  - lt1 31's control in 348,160..348,210 (its boot-to-boot spread was 348,169..348,197);
  - lt1 18 at 196,527 and lt0 18 at 16,326 exact, each read once before;
  - every call's highest mapping slot at or below 402 + 10.
