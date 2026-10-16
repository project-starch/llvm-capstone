# spellfixoom -- fix-2026-07-26

Upstream fix `2026-07-26`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row24 / sqlite-d9305a65c876 -- spellfix editDist3Install() registers editdist3()
 * three times sharing ONE heap EditDist3Config, and only the arity-1 registration
 * carries the editDist3ConfigDelete destructor. The structure is:
 *
 *     pConfig = sqlite3_malloc64(sizeof(*pConfig));                       // (1)
 *     rc = create_function_v2("editdist3", 2, ..., pConfig, ..., 0);       // (2)
 *     if( rc==SQLITE_OK ){ rc = create_function_v2("...", 3, ..., pConfig, ..., 0); }  // (3)
 *     if( rc==SQLITE_OK ){ rc = create_function_v2("...", 1, ..., pConfig, ...,
 *                                                  editDist3ConfigDelete); }          // (4)
 *     }else{ sqlite3_free(pConfig); }
 *
 * NOTE the else belongs to the THIRD if, so it fires when (2) or (3) failed:
 *   - (3) fails after (2) succeeded -> pConfig is FREED while editdist3/2 is still
 *     registered on it. Calling editdist3(a,b) then runs editDist3SqlFunc ->
 *     editDist3FindLang over the freed config. THIS is the use-after-free.
 *   - (2) fails -> nothing got registered, so no UAF.
 *   - (4) fails -> the else is NOT taken, so pConfig merely leaks (no destructor).
 * There is no refcount, hence the bug. Trunk-only fix. Ext: ext/misc/spellfix.c.
 *
 * SQLITE_UNTESTABLE removes sqlite3_test_control, so the only OOM lever is
 * exhausting the memsys5 arena. editDist3Install is the LAST step of
 * spellfix1Register, so everything before it can be allowed to succeed.
 *
 * Rather than guess the reserve size across many QEMU boots, this domain SWEEPS it
 * itself in one boot: for each k it drains the arena to ~k*64 free bytes, runs the
 * init, then releases the arena and asks whether editdist3/2 is still callable.
 * rc==SQLITE_NOMEM together with a callable editdist3/2 IS the bug state, and the
 * domain then calls it to perform the freed-config read.
 * NOTE: -DSQLITE_DQS=0 -> SQL string literals must be single-quoted.
```

## 2026-10-11: the trigger is the CheriBSD copy

`case.c` is now byte-identical to `ports/sqlite/cheribsd/cases/` (the copy the CheriBSD probe runs used).
The source above is the earlier trigger. Why it was replaced: the corpus copy forced OOM by filling the heap with blocks; host ASan was silent on it. The CheriBSD copy sweeps an allocation-failure point through a wrapped allocator until spellfix init returns NOMEM with editdist3 still registered, then calls it; host ASan (the wrapper on the system allocator) reports the heap-use-after-free in editDist3FindLang. Evidence:
`results/2026-10-11-host-asan/results.json`, from `shared/run-host-asan.py` (every SQLite object its own
malloc, lookaside off).

One change against the CheriBSD copy, made the same day: `run_case` turns the lookaside off
(`SQLITE_CONFIG_LOOKASIDE, 0, 0`) before `sqlite3_initialize`. A lookaside slot is handed out
without calling the allocator, so the counting wrapper cannot fail it. With the lookaside on, 8 of
spellfix init's 16 allocations reached the wrapper on Capstone and the sweep printed `NO BUG STATE
FOUND`; host ASan, built with the lookaside off, found the bug state at the 14th.

## Review 2026-10-16 -- the three oracles agree on the access, and two of the three detections are not

### The defect, re-read in the pinned source

`editDist3Install` in `ext/misc/spellfix.c`, line for line:

```c
  EditDist3Config *pConfig = sqlite3_malloc64( sizeof(*pConfig) );
  memset(pConfig, 0, sizeof(*pConfig));
  rc = sqlite3_create_function_v2(db, "editdist3", 2, ..., pConfig, ..., 0);        /* (2) */
  if( rc==SQLITE_OK ){
    rc = sqlite3_create_function_v2(db, "editdist3", 3, ..., pConfig, ..., 0);      /* (3) */
  }
  if( rc==SQLITE_OK ){
    rc = sqlite3_create_function_v2(db, "editdist3", 1, ..., pConfig, ...,
                                    editDist3ConfigDelete);                        /* (4) */
  }else{
    sqlite3_free(pConfig);
  }
```

The `else` belongs to the **third** `if`, so it fires when (2) or (3) failed. (3) failing after
(2) succeeded frees `pConfig` while `editdist3/2` is still registered on it. No refcount. The
case's own output confirms it reaches exactly that state, on every arm:

```
spellfixoom wrapper active=1
spellfixoom unarmed init rc=0 allocs=16 callable=1
spellfixoom BUG STATE at failAt=14 rc=NOMEM callable=1 (init allocs=14)
```

and probe site 57 prints `rc=7` (SQLITE_NOMEM) at the free, which is (3) having failed.

**The upstream reference cannot be checked here.** It is given two ways -- `fix-2026-07-26` in
`upstream_fix` and the directory name, `d9305a65c876` in `live_proof` and the case header -- and
neither appears anywhere outside this corpus's own files. This environment is offline, so that
is recorded rather than resolved. Unlike case 09 the reference does not carry the claim: the
defect is established by the source above, and all three oracles agree on which access it is.

### The function reads the freed config three times, and the arms fault on different ones

```c
static const EditDist3Lang *editDist3FindLang(EditDist3Config *pConfig, int iLang){
  for(i=0; i<pConfig->nLang; i++){            /* (1) offset 0  */
    if( pConfig->a[i].iLang==iLang )          /* (2) offset 16, then (3) deref a[i] */
```

| arm | cause | pc | which step | fault address |
|---|---|---|---|---|
| virtual-nested-pools | 25 | `editDist3FindLang+164` | **(1)**, the `lw` of `nLang` | `0x3fa577b890`, inside the arena |
| virtual-malloc | 24 | `editDist3FindLang+240` | **(3)** | `0x0` |
| cheribsd-revocation | SIGPROT(34) si_code=2 | `editDist3FindLang+0x40` | **(3)** | -- |

The disassembly is what separates them. On the Sublet image, `+164`:

```
  li a3, 0x10 ; add a3, a2, a3 ; shrink a1, a2, a3
  ldc a1, 0x0(a1)          ; pConfig off the stack
  lw  a1, 0x0(a1)          ; <-- +164, pConfig->nLang
```

On the base image, `+240`:

```
  lw  a1, 0x0(a1)          ; i
  slli a1, a1, 0x5         ; i * 32 == sizeof(EditDist3Lang)
  cincoffset a0, a0, a1    ; &a[i]
  lw  a0, 0x0(a0)          ; <-- +240, a[i].iLang
```

and on CheriBSD, `+0x40`, the same shape with the load of `a` two instructions earlier:

```
  lw  a1, 0(ca1)           ; pConfig->nLang      -- SUCCEEDS
  lc  ca0, 16(ca0)         ; pConfig->a          -- loads untagged garbage, does not fault
  slli a1, a1, 5 ; cincoffset ca0, ca0, a1
  lw  a0, 0(ca0)           ; <-- +0x40, a[i].iLang
```

**Host ASan pins which access the case is about**: `0 bytes inside of 16-byte region`, and
offset 0 of a 16-byte `EditDist3Config` under the host's 8-byte-pointer layout is `nLang`, step
(1). That is the access Sublet faults on. The other two arms let steps (1) and (2) succeed and
fault on step (3), on a pointer they read **out of** the dead object.

### What the dead object contains when it is read

Probe site 57, identically on native memsys5 and on CheriBSD purecap:

```
ALLOC   pConfig=... size=32
FREE    pConfig=... rc=7 nLang=0 a=0x0   <- else of the THIRD if
...BUG STATE at failAt=14 rc=NOMEM callable=1...
READ    pConfig=... nLang_read=1953786219 a_read=0x74746973272c276e
```

At the free the fields are the `memset` values, `nLang=0` and `a=NULL`; `a` is never populated,
because only `editDist3ConfigLoad` fills it and this trigger never gets there. At the read both
come back as something else, and the same something on both platforms: the bytes of
`a_read` are SQL text (`n','sitt`), so the block has been reissued. That matters twice over --
`nLang` coming back as a large positive number is what makes the loop run at all, and `a`
coming back as eight bytes of another object's string is what gets dereferenced.

The CheriBSD witness settles that arm:

```
LB_WITNESS site=57 kind=OOB off=0 n=8 req=7 blk=0x1d91c0 tag=1 cb=0x19c200+262144
```

`tag=1` and bounds covering the whole 256 KiB arena: the defective read goes through a fully
valid capability, so CHERI has nothing to object to. `req=7` is the size the block was reissued
at -- a short SQL string -- and an 8-byte field read exceeding it is why the probe reports
`kind=OOB` rather than `kind=UAF`. That is the address judgement misclassifying, as in cases 12
and 21, not an over-read.

### Records corrected

* `base`: **detected -> not a detection.**
* `cheribsd-revocation`: **detected -> not a detection.** The signal is real and reproduced; what
  it means is not. Every one of the five detections this arm had recorded has now been shown not
  to be one.

  **RETRACTION, later the same day.** This paragraph originally continued "With this row
  CheriBSD's SQLite column reaches **0 of 33**". That was wrong, and the commit carrying it was
  `9ee8f131`. The column is **2 of 33**. The nine spatial cases were all recorded `not-detected`
  on 2026-10-06, and those readings were taken **without their corrupt database images present**
  -- the silent failure found while reviewing case 26 and fixed in `e9199203`, where
  `sqlite3_open` creates an empty database, the first statement returns *no such table*, and the
  case prints `RETURNED rc=0`. With the images restored, cases **29** and **31** fault on this
  arm with `SIGPROT si_code=1`, `PROT_CHERI_BOUNDS` -- a bounds violation on a **valid**
  capability, categorically different from the `si_code=2` tag faults every false positive in
  this family turned out to be. Both are real: probe site 64 shows case 31's cell walk reaching
  `iOff=52992` with the witness reporting `kind=CAP-OOB`, `cb=<arena>+262144`, `ca=0x21d120`,
  past the arena's own end. A spatial over-read that leaves the whole 256 KiB arena is visible
  even where bounds are at `malloc` granularity; one that stays inside it is not, which is why
  the other six spatial cases are silent here. What this row establishes stands unchanged -- the
  five **temporal** detections this arm recorded are not detections -- and the count it drew
  from them does not.
* `sublet`: holds, on the virtual arm. The physical row is one of the 28 kernel-pc halts and is
  marked.
* The old `status` line said `cheribsd-revocation` completes with a clean exit. Stale: it faults.
* `live_proof` now records that the upstream reference cannot be verified here, and that the two
  identifiers in this record are the same one written two ways.
* `fault_sites_why` records that attribution by function cannot work for this case, because all
  three arms fault inside the one declared function on different accesses.

### A gap in the CheriBSD arm's build, fixed

`ports/sqlite/cheribsd/build-corpus-cheri.sh` selected the extension source for `spellfixoom`
unconditionally:

```bash
spellfixoom) xsrc=("$SRC/spellfix.c"); xinc=(-DSQLITE_CORE -I"$EXT/misc") ;;
```

with no `PROBE` branch, unlike `expertrem` beside it. This case's defect is in the extension, so
without that branch the arm could never place a site at it -- which is what the old `status`
line meant by "no defect-site probe in that run". The branch is now there, and the CheriBSD
trace above is the first probe reading this case has had on that arm.

### Why the chain reaches step (3) at all, and who wrote the forged pointer

Three details decide this, and they are worth stating because they are what makes the pattern
possible rather than incidental to this case.

**Loading an untagged value is not a fault.** Step (2) is an `lc` -- a capability load -- of the
sixteen bytes at offset 16. Those bytes were written as ordinary character data by whoever now
owns the block, so they carry no tag, and the loaded value is a non-capability. Neither
architecture objects: the tag is checked on **dereference**, not on load. That is why execution
gets as far as step (3), where the `lw` through that value is what traps.

**The writer can only have been the block's new owner, not the allocator.** `a` sits at offset
16, and memsys5's `Mem5Link{next,prev}` is eight bytes at offset 0 -- so the allocator's own
metadata cannot reach this field. That distinguishes this row from cases 19 and 20, where the
clobbered pointer was the struct's first member and `Mem5Link` was the writer. The probe's bytes
confirm it: decoded, `nLang_read` is `6b 69 74 74` and `a_read` is `6e 27 2c 27 73 69 74 74`,
ASCII fragments of the trigger's own SQL word list -- which for an edit-distance test is words
like `'kitten'` and `'sitting'`.

**The reissue is what CREATES the fault here, and that is the opposite of cases 07, 14 and 17.**
At the free the fields are `nLang=0` and `a=NULL`; with the block left alone, `nLang` reads back
0, the loop is never entered, and both non-Sublet arms complete in silence. It is the foreign
bytes that make `nLang` large enough to enter the loop and give step (3) something to
dereference. In cases 07, 14 and 17 a reissue did the reverse -- offset 0 came back holding a
live object's properly tagged pointer, and the arms went silent. Same allocator behaviour,
opposite effect on what these two arms report, which is the sharpest available statement of the
point: **what base Capstone and CheriBSD report on a nested use-after-free is decided by what
memsys5 happened to do with the block, not by the defect.** Sublet's reading does not move with
it, because the lifetime of the sub-allocation is the thing it checks.
