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
  it means is not. With this row CheriBSD's SQLite column reaches **0 of 33**, and every one of
  the five detections it had recorded has now been shown not to be one.
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
