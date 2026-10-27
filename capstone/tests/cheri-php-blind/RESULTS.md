# CheriBSD purecap vs the 31 ASAN-blind PHP defects — measured

**CHERI purecap detects none of the 31, in every revocation configuration
including revoke-on-every-free.** Every control fired, so this is a measurement
and not an absence of one.

## Vehicle (config reality — a verdict is void without this)

| component | version |
|---|---|
| CheriBSD | FreeBSD 15.0-CURRENT `releng/26.07-88f39900c329`, `CHERI-PURECAP-QEMU` (published dist sets) |
| CHERI-LLVM | clang 17.0.0 (CTSRD-CHERI), built from source |
| CHERI-QEMU | 7.2.0 (CHERI-Alliance fork), `qemu-system-riscv64xcheri` |
| purecap ABI | `-march=rv64gcxcheri -mabi=l64pc128d`; binaries report `cheriabi, capability mode` |
| boot flag | `-cpu rv64,sstc=false` — required; see `VEHICLE-BRINGUP.md` |

Measured per pass, from the guest:

| config | `runtime_revocation_default` | `..._every_free_default` | `malloc_revoke_enabled()` |
|---|:--:|:--:|:--:|
| spatial | 0 | 0 | **0** |
| temporal | 1 | 0 | **1** |
| eager | 1 | 1 | **1** |

`hdr=48` on purecap (two 16-byte capabilities plus the size/cached bitfield),
versus 24 on x86-64.

## The result

Columns are the three configs; `@` gives the phase the process died in, which is
what separates a catch of the defect from a catch during shutdown.

| row | models | stockalloc (real PHP) | patched (corpus patches) |
|---|---|---|---|
| `a1_uaf_read` | 23 cache-hidden UAF | **MISS** | BLOCKED-SWEEP(eager) |
| `a2_uaf_write` | " | **MISS** | BLOCKED-SWEEP(eager) |
| `a3_uaf_recycled` | " | **MISS** | BLOCKED-SWEEP(eager) |
| `a4_uaf_zval_member` | " | **MISS** | BLOCKED-SWEEP(eager) |
| `b1_oob_read_slack` | 6 CWE-125 | **MISS** | **MISS** |
| `b2_oob_write_slack` | 1 CWE-787 | **MISS** | **MISS** |
| `c1_stack_after_return` | 1 CWE-562 | **MISS** | **MISS** |
| `b4_header_underflow` | supplementary | MISS | MISS |
| — controls — | | | |
| `b3_oob_read_past_block` | bounds control | BLOCKED-SYNC | BLOCKED-SYNC |
| `a6_uaf_after_shutdown` | revocation control | **BLOCKED-SWEEP(eager)** | BLOCKED-SWEEP(eager) |
| `a5_double_efree` | list-corruption control | LATE@SHUT | BLOCKED-SYNC |
| `z0_sanity` | sanity | RAN-CLEAN | RAN-CLEAN |

### The one line that carries the claim

`a1`–`a4`, eager config: **identical defect, identical access, identical kernel
policy — `MISS` with PHP's allocator, `SIGPROT` at the offending access the
moment the allocator is patched.** Nothing about the bug changed; only where the
allocator returns memory from did.

### Why this is not "revocation was never on"

`a6_uaf_after_shutdown` runs the **same stock allocator** but places its use
*after* `shutdown_memory_manager` has drained the size-class cache and genuinely
`free()`d the block. It is caught (`SIGPROT@USE-BEGIN`) under eager. So
revocation demonstrably works on this vehicle, with PHP's own allocator in
place — it simply has nothing to revoke while `efree` is parking blocks in
`AG(cache)`.

Equally, `b3_oob_read_past_block` faults in all three configs and both arms, so
bounds are being enforced. A `MISS` here means in-bounds, not unchecked.

### Async revocation — the realistic default — catches nothing extra

The `temporal` pass is byte-for-byte the same as `spatial`. Every catch in this
experiment comes either from bounds/tags (no revocation needed) or from
revoke-on-every-free.

`z2_libc_uaf` shows this directly, and it is the cleanest single measurement in
the set. It is plain libc `malloc` -> `free` -> use, with no PHP allocator
anywhere, and it reads the capability's **tag** without dereferencing it:

| config | `TAG-AFTER-FREE` | outcome |
|---|:--:|---|
| spatial | 1 | UAF survived |
| temporal (realistic default) | **1** | UAF survived |
| eager | **0** | `SIGPROT` |

The tag flips 1 -> 0 only under revoke-on-every-free, identically in both arms
(`BLOCKED-SWEEP(eager)`). So on this vehicle
revocation is functional but the default async quarantine never sweeps in a
process this short-lived — not even for a bare `malloc`/`free` pair. This
reproduces what the SQLite CHERI baseline next door measured (0 of its 10 UAF
rows caught under the async default) and it rules out the one alternative
explanation that would void the whole table: that revocation simply never ran.

## Four pre-registered predictions that were wrong

`rows.tsv` still carries the original predictions, so `classify.py` keeps
flagging these. They are left visible on purpose.

1. **`b1`/`b2`, patched arm: predicted BLOCKED-SYNC, measured MISS.** The most
   interesting miss, and now measured rather than inferred. `z1_bounds_probe`
   reads `base`/`address`/`length` straight off the capabilities:

   | request | n=26, stock arm | n=26, patched arm |
   |---|---|---|
   | PHP `REAL_SIZE(26)` | 32 | 26 |
   | `malloc(hdr + REAL_SIZE)` | 80 | 74 |
   | capability length | 80 | **80** |
   | bytes readable past the object | **6** | **6** |

   The `emalloc`'d pointer sits at offset 48 with the block's base and full
   length (`d-off=48`, `dlen=rawlen` for every size), so it inherits the whole
   block.

   **Removing PHP's rounding does not help, because the allocator underneath
   rounds by the same amount.** In the stock arm the 6 bytes come from
   `REAL_SIZE`; in the patched arm `REAL_SIZE` is exact and `malloc(74)` is
   bounded to 80 instead. Same 6 bytes, different layer. Measured slack for
   other sizes: `malloc(56)->64`, `malloc(72)->80`, `malloc(88)->96`,
   `malloc(1048)->`**`1280`** (232 bytes).

   The blindness is therefore **size-dependent, not universal**: at n=32 and
   n=48 the patched arm has 0 bytes of slack, so a one-byte over-read there
   *would* be caught. What CHERI cannot do is catch an over-read that stays
   inside the allocator's own rounding, at whichever layer supplies it.

2. **`a3_uaf_recycled`, patched arm: predicted MISS, measured
   BLOCKED-SWEEP(eager).** The prediction said "no `free()` in *either* arm",
   which is wrong: disabling the cache makes `efree` call `free()`, so the stale
   capability is revoked. The stock arm — the one that models the corpus — is
   `MISS` as predicted.
3. **`a5_double_efree`, patched arm: predicted BLOCKED-ABORT, measured
   BLOCKED-SYNC — and my first explanation of it was wrong.** I suggested CHERI
   was catching a *clobbered* capability in the freed header, which would have
   been a strength no shadow-memory tool has. Instrumenting the row refutes it:
   it prints `LINK-TAGS-BEFORE pNext=0 pLast=0` — the links are untagged
   *before* anything is freed, because `ADD_POINTER_TO_LIST` sets both to NULL
   for a single allocation and a NULL capability has tag 0.

   What actually happens: the second `efree` reaches
   `REMOVE_POINTER_FROM_LIST`, `AG(head)` is already NULL, so control takes the
   `else` branch and stores through `p->pLast->pNext` with `pLast == NULL`. That
   is a **null-capability dereference**, which any conventional system would
   also fault on. It is not CHERI-specific, and it is not evidence about
   corrupted pointer fields.

   The instrument was also the wrong one for the question: with a single live
   allocation the links are legitimately NULL, so it cannot distinguish NULL
   from clobbered. Testing the clobbered-pointer idea needs two or more live
   blocks so the links are real capabilities. Left open — `a5` is a control, not
   one of the 31, so this does not bear on the headline.

4. **`a5_double_efree`, stock arm** was pre-registered as `LATE-or-ABORT`, which
   is not a taxonomy label and so could never match. Fixture defect, not a
   prediction error; the measured `LATE@SHUT` is within what was predicted.

## Every control fired

| control | what it excludes | measured |
|---|---|---|
| `b3_oob_read_past_block` | bounds not enforced | `BLOCKED-SYNC`, all 3 configs, both arms |
| `a6_uaf_after_shutdown` | revocation never runs *with PHP's allocator* | `BLOCKED-SWEEP(eager)` in the **stock** arm |
| `z2_libc_uaf` | revocation never runs *at all* | tag 1->0 and `SIGPROT` under eager |
| the `patched` arm | the defect is not reachable/detectable in principle | `a1`-`a4` flip to `SIGPROT@USE` |
| `z0_sanity` | the harness itself is unsound purecap | `RAN-CLEAN`, all configs, both arms |
| `selftest-classify.sh` | the classifier cannot emit a catch | all verdict branches exercised |

So a `MISS` in this table means in-bounds-and-tagged, not unchecked.

## Where CHERI is weaker than ASAN

`c1_stack_after_return` (CWE-562, the class of CRASH-100) is `MISS` in every
config and both arms, while ASAN catches it with
`detect_stack_use_after_return=1`. A returned stack capability stays tagged and
in bounds: the frame is reused, never unmapped, and CheriBSD does not revoke
stack capabilities on return. The row proves it created the real condition —
`DANGLING-NONNULL=1`, and it reads back the clobber byte `0x30` rather than its
own `'S'` (`0x53`).

`b4_header_underflow` is also `MISS` everywhere: a read below the object lands in
the `zend_mem_header`, inside the same malloc'd block and so inside the same
capability. The blindness is two-sided, not just a tail-slack artefact. ASAN
misses this one too, in both arms.

## What this licenses saying

Detector blindness here is a property of **where the detector interposes**, not
of how strong it is. ASAN interposes at `malloc`/`free`; CHERI bounds what
`malloc` returns and revokes what `free` releases. PHP's `emalloc`/`efree` sit
*above* both: `efree` parks a block in a size-class cache without calling
`free()`, and `emalloc` hands out `p + sizeof(header)` by plain pointer
arithmetic that inherits the block's bounds. A nested allocator therefore hides
the same 31 defects from both tools, and from CHERI's most expensive
configuration as well — revoke-on-every-free has nothing to revoke.

This is strictly stronger than the CHERI result the project already had: in the
SQLite baseline revoke-on-every-free blocked 12 of 15 rows, and only the
reuse-not-free row survived every configuration. Here it blocks **0 of 4**
cache-hidden UAF shapes in the stock arm, because the allocator never reaches
`free()` at all.

## Scope

These rows model the 31 defects' **mechanism**; they are not the 31 triggers.
Running those needs a full PHP 5.0.0 purecap build, which is gated separately.
CheriBSD, being a complete FreeBSD, could in principle run all 31 — including
the three that need loopback, an HTTP listener and a writable `/tmp`, which the
Capstone domain cannot reach.

## Reproduce

```bash
bash fetch-host-deps.sh        # no root required
. vehicle-env-no-sudo.sh
bash ../cheri-baseline/provision-cheri-vehicle.sh
bash run-php-blind.sh          # selftests, compiles both arms, boots, classifies
```
