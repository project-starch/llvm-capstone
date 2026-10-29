# php / zend-alloc-repros

Thirty temporal defects live in PHP 5.0.0 that a detector interposing at `malloc` **cannot** see.
`_efree` parks the freed block in `AG(cache)` and a later `emalloc` hands the same bytes to the next
object, so `free()` is never called and the libc block stays live and in bounds for the whole
window. The control is its sibling [`../plain-temporal-repros`](../plain-temporal-repros).

## How the two corpora were split

By one measurement, not by a rule. Each trigger ran twice on the same host, on two builds of the
same source differing only in two defines in `Zend/zend_alloc.c`:

| build | what it is |
|---|---|
| detector | `ZEND_DISABLE_MEMORY_CACHE 1`, `REAL_SIZE` unrounded -- every `efree` reaches `free()` |
| stock | Zend's allocator **as shipped** |

A case whose temporal report survives the stock build is in `plain-temporal-repros`; one that loses
it is here in `zend-alloc-repros`. Both readings are in
[`results/20261028-host-asan/matrix.tsv`](results/20261028-host-asan/matrix.tsv), per case, with the route the block took
out of the allocator and the first frame of the report's access stack.

The mechanism is in the source rather than inferred: `_efree` pushes a block whose payload is at
most 80 bytes onto `AG(cache)` and returns (`Zend/zend_alloc.c:270-282`, `MAX_CACHED_MEMORY` 11 over
an 8-byte index). Size alone does not decide it -- the per-class cache holds 256 entries and falls
back to `free()` once full, and `_erealloc` never consults the cache at all -- which is why the
split comes from the run.

## Arms

| arm | configuration (`tools/arms.json`) | what it adds |
|---|---|---|
| `host-asan` | -- | the pin under ASan. Liveness evidence, not a protection column |
| `virtual-malloc` | `php-real-stock` (`PHP_CAP_MODE=0`) | musl mallocng: every object bounded, its lifetime retired at `free`. The seam is present and revokes nothing. **The baseline** |
| `virtual-nested-pools` | `php-real-exact` (`PHP_CAP_MODE=7`) | every `emalloc` block a `CDERIVE` child bounded at the caller's size, `CREVOKE`d at `efree` |
| `cheribsd-revocation` | `cheribsd-default` | stock CheriBSD purecap: bounds in `malloc`, quarantine and revocation after `free` |

`host-asan` and `cheribsd-revocation` are measured. The two virtual arms are declared and not yet run.

## Run it

    python3 capstone/bug-corpora/php/tools/run-php-corpus.py --self-test

    python3 capstone/bug-corpora/php/tools/run-php-corpus.py \
        --corpus capstone/bug-corpora/php/zend-alloc-repros \
        --arm virtual-malloc --arm virtual-nested-pools \
        --qemu <qemu-system-riscv64> --images <dir> --adapter <dir> \
        --image <php-real-seam.dom> --probe <probe-seam.dom> \
        --llvm-bin <llvm/bin> --out capstone/bug-corpora/php/zend-alloc-repros/results/<stamp>

Pass `virtual-malloc` first: it is also the reach oracle for the arms after it. One guest boot per
arm; the runner writes a `verdicts-v1` bundle per arm and the shared judge
(`bug-corpora/tools/verdicts.py`) forms every verdict. The images come from the app port:
`ports/php/app/build-virtual.sh probe` and `... php`, which write `probe-seam.dom` and
`realphp/php-real-seam.dom` -- one image serves both arms, which differ only in `PHP_CAP_MODE`.

The CheriBSD arm needs no QEMU flags of its own, but it does need the purecap binary and its
control staged first:

    CHERI_SDK=~/cheriBSD/cheri/output/sdk     CHERI_SYSROOT=~/cheriBSD/cheri/output/rootfs-riscv64-purecap     bash capstone/ports/php/cheribsd/build.sh /tmp/capstone/php-cheribsd
    mkdir -p <out>/stage && cp /tmp/capstone/php-cheribsd/php-5.0.0/sapi/cli/php <out>/stage/
    # and the control, built with the same flags (see ports/php/cheribsd/README.md)

    python3 capstone/bug-corpora/php/tools/run-php-corpus.py         --corpus capstone/bug-corpora/php/zend-alloc-repros         --arm cheribsd-revocation         --out capstone/bug-corpora/php/zend-alloc-repros/results/<stamp> --scratch <out>/stage

It boots twice, with `security.cheri.runtime_revocation_default` at 1 and then 0: a purecap
`SIGPROT` carries no symbol, so the pair is what ties a fault to the defect.

## The cases

| # | case | source id | stock allocator | released by |
|--:|---|---|---|---|
| 1 | [`01_b599e434add_assign-selfref-container-dtor-uaf`](01_b599e434add_assign-selfref-container-dtor-uaf) | CRASH-003 | OK | free(), through _efree |
| 2 | [`02_4f161fe28997_assignref-free-container-before-element-use`](02_4f161fe28997_assignref-free-container-before-element-use) | CRASH-004 | OK | free(), through _efree |
| 3 | [`03_fc96c7f7fa18_foreach-key-eq-value-refcount-uaf`](03_fc96c7f7fa18_foreach-key-eq-value-refcount-uaf) | CRASH-010 | OK | free(), through _efree |
| 4 | [`04_18a0d46a1b44_user-sort-comparator-frees-live-bucket-uaf`](04_18a0d46a1b44_user-sort-comparator-frees-live-bucket-uaf) | CRASH-022 | OK | free(), through _efree |
| 5 | [`05_79d6f11a4a3f_assign-ref-of-function-or-method-call-return`](05_79d6f11a4a3f_assign-ref-of-function-or-method-call-return) | CRASH-024 | OK | free(), through _efree |
| 6 | [`06_44325e647302_assign-ref-of-new-object-refcount-off-by-one`](06_44325e647302_assign-ref-of-new-object-refcount-off-by-one) | CRASH-025 | OK | free(), through _efree |
| 7 | [`07_f5cf052225ff_destructor-run-on-exception-failed`](07_f5cf052225ff_destructor-run-on-exception-failed) | CRASH-031 | ASAN:SEGV | free(), through _efree |
| 8 | [`08_07aa25b44276_error-handler-frees-autovivified-default`](08_07aa25b44276_error-handler-frees-autovivified-default) | CRASH-035 | OK | free(), through _efree |
| 9 | [`09_6319efa013ec_fci-cache-stores-freed-call-user-call`](09_6319efa013ec_fci-cache-stores-freed-call-user-call) | CRASH-038 | ASAN:heap-buffer-overflow | free(), through _efree |
| 10 | [`10_c42b3bf68938_hash-apply-deleter-runs-destructor-before`](10_c42b3bf68938_hash-apply-deleter-runs-destructor-before) | CRASH-040 | ASAN:SEGV | not in the report |
| 11 | [`11_41ad9b4d1fdd_magic-getter-frees-container-then-in-get`](11_41ad9b4d1fdd_magic-getter-frees-container-then-in-get) | CRASH-042 | OK | free(), through _efree |
| 12 | [`12_b8360c376b5d_objects-store-call-destructors-dtor-call`](12_b8360c376b5d_objects-store-call-destructors-dtor-call) | CRASH-044 | ASAN:SEGV | free(), through _efree |
| 13 | [`13_ce23692663fe_objects-store-del-ref-dtor-call-omits`](13_ce23692663fe_objects-store-del-ref-dtor-call-omits) | CRASH-045 | ASAN:SEGV | free(), through _efree |
| 14 | [`14_41ae8de13666_overloaded-array-read-returns-unseparated`](14_41ae8de13666_overloaded-array-read-returns-unseparated) | CRASH-047 | OK | free(), through _efree |
| 15 | [`15_8365e71cb92b_tmp-value-reused-after-freed-in-string`](15_8365e71cb92b_tmp-value-reused-after-freed-in-string) | CRASH-057 | OK | free(), through _efree |
| 16 | [`16_89e53d5ab231_zend-do-free-marks-unrelated-jmp-no-ctor`](16_89e53d5ab231_zend-do-free-marks-unrelated-jmp-no-ctor) | CRASH-065 | OK | free(), through _efree |
| 17 | [`17_8ce349b8e03c_byref-return-parenthesized-undefined-var`](17_8ce349b8e03c_byref-return-parenthesized-undefined-var) | CRASH-067 | OK | free(), through _efree |
| 18 | [`18_2507a544296b_overloaded-call-trampoline-fbc-efreed-then`](18_2507a544296b_overloaded-call-trampoline-fbc-efreed-then) | CRASH-069 | EXIT:255 | free(), through _efree |
| 19 | [`19_37d7df72a62e_array-walk-byref-userdata-shared-zval-ptr`](19_37d7df72a62e_array-walk-byref-userdata-shared-zval-ptr) | CRASH-074 | OK | free(), through _efree |
| 20 | [`20_72c6d5cbafc9_array-reduce-initial-arg-unseparated-result`](20_72c6d5cbafc9_array-reduce-initial-arg-unseparated-result) | CRASH-080 | OK | free(), through _efree |
| 21 | [`21_562f886ecb14_unregister-tick-function-frees-executing`](21_562f886ecb14_unregister-tick-function-frees-executing) | CRASH-086 | OK | free(), through _efree |
| 22 | [`22_b90d985a23a5_stream-select-convert-to-long-ex-force`](22_b90d985a23a5_stream-select-convert-to-long-ex-force) | CRASH-099 | OK | free(), through _efree |
| 23 | [`23_90f4590d87fe_stristr-zval-ptr-dtor-on-own-passed-args`](23_90f4590d87fe_stristr-zval-ptr-dtor-on-own-passed-args) | CRASH-103 | OK | free(), through _efree |
| 24 | [`24_4c0970bec69b_userfilter-dtor-free-zval-on-live-object`](24_4c0970bec69b_userfilter-dtor-free-zval-on-live-object) | CRASH-112 | OK | not in the report |
| 25 | [`25_77acc692fd9b_user-filter-bucket-double-append-missing`](25_77acc692fd9b_user-filter-bucket-double-append-missing) | CRASH-114 | ASAN:SEGV | free(), through _efree |
| 26 | [`26_1088e28dfa6a_debug-print-backtrace-arg-array-not-reset`](26_1088e28dfa6a_debug-print-backtrace-arg-array-not-reset) | CRASH-140 | OK | free(), through _efree |
| 27 | [`27_f5f8cba8e978_assign-to-nonobject-property-error-arm`](27_f5f8cba8e978_assign-to-nonobject-property-error-arm) | CRASH-151 | OK | free(), through _efree |
| 28 | [`28_e88cdaa0143a_zval-dtor-array-arm-retains-is-array-type`](28_e88cdaa0143a_zval-dtor-array-arm-retains-is-array-type) | CRASH-154 | ASAN:SEGV | free(), through _efree |
| 29 | [`29_abb09693ac4d_assign-op-and-incdec-property-helpers-hold`](29_abb09693ac4d_assign-op-and-incdec-property-helpers-hold) | CRASH-159 | ASAN:SEGV | free(), through _efree |
| 30 | [`30_c3a317117ad8_hash-add-or-update-runs-destructor-on-still`](30_c3a317117ad8_hash-add-or-update-runs-destructor-on-still) | CRASH-160 | OK | free(), through _efree |

## What counts as caught

One rule for both corpora, and it is narrower than "something went wrong":

- the verdict comes from the **platform default** boot only (`runtime_revocation_default=1`,
  `..._every_free_default=0`). That is the configuration CheriBSD ships;
- a `SIGPROT` there is caught. Its cause is attributed by the paired revocation-**off** boot,
  because a purecap `SIGPROT` carries no symbol: off-also-faults means bounds or an already-cleared
  tag, off-completes means revocation;
- a `SIGABRT` there is caught as well. It is CheriBSD's allocator consistency check, not a
  capability fault, and it is counted because the check is **capability-enabled**: a hybrid-ABI
  build in the same guest against the same jemalloc survives the same double free that aborts under
  purecap, so the ABI is the only variable (`ports/php/cheribsd/results/20261029-hybrid-ab`);
- a fault that appears **only** under every-free revocation is **not** counted. It is recorded as
  `NOT-REISSUED` in the notes with the place it lands, because it says the stale pointer is
  followed and the chunk sits in the quarantine -- but the deployed policy does not sweep in time,
  so the program runs on.

Where a fault lands is read from a marker, not inferred: `auto_prepend_file` prints
`PHPB_TRIGGER_START` and `auto_append_file` prints `PHPB_TRIGGER_END`, so a fault with no END line
is inside the script body and one after it is in request shutdown. The runner refuses to place a
fault in a boot where **no** case printed END -- without that check a record predating the marker
reads as "every case faulted in the script body", a placement that was never measured.

## What the CHERI arm measured (2026-10-29)

Three boots of one purecap binary: the platform default, the same binary with revocation off, and
a third at every-free. The control `bounds-malloc` faulted in all three, and the sysctl read-backs
are in the bundle's `inputs.json`.

| verdict | cases | which |
|---|---:|---|
| CAUGHT | 7 | 07, 10, 12, 13, 15, 25, 29 |
| MISSED | 20 | 01, 02, 03, 05, 06, 08, 11, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 30 |
| NO-READING: inconclusive | 3 | 04, 09, 28 |

**Counted as caught: 10 of 30** -- the 7 `SIGPROT` plus the 3 `SIGABRT`.

| group | cases | which | what fired |
|---|---:|---|---|
| A | 7 | 07, 10, 12, 13, 15, 25, 29 | `SIGPROT` at the default -- but **all seven fault with revocation off too**, so none of them is revocation's. The tag is cleared by PHP writing a non-capability value over a capability slot |
| B | 3 | 04, 09, 28 | the allocator's consistency check (`SIGABRT`), at the shutdown cache drain where `shutdown_memory_manager` hands every cached block to real `free()` (`Zend/zend_alloc.c:478-495`). 04 prints its last line first, then aborts |
| C | 7 | 01, 02, 06, 08, 17, 26, 30 | silent at the default and off, **faulting under every-free inside the script body**. Not counted, and the reason matters -- see below |
| D | 13 | the rest | silent in all three configurations |

### Group C: the ABI pushes the object out of Zend's cache

This corpus exists because `_efree` parks a block of at most 80 bytes in `AG(cache)` and never
calls `free()`. For these seven that premise **fails on purecap**, and the measurement says so:

- the fault is `CHERI fault (type 0x2)`, a tag violation, on a capability the register dump marks
  `(invalid)`. The control in the same boot produced `type 0x1`, a length violation, so the two are
  distinguishable here;
- it appears only under every-free while the default and off boots both complete, so **revocation**
  cleared that tag -- which it can only do to a chunk libc was given back;
- the faulting `sepcc`, with the load bias validated against the control (`bounds-malloc.c:26`,
  its declared faulting line), resolves to the defect's own dereference site in every case:
  `zend_assign_to_variable`, `zend_assign_to_variable_reference`, `zend_std_write_property`,
  `make_real_object`, `zend_hash_internal_pointer_reset_ex`, `_zval_ptr_dtor`;
- the revoked block is **192 bytes** in all of them. `sizeof(zend_mem_header)` is 48 on purecap
  (DWARF) and `MEM_HEADER_PADDING` is 0, so the Zend payload is 144 bytes, `REAL_SIZE >> 3` is 18,
  and `MAX_CACHED_MEMORY` is 11 -- the block went to real `free()`.

The same objects on the host are a 24-byte header over a 24-77 byte payload, index 10 or less, so
they stay in the cache forever, which is exactly why the host's stock-allocator run reports nothing
for all seven. `MAX_CACHED_MEMORY = 11` is an 80-byte ceiling fixed in 2004 for an ABI where a
pointer was 8 bytes; under the capability ABI every pointer field doubles and the same object lands
several classes above it.

Two consequences, both of which belong to whoever writes this up:

1. these seven are missed at the platform default, but **not because the nested allocator hid them**
   -- the chunk was in libc's quarantine and the deferred sweep is what let the access through;
2. the split between this corpus and `../plain-temporal-repros` was measured on x86-64 and **does
   not transfer unchanged to purecap**: at least 7 of these 30 do reach the system allocator there.

One loose end: the exact C type of the 144-byte payload is not pinned. `Bucket` fits the arithmetic
(DWARF: 128 bytes purecap against 72 on the host, allocated `sizeof(Bucket)-1+nKeyLength`, so a
10-17 byte key lands in 137-144), but the fit is not unique and case 26's faulting register is used
as a `HashTable *`. It does not affect the conclusion above: every candidate is above index 11.
