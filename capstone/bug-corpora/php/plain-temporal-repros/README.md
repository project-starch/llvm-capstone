# php / plain-temporal-repros

Twenty-six temporal defects live in PHP 5.0.0 whose **release reaches the system allocator**, so a
detector that interposes at `malloc` can still see them. This corpus is the control for its sibling
[`../zend-alloc-repros`](../zend-alloc-repros), which holds the thirty whose release stops at Zend's
cache.

> **Superseded for placement (2026-10-30), kept whole.**
> [`../engine-repros`](../engine-repros) carries 70 defects of this same family -- 52 of them
> these -- and is the live corpus; the `cheribsd-revocation` arm now runs there too. Its split is
> decided by the same two-ASan-build measurement, on a different toolchain (gcc 15.2.0 against
> gcc 7.5.0 / `libasan.so.4` here), and the two **disagree on 18 of the 52**: these populations
> are a second reading of the axis, not the reading, so no count here may be added to one there.
> Nothing is removed. Every case, every host-ASan row and every bundle stays as measured -- they
> are what the CheriBSD arm's first run rests on, and cases 18-21 (`ext/mbstring`, `ext/pcre`)
> have a reading nowhere else, because `engine-repros` admits only cases the ext/standard-only
> build runs. See that corpus's README, "Not here".

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
        --corpus capstone/bug-corpora/php/plain-temporal-repros \
        --arm virtual-malloc --arm virtual-nested-pools \
        --qemu <qemu-system-riscv64> --images <dir> --adapter <dir> \
        --image <php-real-seam.dom> --probe <probe-seam.dom> \
        --llvm-bin <llvm/bin> --out capstone/bug-corpora/php/plain-temporal-repros/results/<stamp>

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

    python3 capstone/bug-corpora/php/tools/run-php-corpus.py         --corpus capstone/bug-corpora/php/plain-temporal-repros         --arm cheribsd-revocation         --out capstone/bug-corpora/php/plain-temporal-repros/results/<stamp> --scratch <out>/stage

It boots twice, with `security.cheri.runtime_revocation_default` at 1 and then 0: a purecap
`SIGPROT` carries no symbol, so the pair is what ties a fault to the defect.

## The cases

| # | case | source id | stock allocator | released by |
|--:|---|---|---|---|
| 1 | [`01_fd5bd37ab129_array-walk-internal-pointer-callback-uaf`](01_fd5bd37ab129_array-walk-internal-pointer-callback-uaf) | CRASH-002 | ASAN:heap-use-after-free | free(), through _efree |
| 2 | [`02_1d33a3e95e4e_globals-hashtable-destroy-efree-no-symtable`](02_1d33a3e95e4e_globals-hashtable-destroy-efree-no-symtable) | CRASH-012 | ASAN:attempting free on address | not in the report |
| 3 | [`03_d7b30e457a3d_call-method-on-destroyed-object-store-bucket`](03_d7b30e457a3d_call-method-on-destroyed-object-store-bucket) | CRASH-030 | ASAN:heap-use-after-free | not in the report |
| 4 | [`04_4e2c0bc9050c_objects-store-stale-obj-ptr-after-dtor`](04_4e2c0bc9050c_objects-store-stale-obj-ptr-after-dtor) | CRASH-046 | ASAN:heap-use-after-free | realloc(), through _erealloc |
| 5 | [`05_1b317f15264e_return-generated-loop-free-refreed-by-brk`](05_1b317f15264e_return-generated-loop-free-refreed-by-brk) | CRASH-050 | ASAN:heap-use-after-free | free(), through _efree |
| 6 | [`06_2e1a2438b5e6_user-error-handler-recursive-compile`](06_2e1a2438b5e6_user-error-handler-recursive-compile) | CRASH-062 | ASAN:heap-use-after-free | free(), through _efree |
| 7 | [`07_1ea22c90046b_register-constant-redefine-frees-name-before`](07_1ea22c90046b_register-constant-redefine-frees-name-before) | CRASH-068 | ASAN:heap-use-after-free | free(), called directly |
| 8 | [`08_d8efe626b535_begin-class-member-fcall-indirect-static`](08_d8efe626b535_begin-class-member-fcall-indirect-static) | CRASH-070 | ASAN:heap-use-after-free | realloc(), through _erealloc |
| 9 | [`09_d5bb03b11a96_open-stream-derefs-freed-volatile-url`](09_d5bb03b11a96_open-stream-derefs-freed-volatile-url) | CRASH-072 | ASAN:heap-use-after-free | free(), called directly |
| 10 | [`10_dba1f1d42fbb_array-walk-func-name-arg-stack-pointer`](10_dba1f1d42fbb_array-walk-func-name-arg-stack-pointer) | CRASH-075 | ASAN:heap-use-after-free | realloc(), through _erealloc |
| 11 | [`11_96d755978cc8_array-diff-intersect-globals-symtable-alias`](11_96d755978cc8_array-diff-intersect-globals-symtable-alias) | CRASH-078 | ASAN:heap-use-after-free | free(), through _efree |
| 12 | [`12_e615889d6b43_array-filter-stale-argstack-zvalpp-across`](12_e615889d6b43_array-filter-stale-argstack-zvalpp-across) | CRASH-081 | ASAN:heap-use-after-free | realloc(), through _erealloc |
| 13 | [`13_8bd18e174b4e_call-user-func-params-into-argument-stack`](13_8bd18e174b4e_call-user-func-params-into-argument-stack) | CRASH-083 | ASAN:heap-use-after-free | realloc(), through _erealloc |
| 14 | [`14_3053a6262153_exit-in-stream-filter-unclean-shutdown`](14_3053a6262153_exit-in-stream-filter-unclean-shutdown) | CRASH-113 | ASAN:heap-use-after-free | not in the report |
| 15 | [`15_83ac79600e3a_unserialize-buf-zval-ptr-into-arg-stack`](15_83ac79600e3a_unserialize-buf-zval-ptr-into-arg-stack) | CRASH-117 | ASAN:heap-use-after-free | realloc(), through _erealloc |
| 16 | [`16_630f9c33c236_unserialize-process-nested-data-duplicate`](16_630f9c33c236_unserialize-process-nested-data-duplicate) | CRASH-118 | ASAN:heap-use-after-free | free(), through _efree |
| 17 | [`17_1ac4d8f2c632_unserialize-r-lowercase-backref-zval-ptr`](17_1ac4d8f2c632_unserialize-r-lowercase-backref-zval-ptr) | CRASH-121 | ASAN:heap-use-after-free | free(), through _efree |
| 18 | [`18_09370413757a_mb-detect-encoding-parse-encoding-list-fail`](18_09370413757a_mb-detect-encoding-parse-encoding-list-fail) | CRASH-125 | ASAN:heap-use-after-free | free(), through _efree |
| 19 | [`19_3e0680f2ee00_mb-ereg-replace-e-option-eval-failure-stale`](19_3e0680f2ee00_mb-ereg-replace-e-option-eval-failure-stale) | CRASH-129 | ASAN:heap-use-after-free | free(), through _efree |
| 20 | [`20_631da59b5032_preg-match-all-named-subpat-match-set-double`](20_631da59b5032_preg-match-all-named-subpat-match-set-double) | CRASH-131 | ASAN:heap-use-after-free | free(), through _efree |
| 21 | [`21_634fef42d64c_preg-replace-callback-subpats-unconditional`](21_634fef42d64c_preg-replace-callback-subpats-unconditional) | CRASH-132 | ASAN:heap-use-after-free | free(), through _efree |
| 22 | [`22_e000da960b5c_call-user-function-byref-param-refcount1`](22_e000da960b5c_call-user-function-byref-param-refcount1) | CRASH-141 | ASAN:heap-use-after-free | free(), through _efree |
| 23 | [`23_8df40bdb313d_arg-stack-slots-not-nulled-during-arg-free`](23_8df40bdb313d_arg-stack-slots-not-nulled-during-arg-free) | CRASH-142 | ASAN:heap-use-after-free | not in the report |
| 24 | [`24_2a31dbbadfb9_cast-to-object-adopts-globals-symbol-table`](24_2a31dbbadfb9_cast-to-object-adopts-globals-symbol-table) | CRASH-148 | ASAN:attempting free on address | not in the report |
| 25 | [`25_1aab36e3f22a_is-a-impl-autoload-reentry-arg-stack-realloc`](25_1aab36e3f22a_is-a-impl-autoload-reentry-arg-stack-realloc) | CRASH-155 | ASAN:heap-use-after-free | realloc(), through _erealloc |
| 26 | [`26_ed4c0245c7ca_zval-copy-ctor-array-hands-over-hashtable`](26_ed4c0245c7ca_zval-copy-ctor-array-hands-over-hashtable) | CRASH-161 | ASAN:heap-use-after-free | free(), through _efree |

## Verdict and protection are two different readings

The corpus keeps them apart, because conflating them is how an arm gets credit for a fault that
had nothing to do with the defect.

**The verdict** is what the arm measured at the **platform default** boot
(`runtime_revocation_default=1`, `..._every_free_default=0`) -- the configuration CheriBSD ships.
It is formed by the shared judge from one `Observation` per case and written to
`results/20261029/cheribsd-revocation/verdicts.jsonl`; nothing is tallied into it by hand.

**Protection** asks a different question -- *does CHERI stop this program on this defect at all,
in any configuration the platform supports* -- and it is an **attribution**, so it needs more than
the default boot:

- a **`SIGPROT` at the default** is attributed by the paired revocation-**off** boot, because a
  purecap `SIGPROT` carries no symbol: off-completes means revocation fired, off-also-faults means
  bounds or a tag PHP had already cleared itself;
- a **`SIGABRT` at the default** is CheriBSD's allocator consistency check rather than a capability
  fault, and it still counts, because the check is **capability-enabled**: a hybrid-ABI build in
  the same guest against the same jemalloc survives the same double free that aborts under purecap,
  so the ABI is the only variable (`ports/php/cheribsd/results/20261029-hybrid-ab`);
- a fault that appears **only under every-free revocation** is recorded as `NOT-REISSUED`, with the
  place it lands. It does not change the verdict -- at the default the chunk sits in the quarantine
  unswept and the program runs on -- but it does answer the protection question, because the stale
  pointer *is* followed and the capability *is* revoked when the sweep happens. Every-free is a
  supported CheriBSD policy (`security.cheri.runtime_revocation_every_free_default`), not a switch
  invented here;
- a case with **no usable reading at the default** -- the trigger's own ending is a fatal PHP error,
  or the guest watchdog kills it -- is excluded from the protection denominator rather than scored,
  and so is a case that is **silent in all three configurations**.

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
| CAUGHT | 3 | 10, 12, 17 |
| NO-READING: inconclusive | 14 | 01, 02, 03, 05, 06, 08, 16, 18, 19, 20, 21, 22, 23, 24 |
| MISSED | 8 | 04, 07, 09, 11, 13, 14, 15, 25 |
| NO-READING: infra | 1 | 26 |

### Protection: CHERI protects 21 of the 26

| what stops the program | cases | which |
|---|---:|---|
| `SIGPROT` at the platform default | 3 | 10, 12, 17 |
| `SIGABRT` at the platform default -- the capability-enabled allocator check | 11 | 01, 02, 05, 16, 18, 19, 20, 21, 22, 23, 24 |
| `SIGPROT` under every-free only (`NOT-REISSUED`) | 7 | 04, 09, 11, 13, 14, 15, 25 |
| **protected** | **21** | |

The five that are not counted, and why none of them is evidence that CHERI failed:

| cases | why not counted |
|---|---|
| 03, 06, 08 | the trigger's own ending at the default is a fatal PHP error (exit 255), neither a fault nor a clean run, so the case yields no reading about protection |
| 26 | the guest watchdog kills it in all three configurations. No reading |
| 07 | **silent in all three configurations** -- the one genuine miss in either corpus |

So on this corpus -- the defects whose release **does** reach the system allocator -- the reading
is that **CHERI protects jemalloc's heap**: for 21 of 26 the program is stopped, and the only
outright failure is 07. What separates the 7 `NOT-REISSUED` rows from the other 14 is *when* the
sweep runs, not whether the capability is revocable: the chunk is in the quarantine in both cases.

| group | cases | which | what fired |
|---|---:|---|---|
| A | 3 | 10, 12, 17 | `SIGPROT` at the default. **Only 12 is revocation's**: it completes with revocation off. 10 and 17 fault with revocation off as well, so bounds or a tag already cleared by PHP itself |
| B | 11 | 01, 02, 05, 16, 18, 19, 20, 21, 22, 23, 24 | the allocator's consistency check (`SIGABRT`), capability-enabled per the hybrid A/B |
| C | 3 | 03, 06, 08 | exit 255 at the default, with no fault and no clean end |
| D | 8 | 04, 07, 09, 11, 13, 14, 15, 25 | silent at the default. **7 of the 8 fault under every-free** -- 6 inside the script body, 1 in request shutdown -- so the chunk is in the quarantine and the stale pointer is followed; the deferred sweep is what lets the access through. **07 is silent in all three configurations: the only true miss in either corpus** |
| E | 1 | 26 | the guest watchdog kills it in all three. No reading |

One reversal is worth recording rather than smoothing over: **17 completes under every-free** while
faulting at both the default and off. Every-free changes when chunks are reissued, so it is not a
strictly stronger configuration, which is a second reason it is context and not a verdict.
