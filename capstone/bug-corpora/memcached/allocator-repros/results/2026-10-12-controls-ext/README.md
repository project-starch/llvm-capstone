# memcached cache.c object cache: parent, sibling and successor controls, 2026-10-12

This is memcached's row of the pre-registration
[`docs/plans/2026-10-12-nested-lifetime-controls-extended.md`](../../../../../docs/plans/2026-10-12-nested-lifetime-controls-extended.md)
(5d336c73), with the control definitions of
[`docs/plans/2026-10-11-nested-lifetime-controls.md`](../../../../../docs/plans/2026-10-11-nested-lifetime-controls.md).
The nested allocator is `cache.c`'s object cache (`cache_create`, `cache_alloc`, `cache_free`), the half
of patch 0006 that the slab controls of 2026-10-11 did not exercise
([`../2026-10-11-controls/`](../2026-10-11-controls/README.md)). The protected build is the application's
patch 0006 (`-DMCP_SUBLET=ON`, the controls compiled with `-DMC_CAPSTONE_SUBLET`): the
`virtual-nested-pools` arm, `virtual-memcached-sublet` in `tools/arms.json`. The release build is the same
corpus's `virtual-malloc` arm (`-DMCP_STOCK_MALLOC=ON`, `virtual-memcached-stock`), as in the slab run.

**Every prediction held: 24 runs, 3 boots per arm.** The stale access completed on release and read C's
data (0xc3). On protected it faulted at the labelled probe `read_probe` with cause 25 in all three boots.

## Where the controls are and how they run

The controls are in `controls/lifetime-cache/`, a sibling of the slab controls, so that
`controls/lifetime/` (its cases 91-93 and its build script, which requires exactly three cases) stays
unchanged:

| program | file | what it does |
|---|---|---|
| `ctl-94 94 list-head` | `94_ctl_parent_cache_list_reference/case.c` | control 1, parent |
| `ctl-95 95 sibling` | `95_ctl_sibling_same_cache/case.c` | control 2, sibling |
| `ctl-96 96 through-c` / `ctl-96 96 stale` | `96_ctl_successor_reuses_object/case.c` | control 3, successor; the stale read runs separately |

`cache-controls.h` holds the cache helpers. The probes, the `ACCESS-BEGIN` / `ACCESS-DONE` lines, the span
helpers and `main()` are the slab controls' own, used unchanged: `controls/lifetime/controls.h` and
`controls/lifetime/driver-controls.c`. `build-controls.sh` and `run-controls.py` are copies of the slab
controls' scripts, with the same port build, compile flags, boot (`tools/virtualvm.py`,
`runtime/virtual/run-staged.py`, `-smp 1`, `x-capstone-exact-bounds=true`), heap controls and fault
resolution. Only the programs, the variants and the record lines differ. Each boot first runs the virtual
heap's own controls from `postgres/c-repros`: `uaf-malloc` faulted with cause 25 and `bounds-malloc` with
cause 28, in all six boots.

Each control uses one cache, `cache_create(name, 256, sizeof(char *))`, with no limit (`limit=0`).

| arm | `cache->bufsize` | object layout |
|---|---|---|
| protected | 272 | a 16-byte slot, then the 256-byte object; the object capability is `[slot, slot + 272)` with its cursor at `slot + 16` |
| release | 256 | the virtual-mallocng object, `[object, object + 256)` |

Each access goes through the noinline probes of `controls/lifetime/controls.h`:

| probe | source line |
|---|---|
| `ctl_read_u8` | `controls/lifetime/controls.h:48` |
| `ctl_write_u8` | `controls/lifetime/controls.h:51` |
| `ctl_load_ptr` | `controls/lifetime/controls.h:55` |
| `read_probe` (the stale access) | `shared/corpus.h:81` |

`cache_free` writes the free-list link (`STAILQ`) into the first 16 bytes of what it pushes. That is the
slot under patch 0006 and the object's first 16 bytes on release. Every control write therefore goes to
the object's bytes `[16, 32)`, which nothing in `cache.c` touches.

## Which reference is the parent (control 1)

Under patch 0006, `do_cache_alloc` derives a new object's **lifetime** from what `malloc` returned:
`capstone_cap_derive(ret, 0, bufsize)`. It then stores the lifetime in the slot and returns
`get_object(capstone_cap_derive(lifetime, 0, bufsize))`. It does this on both paths, a new object and one
taken from the free list. `do_cache_free` loads the lifetime from `ptr[-1]` and calls
`capstone_cap_revoke_child(lifetime, ptr)`. It then pushes the **lifetime** onto the free list. So after
`cache_free(A)` the cache's own reference to A's storage is `cache->head.stqh_first`. Under patch 0006
that reference is A's direct parent, the CREVOKE parent, and it is also the reference the next object
will be derived from. The control reads it from the `cache_t` (`cache.h`'s public `STAILQ_HEAD`).

Before the free, the control reads the slot through A and records the lifetime. After the free, it
requires the list head to have the same address and bounds, with tag 1. In run f1 that check printed:

    LIFETIME name=A-slot-lifetime addr=0x3fbe6fc420 base=0x3fbe6fc420 end=0x3fbe6fc530 tag=1
    PARENT-REF name=list-head addr=0x3fbe6fc420 base=0x3fbe6fc420 end=0x3fbe6fc530 tag=1
    CHECK name=list-head-is-A-lifetime addr=0x3fbe6fc420 bounds=[0x3fbe6fc420,0x3fbe6fc530) ok

On release the list head is the pointer the ledger handed back (`mcp_object_release` returns the
virtual-mallocng object's own pointer). In f1 it was `[0x3fa2ee2040, 0x3fa2ee2140)`, which is A itself.

Through the list head, at A's former storage, control 1 does the following:

* It loads the head's first pointer. This is the free-list link `cache_free` wrote, which is 0 because
  the list held only A. Under patch 0006 that pointer overwrote the lifetime in the slot.
* It reads bytes `[16, 32)`. They hold the `0xa1` the owner wrote through A before the free.
* It writes `0x7e` there and reads it back.

Afterwards `cache_alloc` returns A's address again, holding the parent's `0x7e`. That shows the free list
was left intact.

## How A was shown to come from the cache's own, patched path

`cache.c` hands out from its own storage, the free list, only when the list is not empty. Otherwise it
calls `malloc`. Each control therefore primes the cache: it calls `cache_alloc` (the malloc path) and then
`cache_free`. It then takes A, and in control 2 also B, with `cc_take_from_list`. That helper refuses the
run unless all of these hold:

* `freecurr` drops by one and `total` stays the same, so no `malloc` happened;
* A's address equals the list head's, plus the 16-byte slot under patch 0006;
* **protected:** A's capability is `[head, head + bufsize)` with its cursor at `head + 16`, and the slot,
  read through A, holds a tagged capability with the same address and bounds. That is the shape of
  `get_object(capstone_cap_derive(lifetime, 0, bufsize))` with `*(void **)ret = ret`;
* **release:** the ledger's `object_reuses` rises by one, which is `mcp_object_issue`, the free-list
  branch of patch 0002.

These lines are from f1:

    SOURCE name=A addr=0x3fbe6fc430 path=free-list list_head_before=0x3fbe6fc420 freecurr=1->0 total=1->1
    CHECK name=A-patched-derive-shape object=[0x3fbe6fc420,0x3fbe6fc530) cursor=0x3fbe6fc430 slot_lifetime=[0x3fbe6fc420,0x3fbe6fc530) ok
    SOURCE name=A addr=0x3fa2ee2040 path=free-list list_head_before=0x3fa2ee2040 freecurr=1->0 total=1->1   (release)
    CHECK name=A-ledger-issue object_reuses=0->1                                                              (release)

The size also shows patch 0006's `cache_create` ran: `bufsize=272` on protected against `256` on release.

The images were also checked statically. `llvm-objdump` was run on each image, with `auipc`+`addi`
call targets resolved against `llvm-nm`:

* In the protected `ctl-94`, `do_cache_alloc` calls `capstone_cap_derive` at `0x25098` (the new object's
  lifetime) and at `0x25190` (the hand-out). `do_cache_free` calls `capstone_cap_revoke_child` at
  `0x25438`.
* The callees contain the CDERIVE words `0xa2e395db` and `0xa20515db` (`.insn r 0x5b`, funct3 1, funct7
  0x51) and the CREVOKE word `0xa4b5105b` (funct7 0x52).
* `ctl-95` and `ctl-96` have the same calls.
* On release, `do_cache_alloc` calls `mcp_object_issue` and `mcp_object_backing`, and `do_cache_free`
  calls `mcp_object_release` and `mcp_object_discard`. There are no derive or revoke calls.

The call addresses for every image are in `results.json` under `static_check`.

## Results (rep f1; f2 and f3 agree, with addresses in `results.json`)

| control | arm | outcome | A | B / C | access sites (`controls/lifetime-cache/`) |
|---|---|---|---|---|---|
| 1 parent | protected | completed; link 0, `0xa1` read, `0x7e` written and read back, A reissued holding `0x7e` | 0x3fbe6fc430 | — | via `list-head` (0x3fbe6fc420): load at `94_…/case.c:56`, reads at `:61`, `:63`, write at `:62`; reissue read at `:71` |
| 1 parent | release | completed; same readings | 0x3fa2ee2040 | — | same sites; `list-head` 0x3fa2ee2040 |
| 2 sibling | protected | completed; B's slot unchanged (`0x3faf64f560 [0x3faf64f560,0x3faf64f670) tag=1`), B read `0xb2`, rewritten `0xb5`, then `cache_free(B)` completed | 0x3faf64f430 | B 0x3faf64f570 | slot loads at `95_…/case.c:28` (before) and `:37` (after); data at `:44`, `:45`, `:46` |
| 2 sibling | release | completed; B intact, `cache_free(B)` completed | 0x3fa3b99040 | B 0x3fa3b99180 | data at `95_…/case.c:44-46` |
| 3 successor, `through-c` | protected | completed; C = A after 1 allocation | 0x3f8593f430 | C 0x3f8593f430 | `96_…/case.c:53` (write), `:54` (read) |
| 3 successor, `through-c` | release | completed; C = A after 1 allocation | 0x3f8ea1b040 | C 0x3f8ea1b040 | same sites |
| 3 **stale** | protected | **fault cause=25 pc=0x3f95c90df0**, `read_probe`+20 = `shared/corpus.h:81:10` (link pc 0x20df0); the launcher reports the faulting address 0x3f95d9e440 = A + 0x10 (the bytes the owner wrote) | 0x3f95d9e430 | C 0x3f95d9e430 | stale read at `96_…/case.c:59` through `read_probe` |
| 3 **stale** | release | **completed**; read 0xc3, which is C's data, not A's 0xa1 | 0x3f9c219040 | C 0x3f9c219040 | same site |

The protected stale run's evidence lines in f1:

    ACCESS-BEGIN name=stale-A-read kind=read via=A(stale) addr=0x3f95d9e440 site=…/96_ctl_successor_reuses_object/case.c:59 probe=read_probe@shared/corpus.h
    capstone-exec: domain fault cause=25 pc=0x3f95c90df0 address=0x3f95d9e440 … image=./images/ctl-96
    [CAPSTONE] Cap mem access on revoked capability: pc = 3f95c90df0, rs1 = x10, imm = 0, cursor = 3f95d9e440, bounds = (3f95d9e420, 3f95d9e530)

The protected stale run faulted with cause 25 at the same link pc (`read_probe`+20) in all three reps. The
pcs were 0x3f95c90df0, 0x3f93190df0 and 0x3fa1110df0; they differ only by load address. The faulting
addresses were A + 0x10 each time. The release stale run read 0xc3 in all three reps.

## Prediction verdicts

| prediction (pre-registration) | protected | release |
|---|---|---|
| control 1: completes; the read returns what the allocator left there | holds | holds |
| control 2: completes; B's contents intact | holds | holds |
| control 3: C's start equals A's; access through C completes | holds | holds |
| stale read through A: protected rejects with cause 25 at the probe; release completes and reads C's data | holds (cause 25 at `read_probe`, 3/3) | holds (read 0xc3, 3/3) |

## Provenance

* **llvm-capstone source.** The source is 5d336c73931c (`ports/nested-controls-extended`, on 88d0960bf9ae),
  the pre-registration commit, plus the uncommitted files under `controls/lifetime-cache/`. Their sha256
  values are in `results.json` under `revision.uncommitted_files_used`. No tracked file is modified.
  * The snapshot is a `git archive` of HEAD (`capstone`, `compiler-rt/lib/builtins`) plus those files,
    with `.git` initialised over it and an empty `llvm/` so that the environment script accepts the root.
  * The run host's clock was 26 minutes behind, so the snapshot's mtimes were reset to the past; without
    that, the SDK's ninja looped on a manifest that was always dirty.
* **SDK.** Built from that snapshot (`ports/memcached/app/deps/env.sh`, virtual profile). The key is
  `cb2b69ff311ff186`, the same key as the slab run. Compiler sha256 `4c6135feafae…`.
* **Platform.** All files are byte-identical to the slab run's, and so to the security table's 2026-10-11
  bundle. The run checked these sha256 values before each boot:

  | component | sha256 |
  |---|---|
  | QEMU | `af37cc3269d1…` |
  | launcher | `3975314a7a6d…` |
  | module | `a1c6cb6b79a9…` |
  | kernel | `e833dfb12bcc…` |
  | firmware | `44cd6cbc2d22…` |
  | rootfs | `62049800ed0f…` |

  The kit files are copies held in the run's own directory, and each boot used its own QEMU lock.
* **Images** (sha256):

  | program | protected | release |
  |---|---|---|
  | ctl-94 | `c3331fe70259…` | `362bcd25a569…` |
  | ctl-95 | `d9425fdadd4c…` | `3e462b7a2dd2…` |
  | ctl-96 | `982e6808877d…` | `657ff1bfb3d0…` |

  The full hashes are in `results.json`. The launcher's fault lines carry the same `ctl-96` hash.
* **Runner.** `run-controls.py` sha256 `2bf6bfd9c185…`.
* **Wall time.**
  * The SDK took 33 s cold, and then 2 s.
  * The port and the three controls took 2–5 s per arm.
  * Each boot took 8.8–9.4 s, and each `run.sh` took 15 s.
  * All six boots ran in parallel, one VM each.

## Deviations, and what this does not show

* **Harness location.** The cases are numbered 94-96 in a sibling directory, `controls/lifetime-cache/`,
  not in `controls/lifetime/`. The slab build script requires exactly three cases there.
* **A superseded first boot pair (r1, kept on the run host).** It used an earlier build of these
  controls. On protected, control 95 refused itself with `CONTROL-FAILED 931`. That was a harness check
  that added the 16-byte slot to `primed[]`, which already held object addresses. It passed on release
  only because the slot is 0 there. Every other r1 run gave the outcomes recorded here, including cause 25
  at `read_probe` for the stale read.

  The fix changed only that comparison and the file name printed for the slot reads (a header name
  printed with a `case.c` line). The images were rebuilt, and f1-f3 are on the rebuilt images.
* **The release build is the registered release arm, not patch 0006 removed.** As in the slab run,
  release is the shim patch plus the port's lifetime-hooks patch (0002), with the stock ledger in mode 0.
  `cache.c` objects come from virtual mallocng through `mcp_object_backing` and are never revoked.
* **Additions beyond the pre-registration.** The pre-registration does not ask for these, and none of them
  is used to judge a prediction:
  * the link read in control 1;
  * the reissue check in control 1;
  * the slot comparison and the final `cache_free(B)` in control 2;
  * the priming that puts A on the free list;
  * the static disassembly check.
* **What these controls do not exercise.**
  * They use one cache and one thread, with no limit, so `cache_free` never takes the over-limit branch
    (`free`).
  * They do not call `cache_destroy`.
  * The hand-out derive also runs on the malloc path, but A is always taken from the free list. That is
    the cache's own storage, as the pre-registration asks.

## On silicon

The same controls ran on silicon on 2026-10-16 (board set 6, `caplifive_sublet_d93a8d66b`). Their readings sit beside
this record's in `../2026-10-16-silicon-controls/`: every silicon run reads as the same control does here.
