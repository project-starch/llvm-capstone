# memcached slab allocator: parent, sibling and successor controls, 2026-10-11

This is memcached's row of the pre-registration
[`docs/plans/2026-10-11-nested-lifetime-controls.md`](../../../../../docs/plans/2026-10-11-nested-lifetime-controls.md).
The nested allocator is the slab chunk allocator (`slabs_alloc` / `slabs_free`). The protected build
is the application's patch 0006 (`-DMCP_SUBLET=ON`, which compiles the controls with
`-DMC_CAPSTONE_SUBLET`). That is the `virtual-nested-pools` arm, `virtual-memcached-sublet` in
`tools/arms.json`, as in the security table. The release build is the same corpus's `virtual-malloc`
arm (`-DMCP_STOCK_MALLOC=ON`, `virtual-memcached-stock`).

**Every prediction held: 27 runs, 3 boots per arm.** The stale access is the positive control. It
completed on release and read C's data. On protected it faulted at the labelled probe with cause 25.

## Where the controls are and how they run

The controls are in `controls/lifetime/`:

| program | file | what it does |
|---|---|---|
| `ctl-91 91 page` / `ctl-91 91 generation` | `91_ctl_parent_page_reference/case.c` | control 1, parent |
| `ctl-92 92 sibling` | `92_ctl_sibling_same_page/case.c` | control 2, sibling |
| `ctl-93 93 through-c` / `ctl-93 93 stale` | `93_ctl_successor_reuses_chunk/case.c` | control 3, successor; the stale read runs separately |

The controls call slabs.c directly, as the corpus cases do. The binaries are built by
`controls/lifetime/build-controls.sh`, which is a copy of the `capstone-application` branch of
`shared/build-cases.sh` (same port build, same flags) linked with its own `driver-controls.c`. They
are run by `controls/lifetime/run-controls.py`, which uses one boot per arm and repetition through
`tools/virtualvm.py`. The boot uses `runtime/virtual/run-staged.py` with `-smp 1` and
`x-capstone-exact-bounds=true`. Before the controls, each boot runs the virtual heap's own controls
from `postgres/c-repros` (`uaf-malloc`, `bounds-malloc`). Both faulted in all six boots.

Each control uses one slab class: a 96 KiB request, class 30. The geometry differs between arms
because patch 0006 adds a 16-byte slot to every item:

| arm | sizeof(item) | chunk size | chunks per page |
|---|---|---|---|
| protected | 96 | 105968 | 9 |
| release | 80 | 98560 | 10 |

Each access goes through a noinline probe whose only load or store is that access. The program
prints the access before it runs and confirms it afterwards, with the call site, the probe and the
address. That way a fault is tied to one access by the console output and also by the pc.

| probe | source line |
|---|---|
| `ctl_read_u8` | `controls/lifetime/controls.h:48` |
| `ctl_write_u8` | `controls/lifetime/controls.h:51` |
| `ctl_load_ptr` | `controls/lifetime/controls.h:55` |
| `read_probe` (the stale access) | `shared/corpus.h:81` |

Under patch 0006, byte 0 of every chunk holds its generation, and the free-list link (`next`,
`prev`), `it_flags` and `slabs_clsid` sit in the item header. No control writes the first
`sizeof(item)` bytes of a chunk. Every write goes to `data[0..16)`, which starts at offset
`sizeof(item)`.

## Which reference is the parent (control 1)

`slabs_peek_page(id, ...)` returns `slab_list[0]`, the reference slabs.c keeps for the class's first
page. It is the page mover's way into the page, and it answers only once the class has two pages.
To get there, each control fills page 0 and opens page 1; those filler chunks stay live.

Under patch 0006, `slab_list[0]` is the page's lifetime. `memory_allocate` derives it with CDERIVE
from the virtual-mallocng object, bounded to the 1 MiB page. It is **A's grandparent**, not A's
parent. `split_slab_page_into_freelist` derives the class's **generation** from it, bounded to
`perslab × size`, and writes that generation into byte 0 of every chunk. `do_slabs_alloc` hands A out
as a child of the generation. `slabs_free` performs `CREVOKE(generation, A)`. So the chain is:

    virtual-mallocng object -> page lifetime (slab_list[0]) -> generation (chunk byte 0) -> A

Control 1 therefore runs twice on protected:

* **`page`** goes through `slab_list[0]`. In run f1 the page reference was
  `[0x3f981fe050, 0x3f982fe050)`, 1 MiB.
* **`generation`** goes through A's direct parent, the CREVOKE parent. It loads the generation from
  A's former chunk, slot 0, through `slab_list[0]`. In run f1 the generation was
  `[0x3fb4efe050, 0x3fb4fe6dc0)` with tag 1, which is the page base plus 9 × 105968.

On release, `slab_list[0]` is virtual mallocng's own pointer to the page and there is no generation,
so only `page` runs.

Through the parent reference, at A's former chunk, control 1 does the following:

* It reads `it_flags` and `slabs_clsid`. These are what `do_slabs_free` wrote: `0x4` (ITEM_SLABBED)
  and 30.
* It reads `data[0..16)`. This is the `0xa1` the owner wrote through A before the free, which
  `slabs_free` leaves in place.
* It writes `0x7e` to `data[0..16)` and reads it back.

Afterwards `slabs_alloc` returns A's chunk again, holding the parent's `0x7e`. That shows the free
list was left intact.

## Results (rep f1; f2 and f3 agree, with addresses in `results.json`)

| control | arm | outcome | A | B / C | access sites (`controls/lifetime/`) |
|---|---|---|---|---|---|
| 1 parent, `page` | protected | completed | 0x3f982ccfd0 | — | reads at `91_…/case.c:69,70,71,75,77`; write at `:76`; via `slab_list0` |
| 1 parent, `generation` | protected | completed | 0x3fb4fccfd0 | — | load of slot 0 at `91_…/case.c:58`, then reads and writes as above via `generation` |
| 1 parent, `page` | release | completed | 0x3fa15d2960 | — | same sites as protected |
| 2 sibling | protected | completed; B's header and data intact | 0x3f96bccfd0 | B 0x3f96bb31e0 | `92_…/case.c:28-36` (header), `:59`, `:72-74` (data) |
| 2 sibling | release | completed; B intact | 0x3f83cd2960 | B 0x3f83cba860 | same sites |
| 3 successor, `through-c` | protected | completed; C = A after 1 allocation | 0x3f818ccfd0 | C 0x3f818ccfd0 | `93_…/case.c:53` (write), `:54` (read) |
| 3 successor, `through-c` | release | completed; C = A after 1 allocation | 0x3f906d2960 | C 0x3f906d2960 | same sites |
| 3 **stale** | protected | **fault cause=25 pc=0x3fb5190d94**, `read_probe`+20 = `shared/corpus.h:81:10` (link pc 0x20d94); the launcher reports the faulting address 0x3fb478c030 = A + 0x60 (`data[0]`) | 0x3fb478bfd0 | C 0x3fb478bfd0 | stale read at `93_…/case.c:59` through `read_probe` |
| 3 **stale** | release | **completed**; read 0xc3, which is C's data, not A's 0xa1 | 0x3f9aed2960 | C 0x3f9aed2960 | same site |

The protected stale run faulted with cause 25 at the same link pc (`read_probe`+20) in all three
reps. The pcs were 0x3fb5190d94, 0x3fad590d94 and 0x3fb6c10d94; they differ only by load address.
The release stale run read 0xc3 in all three reps.

The bounds the programs printed show what each arm gives A:

* **Protected:** A's capability is the chunk, `[A, A + 105968)`. B ends where A begins. C is a new
  child with the same bounds as A.
* **Release:** every chunk reference, including A, B and C, carries the whole page's bounds,
  `[page, page + 1 MiB)`. That is why the stale read completes there.

## Prediction verdicts

| prediction (pre-registration) | protected | release |
|---|---|---|
| control 1: completes; the read returns what the allocator left there | holds (`page` and `generation`) | holds |
| control 2: completes; B intact | holds | holds |
| control 3: C's start equals A's; access through C completes | holds | holds |
| stale read through A: protected rejects with cause 25 at the probe; release completes and reads C's data | holds (cause 25 at `read_probe`) | holds (read 0xc3) |

## Provenance

* **llvm-capstone source.** 53325bba0d5c (`ports/nested-lifetime-controls`) plus the uncommitted
  files under `controls/lifetime/`; their sha256 values are in `results.json` under
  `revision.uncommitted_files_used`. No tracked file is modified. The snapshot is a `git archive` of
  HEAD (`capstone`, `compiler-rt/lib/builtins`) plus those files, with `.git` initialised over it so
  the SDK key can list its sources.
* **SDK.** Built from that snapshot (`ports/memcached/app/deps/env.sh`, virtual profile), key
  `cb2b69ff311ff186`. Compiler sha256 `4c6135feafae…`, the same toolchain as the security-table run.
* **Platform.** All files are byte-identical to the security table's 2026-10-11 bundle:

  | component | sha256 |
  |---|---|
  | QEMU | `af37cc3269d1…` |
  | launcher | `3975314a7a6d…` |
  | module | `a1c6cb6b79a9…` |
  | kernel | `e833dfb12bcc…` |
  | firmware | `44cd6cbc2d22…` |
  | rootfs | `62049800ed0f…` |

  The kit files are copies held in the run's own directory, each hashed before the run, and the run
  uses its own QEMU lock.
* **Images** (sha256):

  | program | protected | release |
  |---|---|---|
  | ctl-91 | `87f41d7c144f…` | `1f76392543265…` |
  | ctl-92 | `50c6ee9ae2a9…` | `621b81e21546…` |
  | ctl-93 | `c19d1c66037d…` | `3505c02cba30…` |

  The full hashes are in `results.json`.
* **Runner.** `run-controls.py` sha256 `48d5237f247d…`.
* **Wall time.** The SDK took 8 s cold; the port and the three controls took 2–3 s per arm. Each
  boot took 10.4–11.2 s, and each `run.sh` took 17–18 s. All six boots ran in parallel, one VM each.

## Deviations, and what this does not show

* **API spelling.** In 1.6.45 the call is `slabs_alloc(id, flags)`. There is no size argument; the
  class comes from `slabs_clsid(size)`.
* **The release build is the registered release arm, not patch 0006 removed.**
  * Protected is the port's shim patch plus application patch 0006, with no hooks and no ledger.
  * Release is the shim patch plus the port's lifetime-hooks patch (0002), with the stock ledger in
    mode 0. In that build each page is one virtual-mallocng object and the hooks hand back page
    pointers, revoking nothing.

  This is the `virtual-malloc` arm the security table reports.
* **The SDK differs from the security table's.** That run's SDK, key `07c817090cfcfe54`, was built
  from an earlier commit. This one is built from the snapshot, as the task requires. The platform
  and the compiler are identical.
* **Additions beyond the pre-registration.** The pre-registration does not ask for these, and none
  of them is used to judge a prediction:
  * the `generation` variant of control 1;
  * the free-list check at the end of control 1;
  * the header comparison in control 2;
  * a fill of page 0 so that `slabs_peek_page` answers.
* **A superseded first boot pair.** A first boot pair (r1, kept on the run host) used an earlier
  build of the same controls and gave the same outcomes. It was rerun only to record call sites as
  `case.c` lines rather than helper lines, and to strip host paths from the records.
* **What these controls do not exercise.** They use one class and one thread. They do not exercise
  the page mover (`slabs_finalize_page_move`, which is not built in this port) or cache.c's half of
  patch 0006.
