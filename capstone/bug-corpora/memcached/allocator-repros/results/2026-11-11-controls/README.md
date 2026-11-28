# memcached slab allocator: parent, sibling and successor controls, 2026-11-11

The controls of [`../2026-10-11-controls`](../2026-10-11-controls/README.md), run again after patch
0006 changed and the processor model changed its lifetime instructions. The pre-registration is
unchanged: [`docs/plans/2026-10-11-nested-lifetime-controls.md`](../../../../../docs/plans/2026-10-11-nested-lifetime-controls.md).
The protected arm is `virtual-nested-pools` (`-DMCP_SUBLET=ON`, `virtual-memcached-sublet`), the
release arm is `virtual-malloc` (`-DMCP_STOCK_MALLOC=ON`, `virtual-memcached-stock`).

**Every prediction held: 9 runs, one boot per arm.** The stale access completed on release and read
C's data. On protected it faulted at the labelled probe with cause 25. The 2026-10-11 record ran
three boots per arm; this one ran one, because the boots queued behind a shared lock.

## What changed under the controls

Patch 0006 no longer derives a page lifetime in `memory_allocate`. `slab_list[0]` is the pointer
`malloc` returned for the page. `split_slab_page_into_freelist` derives the class's generation
directly from it, and `do_slabs_alloc` hands A out as a child of the generation. `slabs_free` ends A
with the one-operand `CREVOKE(A)`. So the chain has one level less than on 2026-10-11:

    virtual-mallocng object (slab_list[0]) -> generation (chunk byte 0) -> A

Control 1 still runs twice on protected. `page` goes through `slab_list[0]`, now A's grandparent
and the allocation itself. `generation` loads the generation from slot 0 of A's former chunk and
goes through it; it is A's parent, and `CREVOKE(A)` does not name it.

## Results

| control | arm | outcome | A | B / C | bounds the program printed |
|---|---|---|---|---|---|
| 1 parent, `page` | protected | completed | 0x3fa518c780 | reissued at A | page `[0x3fa50bd800, 0x3fa51bd800)`, A `[A, A + 105968)` |
| 1 parent, `generation` | protected | completed | 0x3f964cd780 | reissued at A | generation `[0x3f963fe800, 0x3f964e7570)`, tag 1: the page base plus 9 x 105968 |
| 1 parent, `page` | release | completed | 0x3f90127100 | reissued at A | A carries the page's bounds `[0x3f9004e800, 0x3f9014e800)` |
| 2 sibling | protected | completed; B's header and data intact | 0x3f7fdcd780 | B 0x3f7fdb3990 | B `[B, A)`, A `[A, A + 105968)` |
| 2 sibling | release | completed; B intact | 0x3fad237100 | B 0x3fad21f000 | both carry the page's bounds |
| 3 successor, `through-c` | protected | completed; C = A after 1 allocation | 0x3fbcf8c780 | C 0x3fbcf8c780 | C `[A, A + 105968)` |
| 3 successor, `through-c` | release | completed; C = A after 1 allocation | 0x3f96e33100 | C 0x3f96e33100 | page bounds |
| 3 **stale** | protected | **fault cause=25 pc=0x3faeb8de18**, `read_probe`, `shared/corpus.h:81:10` (link pc 0x1de18); faulting address 0x3fae18c7e0 = A + 0x60 (`data[0]`) | 0x3fae18c780 | C 0x3fae18c780 | A and C `[A, A + 105968)` |
| 3 **stale** | release | **completed**; read 0xc3, C's data | 0x3f8f02f100 | C 0x3f8f02f100 | page bounds |

Access sites are the 2026-10-11 record's: `91_.../case.c:58` (slot 0), `:69`-`:77`, `:85`;
`92_.../case.c:28`-`:36`, `:59`; `93_.../case.c:53`, `:54`, and the stale read at `:59` through
`read_probe`. The class is 30 on both arms: chunk size 105968 with 9 chunks per page on protected
(`sizeof(item)` 96), 98560 with 10 on release (`sizeof(item)` 80). In control 1 the parent reads
`it_flags` 0x4 and `slabs_clsid` 30, reads the owner's 0xa1, writes 0x7e, and the next `slabs_alloc`
returns A's chunk holding 0x7e.

Each boot ran the virtual heap's own controls first: `uaf-malloc` faulted with cause 25 and
`bounds-malloc` with cause 28, on both arms.

## Prediction verdicts

| prediction (pre-registration) | protected | release |
|---|---|---|
| control 1: completes; the read returns what the allocator left there | holds (`page` and `generation`) | holds |
| control 2: completes; B intact | holds | holds |
| control 3: C's start equals A's; access through C completes | holds | holds |
| stale read through A: protected rejects with cause 25 at the probe; release completes and reads C's data | holds (cause 25 at `read_probe`) | holds (read 0xc3) |

## Provenance

`results.json` carries the runner's record of each boot, every image hash, the hash of patch 0006
and of each control file, and the platform's file hashes.

* **Platform.** QEMU `2d5a7107...`, launcher `0f8e0f90...`, module `e47bca62...`, kernel
  `48eb725a...`, firmware `5b5c047a...`, rootfs `62049800...`. `run-controls.py` registers this
  platform beside the 2026-10-11 one and refuses any other.
* **Options.** `CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`,
  `-smp 1`, module loaded without parameters. Exact bounds is the profile of the 2026-10-11 record.
  Under the model's default compact bounds the protected arm gives no reading. Measured on the
  corpus control `chunk-free-90`: the `CDERIVE` of the generation faults with cause 29, because
  `size * perslab` bytes from the page base is not a representable range for most classes. Here
  the generation is 9 x 105968 = 953712 bytes, and a range of that length needs a multiple of 1024.
* **SDK.** The corpus SDK (`CAPSTONE_SDK_CORPUS=1`, `-fno-builtin`), virtual profile, mallocng.
* **Lifetime instructions** in the protected images outside libc's heap: `CDERIVE` in
  `capstone_cap_derive` and `CREVOKE` in `capstone_cap_release` (the port library is a Debug build,
  so the two helpers stay functions). The release images carry none.

## What these controls do not exercise

One class and one thread. The page mover is not built in this port, so the release of a generation
in `slabs_finalize_page_move` does not run here.
