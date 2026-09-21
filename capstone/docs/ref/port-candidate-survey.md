# Port-candidate survey — which software is worth porting next

*Assembled 2026-09-21. Answers: among widely-deployed C/C++ software, which
programs have a nested allocator of our shape, and how many reported defects
does the best pin give. Supersedes nothing; complements
`paper-bug-inventory.md` (feasibility rules) and `whisper-ggml-defects.md`
(the ggml negative result).*

## The two constraints that decide this, before defect counts

From `paper-bug-inventory.md`, verified 2026-09-21:

* **Pure C is the only thing that compiles for `capstone64`.** C++ requires a
  distilled C shim gated by `xlang/cheri/check_shim_fidelity.py`.
* **No true concurrency** — single hart, no threads. A defect needing a racing
  core is out; callback re-entrancy is in.

Applying these *first* reorders the field completely, and doing so late cost a
wrong recommendation in this very survey: ONNX Runtime was called the strongest
candidate before either filter was applied. Both rules were already written
down. **Screen for language and concurrency before spending research on counts.**

## Screening criteria

All four must hold:

* **(a)** backing block obtained once from the system allocator, carved into objects;
* **(b)** an object's lifetime can end while the backing block STAYS ALLOCATED and
  its storage is re-handed-out;
* **(c)** user handles are **raw interior pointers**, not indices, offsets or
  generation-checked slots — a pointer-capability mechanism cannot see an index dangle;
* **(d)** the allocator is on the **default** path, not an opt-in mode.

Cheap discriminator: base chunks obtained from the system allocator vs objects
handed out. Near 1:1 is a pass-through wrapper (Redis: 77 base chunks for 26.8M
objects). Thousands-to-one is a real arena (PHP: 144k for 66.4M).

## Standing, pure C first

| Candidate | Lang | In-class defects | Pin | State |
|---|---|---|---|---|
| **Wireshark `wmem`** | C | **17** (floor) | `v4.6.8` | 2 out, 6 unresolved; runtime allocator switch — see below |
| **Varnish workspaces** | C | **9, unconditional** | `varnish-7.7.3` (`6884b75af9c9`) | 386 LOC; simplest port in the survey |
| MicroPython | C | 9 (CMASan's count, not ours) | — | port exists, corpus does not |
| **memcached** | C | **8** | `1.6.45` | in progress |
| MariaDB `MEM_ROOT` | C allocator, **C++ consumers** | ~4 distinct / 7 records | `mariadb-12.3.3` (`bd852bcd567f`) | consumers need shims |
| APR | C | 1 + CVE-2017-9798 | 1.7.4 (in-repo) | already ported, census written |
| OpenSSL secure heap | C | 2 | 3.6.1 | |
| `ggml_gallocr` | C | 1 CPU-only | whisper.cpp 1.9.4 | **tree already pinned by us** |
| expat | C | grow-relocation, CVE-2022-40674 | — | fills a thin taxonomy row |
| ONNX Runtime `BFCArena` | **C++** | 6 at pin, ~2 after the concurrency filter | `v1.29.1` (`d9d3b2fc25`) | best *evidence*, not a cheap port |
| RapidJSON / RocksDB / Godot / ArduinoJson / PyTorch | **C++** | 3 / 1 / 1 / 1 / 1 | — | all shim-bound |

**Disqualified with reason:** upb — its allocator pin and its defect pin are
mutually exclusive (`8cbbcec6e080` has the pool and 0 defects; `v36.2` has the
defect and no pool). gRPC arenas: 0 in-class. nlohmann/json: `get_allocator()`
is static, so it structurally cannot hold a stateful arena. sajson: pure bump.
Bitcoin Core, zstd, Redis: 0.

## Varnish — the defensible corpus, and why it beats a bigger count

`bin/varnishd/cache/cache_ws.c` at `varnish-7.7.3`, 386 lines, pure C. Verified
by reading it: `WS_Reset` is assertions plus `ws->f = p` — a cursor rewind —
and the file contains **no `malloc`, `calloc`, `realloc` or `free` at all**
(positive control: the same grep form finds 40 assertion calls).

A workspace is one fixed buffer. The backing storage therefore *cannot* be
returned, so there is no `gc` equivalent and no escape hatch, and there is no
debug-allocator caveat because there is no allocator to substitute. Its **9
in-class defects are unconditional**.

Varnish is the **simplest** port in the survey, and its in-class verdict needs
no inference at all. It is not, however, the richest: see the correction to the
Wireshark entry, which restores that project to first place on defect count,
allocator variety, and one capability Varnish structurally cannot offer.

Note the pin carries no live reproductions, and that is the normal mode for
this corpus rather than a defect: cases are built as **model consumer, real
allocator** (the phrase `case.json` uses), the same way the eight PostgreSQL
defects were backported onto one base. What the pin must supply is the
allocator; the consumers we write.

## Wireshark — verified at source, and the caveat that remains

`wmem_block_free_all` (`wsutil/wmem/wmem_allocator_block.c`, 1125 LOC, pure C,
~4 libc symbols, builds standalone):

```c
/* the existing free lists are entirely irrelevant */
allocator->master_head   = NULL;
allocator->recycler_head = NULL;
/* iterate through the blocks, reinitializing each one */
while (cur) { ... wmem_block_init_block(allocator, cur); cur = cur->next; }
```

Scope reset **retains** the OS blocks and re-issues their storage; only jumbo
blocks go back to the system. Criteria (a)-(d) all hold, confirmed by reading
the allocator, not a description of it.

### Pin: v4.6.8, and the allocator has been frozen for two years

Measured in a full clone at the tag:

* `wmem_allocator_block.c` and `wmem_allocator_block_fast.c` are **blob-identical
  across v4.4.0 (2024-08-28) → v4.6.8 (2026-08-12)**, blobs `4723f0f368af` and
  `538f51878bc5`. The only change through the v4.7.3 development line is one
  commit, `186691b2c8 "Spelling correction"`, 1 insertion / 1 deletion.
* Change rate: `block.c` 5 commits since 2020, `block_fast.c` 3.
* v4.2.0 and earlier differ; v3.x does not have the files at this path.

So **v4.6.8** — newest stable, same blobs the ASan traces' line numbers refer
to (19399 cites `:924` and `:1070` of `4723f0f368af`), two years newer than the
v4.4.18 first proposed. Avoid v4.7.x on principle, not because of the typo.

### Pools in a default run: 12 sites, 5 nested

Enumerated at the pin (`wmem_allocator_new` call sites, tests excluded):

| site | pool | backend | nested |
|---|---|---|---|
| `epan/wmem_scopes.c:126` | `packet_scope` | `block_fast` | yes |
| `epan/wmem_scopes.c:127` | `file_scope` | `block` | yes |
| `epan/wmem_scopes.c:128` | `epan_scope` | `block` | yes |
| `epan/epan.c:575` | `edt->pi.pool`, per dissection, recycled via `pinfo_pool_cache` | `block_fast` | yes |
| `epan/addr_resolv.c:4088` | `addr_resolv_scope` | `block` | yes |
| 7 further sites | tvbuff_lznt1 / lz77 / lz77huff, packet-epl ×2, btmesh-proxy, dfilter | `simple` | **no** |

`simple` is a pass-through: `wmem_simple_alloc` does one `wmem_alloc(NULL, size)`
per request and records the pointer, so ASan sees it. `strict` likewise
(`wmem_alloc(NULL, WMEM_FULL_SIZE(size))` plus canaries) — which is exactly why
it is the arm under which all 17 were reported.

The legacy `emem` framework (`se_alloc`/`ep_alloc`) is dead: nowhere declared,
remaining occurrences sit inside `#if 0`.

**Porting more backends does not multiply cases.** All 17 confirmed defects are
packet-scope, i.e. `block_fast`. `block` unlocks the 2 excluded file-scope ones
and at least 2 of the 6 unresolved, and is worth porting anyway because it is
the **recycler** — free lists with split and coalesce, the pymalloc/aset shape,
which `block_fast` (a bump) and Varnish (a cursor) do not provide. `simple`
adds nothing. `strict` adds no cases and should still be ported: it is the
control arm.

### Correction: the gc objection applies to the wrong scope

An earlier version of this file demoted Wireshark on the grounds that
`wmem_block_gc` returns blocks to the OS. **That objection does not touch the
17 in-class defects.** Verified at source:

* `epan/wmem_scopes.c:126-128` — **`packet_scope` is `WMEM_ALLOCATOR_BLOCK_FAST`**;
  only `file_scope` and `epan_scope` are `WMEM_ALLOCATOR_BLOCK`.
* `wmem_block_fast_gc` is literally `{ /* No-op */ }`. **There is no gc escape
  hatch in packet scope at all.**
* `wmem_block_fast_free_all` — *"freeing all but the first and reinitializing
  that one"* — keeps the first block and rewinds its cursor
  (`cur->pos = WMEM_BLOCK_HEADER_SIZE`), `WMEM_BLOCK_SIZE` = 2 MiB. That is the
  same mechanism as Varnish's `WS_Reset` (`ws->f = p`).

All 17 in-class defects are packet-scope. The two excluded ones (19265, 19399)
are file-scope, reached via `wmem_leave_file_scope` → `wmem_gc`. So the gc
objection excluded exactly the two that were already out and none of the 17.

The residual condition on the 17 is only whether the stale object lives in the
retained first 2 MiB block. **That is under our control**, because corpus cases
are written as model consumers: the reproducer decides how much packet-scope
memory it allocates, so first-block residency becomes a design constraint to
assert, not an uncertainty to carry — exactly as
`10_gh-142560_bytearray_search_realloc` asserts that its realloc moved the block.

### Two release paths, and they fall on opposite sides of the in-class line

Verified by reading the allocator at the pin blob:

* `wmem_block_free_all` — the **scope reset**. Retains the OS blocks and
  reinitialises each one; only jumbo blocks go back. Backing block stays
  allocated, storage re-handed-out. **IN CLASS.**
* `wmem_block_gc` (line 1032ff) — *"If the first chunk is also the last, and is
  unused, then the block as a whole is entirely unused, so **return it to the
  OS**"*, then `wmem_free(NULL, cur);` at line 1070. Backing block **released**.
  An ordinary heap UAF that ASan sees in any build. **OUT OF CLASS**, by the
  same criterion that excluded ggml #24292 and whisper's CVE-2025-14569.

This reclassified issue 19399, reported at first as the most on-target defect
found *because* `wmem_allocator_block.c` appears as both the freeing and the
faulting frame. Its freeing frame is `wmem_block_gc ... :1070` — the
return-to-OS call. The property that made it look best is the one that
disqualifies it. 19265 dies on the same call.

**Partitioned: 17 IN, 2 OUT, 6 UNRESOLVED**, from 24 reported. Both OUT are
file-scope; all 17 IN are packet-scope, where gc is a no-op (see the correction
above). Two of the
unresolved (11740, 21339) fire on capture-file close, which is
`wmem_leave_file_scope` → `wmem_gc` — the path that killed the other two.

A further refinement, verified at `epan/wmem_scopes.c:126-128`: production
**packet scope is `BLOCK_FAST`, not `BLOCK`**, and `wmem_block_fast_free_all`
retains only the *first* block (`WMEM_BLOCK_SIZE` = 2 MiB), returning every
later one; `wmem_block_fast_gc` is a no-op. All 17 are packet-scope-family, so
their production reset frame is `wmem_block_fast_free_all`.

There is a tension to resolve for every defect that survives: if the block was
retained, ASan in a normal build could not have reported it. Either the report
came from the strict/simple allocator build, or gc had run. A defect reported
only under the strict allocator is still usable — our port models the production
block allocator, so what matters is whether the pointer would land in a
retained, reused block in production — but that is an inference to state, not
to hide.

**The caveat, and it is load-bearing:** the ASan traces on those reports come
from the **strict** allocator build. A trace evidences the *stale pointer*; it
does not evidence *block retention* for that particular defect. Block retention
is now proven at the allocator level, but per defect it still has to be shown
that the read follows a **scope reset** (`free_all`) rather than an allocator
teardown. That is a per-case check when the cases are built, not a blocker.

Generalised: **an ASan signature identifies the poison, not the mechanism.**
"Block still allocated" and "storage reused" are two separate checks, and the
second needs the reset call named and a read shown after it.

## Detectability — the reason this class is under-reported

Eleven upstream admissions collected. The two strongest are shipped code, not
issue comments:

* **Wireshark** ships an allocator substitution whose purpose is to make the
  class visible: `getenv("WIRESHARK_DEBUG_WMEM_OVERRIDE")` in
  `wmem_core.c`, with the comment *"Our valgrind script uses this environment
  variable to override the..."*. Production runs `block`; the tools see only
  `strict`/`simple`.
* **ONNX Runtime** disables its own arena under ASan, in
  `onnxruntime/core/framework/allocator_utils.cc:83`:
  `// Using the arena may hide memory issues. Disable it in an ASan build.`
  Consequence: its one ASan CI job never executes `BFCArena` on CPU. An
  ASan-clean result there is evidence the arena was absent.

Others: PyTorch's source comment *"the caching allocator foils cuda-memcheck"*
plus issue #58385 (a poison-on-free proposal, open since 2021, zero comments)
and `PYTORCH_NO_CUDA_MEMORY_CACHING` with zero hits in `.ci/`/`.github/`;
llama.cpp #27096 — *"`test_case` ... does not use `ggml_gallocr`. Thus it cannot
make `dst` an alias of `src1`, and it cannot show the bug"* — plus OSS-Fuzz
"Fuzzers - failing" and upstream issue #11514 admitting the fuzzers have been
broken "for a long time"; RocksDB PR #9770 ("RFC: help ASAN with Arenas", open
since 2022, never merged); APR's `APR_POOL_DEBUG` and Varnish's `ws_emu`, two
bespoke sanitizers built because the general one does not work; the Optionsbleed
writeup for CVE-2017-9798 stating ASan *"doesn't work reliably due to the memory
allocation abstraction done by APR"*; Bitcoin PR #32581; Godot PR #94906;
RapidJSON #1078 and ArduinoJson #1712, both closed as by-design.

## Two sub-shapes, and one is nearly unrepresented

Confirmed **0 grow-relocation** in Wireshark and Varnish both, with the two
nearest misses (19399, 14248) individually refuted rather than merely absent
from a search.

* **(a) scope-reset** — pool cleared, storage re-handed-out, pointer survives.
  This is what essentially all 32 existing corpus cases are.
* **(b) grow-relocation** — `realloc` moves the block, interior pointers not
  rebased. **We have exactly one specimen**,
  `cpython/pymalloc-repros/10_gh-142560_bytearray_search_realloc`, whose own
  `case.json` says it is *"the only case in the corpus whose block is ended by a
  realloc rather than a free"*. Candidates: expat (PR #85, CVE-2022-40674),
  Samba (CVE-2022-32746). Not represented in Wireshark, Varnish, MariaDB or upb.

## ggml — a cheap addition, separately from llama.cpp

`ggml/src/ggml-alloc.c` is **byte-identical** between whisper.cpp 1.9.4, which
`ports/whisper/ggml-context/upstream.json` already pins, and llama.cpp b11067
(md5 `f1ac3a24e54e772de90afa3ec1f3af46`, 1249 lines, `diff` empty with a firing
control). The existing port covers only `ggml_context`, a bump arena with
`ggml_reset` that llama.cpp's own code never calls — one caller tree-wide. The
graph allocator `ggml_gallocr`, which has real free lists and deliberate
cross-tensor reuse, is unported and needs **no new upstream**.

Porting *llama.cpp* is a different and worse proposition: 3 qualifying defects
at the best pin (`b7358`), 1 runnable CPU-only, 0 at tip. Its value is the
allocator, which we already have.

## Recommendation

1. Finish **memcached** (8, pure C).
2. Take **Wireshark `wmem`** next. 17 in-class defects, packet scope has no gc
   escape hatch, and the allocator is 1125 + 300 lines of standalone C behind
   one interface with **four interchangeable backends**
   (`block`, `block_fast`, `simple`, `strict`).

   The decisive argument is one Varnish structurally cannot match:
   `WIRESHARK_DEBUG_WMEM_OVERRIDE` switches the allocator **at runtime**, so a
   single reproducer runs under `strict` (ASan fires — this is how all 17 were
   reported) and under the production `block_fast` (ASan silent, our mechanism
   fires). That turns the paper's central claim from an assertion into a
   controlled A/B inside one program, with a positive control built in by
   upstream. Varnish has no second allocator to switch to.

   Strongest individually-defensible subset, if rows are challenged one by one:
   the repeated signatures — `print_columns` two-pass (21261, 20587, 19960) and
   `find_string_dtbl_entry` (17835, 17809) — the same shape across four years.
3. Take **Varnish workspaces** after it, or first if a fast result is wanted:
   386 lines, 9 rows needing no inference. It is the cheapest port here.
4. Port **`ggml_gallocr`** from the tree already pinned. Low defect yield, near
   zero acquisition cost, and it completes an existing port.
5. Build a **grow-relocation** row from expat, for taxonomy coverage rather
   than volume.

## What was not enumerated

MySQL is not enumerable — bugs.mysql.com returned zero result links even for a
`search_for=crash` positive control. ASF Bugzilla is auth-only, so the APR sweep
is search-engine-derived. MariaDB `text~"use-after-poison"` alone returns 842
issues; 22 were classified in depth, so its count is a floor. Unexamined MariaDB
parents: MDEV-16043, MDEV-26407, MDEV-24176.
