# Is the old capability after realloc() stale but still accessible on CheriBSD?

C17 7.22.3.5 ends the old object's lifetime on every successful `realloc`, whether or not the
returned address is the same. This probe measures what CheriBSD's libc does with the OLD
capability in seven shapes, under both of its layers:

- revocation on (the default): the malloc revocation shim (`lib/libc/stdlib/malloc/mrs/mrs.c`)
  in front of the CHERI-patched jemalloc
- revocation off (`_RUNTIME_REVOCATION_DISABLE=1`): jemalloc alone

`probe.c` runs each scenario in a forked child, prints both capabilities, the old one's tag at the
moment of the access, then reads one byte THROUGH THE OLD CAPABILITY at a labelled `clbu`. The
parent reports completion or SIGPROT. `run.sh` builds it for riscv64-purecap and runs it in one
CheriBSD boot behind the ABI probe, the bounds probe and `revocation-control` (a forced sweep
must clear a tag and the stale read must fault, or the temporal half of the reading is void).

| scenario | malloc | realloc | reads old[i] | why this shape |
|---|---:|---:|---:|---|
| shrink-small-same-class | 256 | 250 | 255 | jemalloc keeps the size class, both layers in place; the byte is past the new size |
| shrink-small-other-class | 256 | 200 | 0 | jemalloc alone moves (class 224); the shim reclaims 56 of 256 and keeps the pointer |
| grow-small-within-class | 100 | 112 | 0 | grows into the slack of the 112 class |
| shrink-large-in-place | 1 MiB | 600 KiB | last byte of the old block | jemalloc trims the extent in place; the shim keeps the whole block |
| grow-moved-no-sweep | 64 | 64 KiB | 0 | moved; the old block is in quarantine, no sweep has run |
| grow-moved-sweep | 64 | 64 KiB | 0 | moved, then `malloc_revoke_quarantine_force_flush()` |
| shrink-moved-sweep | 256 | 64 | 0 | the shim moves (192 of 256 reclaimed), then a forced sweep |

## Reading, 2026-10-27, one boot of the stock CheriBSD riscv64-purecap image on the local host

Predictions were written before the run (in the session, not committed): with revocation on,
every same-address scenario completes with the old tag still set, the two sweep scenarios die of
SIGPROT, the moved-no-sweep one completes; with revocation off everything completes, including
the shrink to another size class, which jemalloc alone moves. All held.

Controls from the same boot:

    CHERI_ABI pointer_bytes=16 runtime_revocation=1
    CHERI_BOUNDARY_READY                                        (exit 162, SIGPROT)
    REVOCATION_CONTROL revocation=1 tag_after_sweep=0 reissued=1 (then SIGPROT at the read)

Revocation ON (the default, shim + jemalloc):

    SCEN shrink-small-same-class  same_addr=1 old_len=256     new_len=256     old_tag=1  -> ACCESS completed
    SCEN shrink-small-other-class same_addr=1 old_len=256     new_len=256     old_tag=1  -> ACCESS completed
    SCEN grow-small-within-class  same_addr=1 old_len=112     new_len=112     old_tag=1  -> ACCESS completed
    SCEN shrink-large-in-place    same_addr=1 old_len=1048576 new_len=1048576 old_tag=1  -> ACCESS completed
    SCEN grow-moved-no-sweep      same_addr=0 old_len=64      new_len=65536   old_tag=1  -> ACCESS completed
    SCEN grow-moved-sweep         same_addr=0 old_len=64      new_len=65536   old_tag=0  -> SIGPROT
    SCEN shrink-moved-sweep       same_addr=0 old_len=256     new_len=64      old_tag=0  -> SIGPROT

Revocation OFF (`_RUNTIME_REVOCATION_DISABLE=1`, jemalloc alone):

    SCEN shrink-small-same-class  same_addr=1 old_len=256     new_len=256     old_tag=1  -> ACCESS completed
    SCEN shrink-small-other-class same_addr=0 old_len=256     new_len=224     old_tag=1  -> ACCESS completed
    SCEN grow-small-within-class  same_addr=1 old_len=112     new_len=112     old_tag=1  -> ACCESS completed
    SCEN shrink-large-in-place    same_addr=1 old_len=1048576 new_len=655360  old_tag=1  -> ACCESS completed
    SCEN grow-moved-no-sweep      same_addr=0 old_len=64      new_len=65536   old_tag=1  -> ACCESS completed
    SCEN grow-moved-sweep         same_addr=0 old_len=64      new_len=65536   old_tag=1  -> ACCESS completed
    SCEN shrink-moved-sweep       same_addr=0 old_len=256     new_len=64      old_tag=1  -> ACCESS completed

Every read returned the fill byte 0x5a.

## What it says

- **Same address: the old capability is stale by C17 and fully usable on CheriBSD**, under both
  layers. It keeps its tag and its old bounds. In `shrink-small-same-class` the read is at byte 255
  of an object whose size is now 250: inside the size class, so nothing can detect it.
- **The shim never shrinks the underlying block in place.** `shrink-large-in-place` with
  revocation on returns the same capability with `new_len` still 1 MiB. Jemalloc alone trims the
  extent to 640 KiB and the old capability still reaches the trimmed tail (`new_len=655360`,
  read at 1048575 completed): the one same-address shape where the stale capability names
  memory the allocator has taken back.
- **The shim and jemalloc disagree on `shrink-small-other-class`.** 256 to 200 stays in place
  under the shim (56 bytes reclaimed, under half) and moves under jemalloc alone (size class 224).
- **Moved: stale and usable until the sweep.** With no sweep between the realloc and the access
  the old capability keeps its tag and the read completes; the block is in quarantine, so it reads
  the old fill, not another object. After a forced sweep the tag is gone and the read faults. With
  revocation off nothing ever revokes.

## Second boot, 2026-10-27: the two-owners shape and the trimmed tail

Three scenarios added to `probe.c` (the first seven reproduced with identical RESULT lines):

- `two-owners-in-place` and `two-owners-moved`: the FFmpeg diracdec shape (upstream fix
  8a4ea9644833): the realloc's result is kept in a second field, the first keeps the old
  pointer and is still written through. The child stores 0xa5 through the OLD capability at a
  labelled `csb`, then reads the same offset through the NEW one.
- `shrink-tail-reissued`: 1 MiB shrunk to 600 KiB, then up to 16 fresh 200 KiB allocations; if
  one lands inside the old capability's range, a store through the old capability at that
  address is read back through the fresh object.

Predictions before the run: in-place two-owners completes and the write is seen through the new
pointer under both layers; moved two-owners completes and is not seen with revocation on; the tail
is reissued only with revocation off. All held, with the same controls as the first boot.

    revocation=1 two-owners-in-place   same_addr=1 old_len=256 new_len=256       stale_write_seen_through_new=1
    revocation=1 two-owners-moved      same_addr=0 old_len=64  new_len=65536     stale_write_seen_through_new=0
    revocation=1 shrink-tail-reissued  same_addr=1 old_len=1048576 new_len=1048576  no fresh allocation inside the old range
    revocation=0 two-owners-in-place   same_addr=1 old_len=256 new_len=256       stale_write_seen_through_new=1
    revocation=0 two-owners-moved      same_addr=0 old_len=64  new_len=65536     stale_write_seen_through_new=0
    revocation=0 shrink-tail-reissued  same_addr=1 old_len=1048576 new_len=655360   fresh=old+655360, stale_write_seen_through_fresh=1

- **Two owners at the same address: silent corruption of the live object on CheriBSD**, under
  both layers. The write through the stale field landed in the slice-parameter array, and
  nothing faulted. Moved, the write went into the dead block and the live object was untouched.
- **The trimmed tail is reissued under jemalloc alone.** With revocation off the first fresh
  200 KiB allocation sat exactly at the old object's new end, and a store through the old
  capability was read back through that fresh object: a write into a different live object
  through a pointer the program got from malloc. With the shim on, the block was never shrunk
  and no fresh allocation could land inside it.

## The Capstone twin, 2026-10-27, virtual profile on p13

`probe-capstone.c` is the same question for our virtual profile (musl mallocng behind
`runtime/virtual/heap.c`, whose `publish()` revokes the slot's child and issues a fresh one on
every successful realloc, in place or not). One launch per scenario, since the VM runs one domain
per launch; `gate-capstone.sh` is the guest script and `run-capstone-p13.sh` the host driver as
run. Two controls in the same boot: `fixed` reads through the NEW capability after a moving
realloc and must complete; `freed` reads through a freed capability and must fault.

Kit: the flat v7 gate platform (`~/v6gate/platform-sdev`: QEMU b290f565 v7 build, images-v7,
gates-1018 adapter) with the dev control SDK's runtime (`~/heap-v6/sdk-control`, dev tree of
2026-10-20, which carries the 2026-10-13 lifetime rotation; its heap.c:322 has the publish()
comment) and headers from a surviving musl 1.2.5 tree, because the SDK's own header path was
deleted with the Juliet kit on 2026-10-22. Against dev of 2026-10-27 the heap's realloc path
differs only by 4f054964a7d8 (lifetime records asked for in batches), which does not touch the
rotation. Predictions before the run: `fixed` completes; `freed` and all five scenarios fault at
the label. All held.

    fixed                    old=..058050[..058050,..058090) new=..007080[..007080,..017080)                 ACCESS completed value=0x5a
    freed                    old=..511050[..511050,..511090)                                                 revoked-capability fault at the label
    shrink-small-same-class  same_addr=1 old=[..615060,..615160) new=[..615060,..61515a) index=255            revoked-capability fault at the label
    shrink-small-other-class same_addr=0 old=[..9c0060,..9c0160) new=[..9c0430,..9c04f8)                      revoked-capability fault at the label
    grow-small-within-class  same_addr=0 old=[..049040,..0490a4) new=[..049420,..049490)                      revoked-capability fault at the label
    shrink-large-in-place    same_addr=0 old=[..2fe800,..3fe800) new=[..267400,..2fd400) index=1048575        revoked-capability fault at the label
    grow-moved               same_addr=0 old=[..13d050,..13d090) new=[..05f080,..06f080)                      revoked-capability fault at the label

Each fault is QEMU's `[CAPSTONE] Cap mem access on revoked capability` line. In all six, pc minus
the base of the domain's code region (the `CAPSTONE_VM_ADD_OK slot=0 ... reg=3` line of that
launch) is 0x10470, which is `realloc_probe_read` (symbol 0x20470 at image base 0x10000), and the
cursor equals the old capability's cursor plus the probed index. The launcher's own
`domain fault cause=25` record did not appear in the captured output of this adapter build; the
QEMU line is the same event as the control kit's malloc gate pairs with cause 25.

- **Same address, the case CheriBSD cannot see:** `shrink-small-same-class` kept the address and
  the old capability still faulted. The reading the malloc gate already had for 100 to 101
  (`inplace-stale`) holds for a shrink that keeps the size class too.
- **Under our musl, four of the five shapes moved** where CheriBSD's layers stayed in place: the
  wrapper rounds the request to a representable size and mallocng's in-place condition then
  fails for 256 to 200, 100 to 112, and the large shrink (which remaps). Moved or not, the old
  capability is revoked at the realloc call, so the distinction does not change the verdict.
- **CheriBSD's exact bounds are size-class bounds, ours are request bounds:** 256 to 250 gives a
  new capability of length 250 here and 256 there.

Second boot on p13, same kit, with the two-owners shape added (`realloc_probe_write`, symbol
0x2074c; the seven earlier scenarios reproduced):

    two-owners-in-place  same_addr=1 old=[..0f6060,..0f6160) new=[..0f6060,..0f615a)   revoked-capability fault at the STORE label
    two-owners-moved     same_addr=0 old=[..9cd050,..9cd090) new=[..8ef080,..8ff080)   revoked-capability fault at the STORE label

So the diracdec shape at the same address, which CheriBSD executes as a silent write into the
live object, faults here at the stale field's first store, before any free.

## Two real upstream bugs of this shape, reconstructed, 2026-10-27

This folder is also a `case-json` corpus (`corpus.json`): these two are tracked as cases
`00_eb398971b_sort_bang_stale_base_after_shrink` and `01_4547997058_hvaux_interior_after_hsplit`,
each a standalone `case.c` reduction; `probe.c` / `probe-capstone.c` remain the runnable instruments.

The synthetic scenarios above are the mechanism in the abstract. These two are reconstructions of
specific upstream fixes whose defect is exactly a stale access across a same-address `realloc`.
Both fixes predate our pins, so neither is a live release-differential case: `mruby eb398971b` is
in the `4.0.0-rc2` pin and `Perl 4547997058` (2014) is long inside `5.36.3`. The only way to
measure the defect is to reconstruct its `realloc` shape, which is what the two scenarios below do.
They are not a reduction of the interpreter; they are the allocator-level shape the interpreter
bug reduces to, with sizes chosen so the `realloc` keeps its address on the layer named.

- `mruby-sort-bang-shrink` (READ, 512 -> 256 B, read index 504): mruby `Array#sort!`
  (fix `eb398971b`, #6649). `sort_cmp` captured the backing-store base and the length; a
  comparison block that shrinks the array makes `ary_shrink_capa()` `realloc` the store smaller,
  and the old guard was a pointer-equality test, so a same-address shrink was missed and the sort
  read the stale tail. 512 -> 256 B hits the shim's shrink shortcut (reclaims exactly half).
- `perl-hsplit-aux-grow` (TWO_OWNERS write, 200 -> 224 B, interior write at 192): Perl
  `Gv_AMupdate` stale `HvAUX` (fix `4547997058`, RT #121362). `aux = HvAUX(stash)` is an interior
  pointer into the tail of the bucket array; adding a method triggers `S_hsplit`, whose `Renew()`
  grows the array, after which `aux->xhv_aux_flags |= ...` writes through the stale interior
  pointer into live buckets. `malloc(200)` already returns a 224 B block, so `realloc(.., 224)` is
  `size <= old_size` at the capability level and keeps the address under both layers.

CheriBSD, third boot of the stock image, same controls as above (ABI `pointer_bytes=16`,
bounds exit 162, `REVOCATION_CONTROL ... tag_after_sweep=0 reissued=1`); the earlier ten scenarios
reproduced:

    revocation=1 mruby-sort-bang-shrink  same_addr=1 old_len=512 new_len=512 old_tag=1  -> ACCESS completed value=0x5a
    revocation=1 perl-hsplit-aux-grow    same_addr=1 old_len=224 new_len=224 old_tag=1  -> stale_write_seen_through_new=1 value=0xa5
    revocation=0 mruby-sort-bang-shrink  same_addr=0 old_len=512 new_len=256 old_tag=1  -> ACCESS completed value=0x5a
    revocation=0 perl-hsplit-aux-grow    same_addr=1 old_len=224 new_len=224 old_tag=1  -> stale_write_seen_through_new=1 value=0xa5

- **mruby:** revocation on, the shim keeps the address with the full 512 B bounds, so the stale
  tail read at 504 completes; revocation off, jemalloc moves (512 and 256 are different size
  classes) but the old capability still reads its old block. Silent under both.
- **Perl:** both layers keep the address, the old interior capability keeps its tag, and the
  stale write at 192 lands in the live grown object (read back through the new pointer as 0xa5).
  A prediction that the shim would move every grow was wrong: it measures the old size by the
  capability length, 224, so 200 -> 224 fits `size <= old_size` and takes the shortcut. Silent
  live-object corruption on CheriBSD under its default.

The Capstone twin on p13, same kit as above, with both scenarios added (this build's labels are
`realloc_probe_read` 0x20730 and `realloc_probe_write` 0x2073c; `fixed` completed and `freed`
faulted):

    mruby-sort-bang-shrink  same_addr=0 old=[..ffa050,..ffa250) new=[..ffa850,..ffa950) index=504  revoked-capability fault at the READ label
    perl-hsplit-aux-grow    same_addr=1 old=[..c41040,..c41108) new=[..c41040,..c41120) index=192  revoked-capability fault at the STORE label

Each is QEMU's `[CAPSTONE] Cap mem access on revoked capability` line; pc minus the launch's code
base is 0x10730 (read) and 0x1073c (store), the two labels, and the fault cursor equals the old
capability's cursor plus the probed index, with the fault reporting the OLD child's bounds
(length 512 for mruby, 200 for Perl). The Perl case is the sharpest: the `realloc` kept its
address (`same_addr=1`, old and new share the base, the grow stayed in place under our musl too),
CheriBSD wrote silently into the live object through the old interior pointer, and here the old
child is revoked at the `realloc` call, so the same store faults.

## Instances in the wild, 2026-10-27 hunt

Upstream fixes whose defect does NOT need the realloc to move the block, found by two
commit-message hunts over full-history clones (FFmpeg, Wireshark, memcached, httpd, APR; PHP,
Perl, CPython, PostgreSQL, SQLite, mruby). Pickaxe searches were only partial, so fixes with
uninformative messages may be missing. Each hash was read with `git show`; the two marked
(re-read) were re-read by the session that wrote this section.

Stale POINTER used after a same-address realloc (the shape this probe measures; CheriBSD
silent, Capstone faults at the first stale access):

- mruby eb398971b (2025-10-27, issue #6649, `src/array.c`, re-read): Array#sort! captured
  `a = RARRAY_PTR(ary)` and `n`; a sort block that shrinks the array reallocs its capacity down
  in place. The guard was `RARRAY_PTR(ary) != p`, a pointer-equality test, so the defect lived
  precisely in the same-address branch. The fix adds `|| RARRAY_LEN(ary) != n`. Reconstructed and
  measured above as `mruby-sort-bang-shrink`.
- Perl 4547997058 (2014-03-04, RT #121362, `gv.c`, re-read): `aux = HvAUX(stash)` taken before
  a method lookup that can split the hash; `S_hsplit` does `Renew` on the bucket array that
  carries the aux struct at its tail and moves the struct to the new tail. After an in-place
  Renew the old aux slot holds live bucket pointers, and `aux->xhv_aux_flags |= ...` writes
  into them. The fix re-reads `HvAUX(stash)`; the companion audit is 339441efdf. Reconstructed and
  measured above as `perl-hsplit-aux-grow`.
- FFmpeg 8a4ea9644833 (`libavcodec/diracdec.c`, corpus case 08): two owners, measured above.
- CPython buffer exports over shrunk storage: #4583 (2008, `arraymodule.c`, a memoryview keeps
  the old base and length over an array resized under it) and gh-159739 (2026-10-22,
  `_io/bytesio.c`, truncate during a reentrant `__index__` while an export is live). For
  buffers above pymalloc's threshold the shrink is a libc realloc.

Stale SIZE with the new pointer after a shrinking realloc (caught by exact bounds, not by
lifetimes; CheriBSD's size-class bounds hide it when the overrun stays inside the class):

- mruby d24ff0e81 (2026-07-23, #6964, `mruby-io/src/io.c`): ungetc shrinks the buffer, the
  memmove reads start+len bytes of the old extent.
- Perl e081490516 (2019, RT #133871, `regcomp.c`): a branch reset shrinks the paren arrays,
  the restored larger count indexes past them.
- PHP b053192a03e6 (2021, bug #81429, `spl_fixedarray.c`): a destructor's setSize reallocs the
  element array down, the write at the stale index lands in the trimmed tail.
- CPython gh-91153, gh-153570, gh-153578 (`bytearrayobject.c`): reentrant `__index__` or
  `__buffer__` shrinks the bytearray under an index or slice bound taken before.
- SQLite e2db2b1978 (2009, experimental `ext/csv/csv.c`): maxRow kept the old size after a
  shrinking realloc; the next fgets wrote into the trimmed tail.

Two owners whose only stale use is a free (both platforms catch at the free):

- FFmpeg 274064a5 (2025, OSS-Fuzz 393523547, `aacdec_usac.c`): a whole-struct copy duplicates
  the owner of a realloc'd block; decode_close frees it twice.
- FFmpeg 00a63bfb (2013, `matroskaenc.c`) and SQLite 5c959118f1 / b6063cf823: the realloc
  wrapper frees on failure or on an oversize request and the owning field keeps the pointer.

Not separable by an allocator (stale pointer into a block the program's own allocator keeps
live): memcached 7af02b0c (read-buffer cache), APR 8f8f9994 (pool hash expand), mruby
1c57532b2 (copy-on-write detach), PostgreSQL ef5f559b95 and ff84efe4fd (palloc chunks,
simplehash compaction). Realloc-to-zero double frees (FFmpeg 3c6607eb, 67e285ce) do not arise
on either platform, which allocate a one-byte object for size zero.
