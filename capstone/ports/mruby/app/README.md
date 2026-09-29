# mruby in a Capstone musl domain

mruby, pinned, built as a Capstone domain on musl with this repo's port runtime,
and its own test suite (`mrbtest`) run in the domain under QEMU against the same
configuration built natively. The goal is a real allocator-heavy interpreter in a
domain, as the ground on which the Sublet corpus rows for mruby can later run on
the real program rather than on extracted reproducers.

## Result (2026-09-26)

mrbtest, `MRB_NO_BOXING`, switch dispatch, `-O2`, gems: stdlib, stdlib-ext,
math, metaprog, mruby-io, mruby-errno, mruby-dir, mruby-pack
(`results/2026-09-26/mrbtest.txt` has the lines):

| | Total | OK | KO | Crash | Skip |
|---|---:|---:|---:|---:|---:|
| native (x86-64, gcc) | 2710 | 2617 | 0 | 0 | 70 |
| Capstone domain (QEMU) | 2710 | 2604 | 0 | 0 | 83 |

The 13 extra skips in the domain are the whole difference, and none of them is a
missing piece of the port: 12 need a child process (`IO.popen`, backticks,
`FileTest.pipe?` on a popen pipe, `IO#close_write` on a popen stream), which a
domain cannot start (`MRB_NO_IO_POPEN`), and 1 needs a UNIX socket
(`FileTest.socket?`), which a domain does not have (patch 0007). The exit report
lists `socket` (198, three calls) as the only unserved syscall.

The same holds in the two other arms, word boxing (`MRBD_BOXING=word`, patch
0006) and direct-threaded dispatch (`MRBD_DISPATCH=direct`): 2604 OK, KO 0,
Crash 0, the same 83 skips.

Scripts: the `mruby` interpreter runs `scripts/driver.rb` in the domain, which
runs `scripts/smoke.rb`'s 13 checks (arrays, hashes, symbols, strings, GC under
load, recursion, exceptions, bignums, file and directory IO, pack/format) and
then four of mruby's `benchmark/` scripts read from their files and evaluated
there (`bm_mandel_term`, `bm_so_lists`, `bm_hash_access`, `bm_so_mandelbrot`).
The output is byte-identical to native (`results/2026-09-26/scripts.txt`).

## Pins (`MRBD_PIN`)

- `head` (default): mruby of 2026-09-17, the results above. Every defect known
  when it was taken is fixed in it, save one fixed the next day (0cf969a2b).
- `4.0.0-rc2` (9d523e2f74f2, 2026-03-12): the pin for the Sublet evaluation.
  An inventory of mruby's fixed temporal defects (CVEs, OSS-Fuzz, issues and
  PRs) against every release tag put 24 at this tag whose memory mruby's own
  allocators manage, 11 of them in reused GC object slots, which ASan cannot
  see; more than at any final release (18 at 4.0.0 and at 3.3.0). Each defect is to
  be measured against this tag plus that one defect's upstream fix.
  mrbtest in the domain: OK 1632, KO 0, Crash 0 (native OK 1639); the 7 extra
  skips are popen and sockets (`results/2026-09-26/mrbtest-4.0.0-rc2.txt`).
  This version still parses with parse.y, so it needs no Prism patches; its
  0001, 0003 and 0007 are rewritten for its code, 0002 is head's. There is no
  0006 for it yet: `MRBD_BOXING=word` stops with a message.

## The Sublet heap (`MRBD_HEAP=sublet`)

mruby's `mrb_malloc` sits on the domain's `malloc`. `MRBD_HEAP=sublet` links
the runtime's `sublet_heap.c` in place of level0: a buddy heap over a region the
host grants, one bounded alias per block, every free revoked. Every body mruby
allocates -- strings, arrays, hashes, the VM stack, ireps -- and its GC heap
pages come from it; single GC object slots do not (a page is revoked only when
the GC frees it whole). mruby itself is unchanged.

4.0.0-rc2 on it (`results/2026-09-26/mrbtest-4.0.0-rc2-sublet-heap.txt`):
mrbtest OK 1632, KO 0, Crash 0, the same skips as on level0. The run spends
259,253 revocation nodes (split + mrev), four times silicon's 65,532 per boot;
under QEMU's default pool, which is that budget, it stopped after 307 passing
tests, and it completes with `CAPSTONE_REV_NODES=16777216`.

## Every GC object slot under Sublet (`MRBD_HEAP=sublet-gc`)

The third arm (4.0.0-rc2, patch 0008, `MRB_CAPSTONE_GC_SUBLET`; the design is
`docs/design/mruby-gc-sublet-port-plan.md`): the GC carves its pages from a
second grant it holds linearly, issues each object its own 80-byte alias
(`sublet_take`) and revokes the slot when the sweep frees the object
(`sublet_give`), so a stale reference to a collected object dies with it rather
than reaching the slot's next occupant. No free slot is read: the page header
and the free list are a sidecar. Without the define `gc.c` preprocesses
identically to before.

mrbtest (`results/2026-09-26/mrbtest-4.0.0-rc2-sublet-gc.txt`): OK 1632, KO 0,
Crash 0, the same skips as on level0; the GC carved 65 pages, issued 109,608
slots and revoked 82,122. Its first run faulted in `obj_free`, on a dying fiber's
frame env that the same sweep had already revoked -- a GC check that reads a
possibly dead object, which the design names and this call site had missed; it
now goes through the GC's own alias. Deviations: an all-dead page is kept, and
`mrb_gc_add_region` is refused. Runs need `MRBD_GC_REGION_BYTES` (the second
grant) and a large `CAPSTONE_REV_NODES`.

## Build and run

    CAPSTONE_LLVM_BUILD_DIR=<llvm build> RUNTIME_REPO=<llvm-capstone tree> \
      MRBD_TESTS=1 bash build-mruby-domain.sh
    bash run-mruby-domain.sh $MRBD_ROOT/src/mruby/build/capstone/bin/mrbtest <work> 2400 -- -v

`build-mruby-domain.sh` builds musl and the port runtime privately, clones mruby
at `ad98f216eb47` (and its Prism submodule at `c0e37816e97e`), applies
`patches/`, and runs rake with `build_config.rb`: a native build (the reference,
and the host `mrbc`) and a `capstone` cross build whose compiler and linker are
`toolchain/capstone-cc`. `MRUBY_MIRROR=<local clone>` avoids the network.
`run-mruby-domain.sh` stages the image and its arguments on the share and runs it
under the libc-test host helper; `toolchain/domain_entry.c` turns the domain's
entry into `main(argc, argv)` from `/mnt/host/dom-args` and `/mnt/host/dom-env`.

Knobs (`build_config.rb`): `MRBD_BOXING=no|word`, `MRBD_DISPATCH=switch|direct`,
`MRBD_OPT`, `MRBD_TESTS`, `MRBD_DEFINES`.

## What mruby needed

### Patches (`patches/<pin>/`, each with its reason in its header; the table is head's)

| | File | Why |
|---|---|---|
| 0001 | `include/mruby/string.h` | The embedded length needs 6 bits at 16-byte pointers (`RSTRING_EMBED_LEN_MAX` is 59); coderange and encoding move up with it. Widening the length alone overlapped the coderange (four string tests failed, `"#{"A"*32}:"` gave `":"`). |
| 0002 | `src/symbol.c` | The literal flag in a name pointer's low bit went through `uintptr_t` (8 bytes): untagged name, fault before any Ruby ran. Now `__uintcap_t`. |
| 0003 | `src/gc.c` | A GC region's base aligned through `uintptr_t`. Now `__uintcap_t`. |
| 0004 | `mruby-compiler/src/ccontext.c` | The Prism arena answered 8 bytes off a 16-byte boundary; misaligned capability store. |
| 0005 | `prism/src/util/pm_constant_pool.c` | The constants array followed 8-byte buckets unaligned. |
| 0006 | `include/mruby/boxing_word.h` | `MRB_WORD_BOXING` only: the boxed word becomes `__uintcap_t`. |
| 0007 | mruby-io's tests | The test setup raised when `socket()` failed; it now skips the one socket test. |

The two `__uintcap_t` patches (and 0006) need the compiler's `__intcap` type.

### Compiler (llvm-capstone, stacked branches)

- `compiler/intcap*`: `__intcap`/`__uintcap_t`, and an `__intcap` operand in
  pointer arithmetic and subscripts using its address.
- `compiler/cap-init-blockaddress`: static tables of label addresses (mruby's
  direct-threaded dispatch, `MRBD_DISPATCH=direct`) are initialised at run time
  like other pointers in static data.
- `compiler/sroa-keep-capability-whole`: SROA split `mrb_method_t` (`{ i32
  flags, union { proc pointer, function pointer } }`) so that the pointer landed
  12 bytes into an `align 4` alloca and was copied as words; the method's proc
  pointer arrived untagged and `mrb_vm_exec` faulted at `-O2` (localised by object
  bisection to `class.o`, then `-opt-bisect-limit` to SROA on
  `mrb_define_method_raw`).

### Port runtime (`runtime/hostcall-more-files`, `runtime/hostcall-mruby-io`)

128 open files instead of 8; and for mruby-io: `symlink`, `lstat`
(`AT_SYMLINK_NOFOLLOW`), `chmod`, `flock` through the helper, `dup`/`dup3`/
`F_DUPFD` sharing one file position, `FD_CLOEXEC` kept per descriptor, and
`/dev/tty` refused with ENXIO (a domain has no controlling terminal); and pipes
that hold 64 KiB, as Linux's do (4.0.0-rc2's mrbtest writes 4097 bytes into one
before reading).
`tests/runtime-qemu/fd-path-ops/` tests these with a control.

### Configuration

`capstone64-unknown-elf` defines neither `__unix__` nor `__linux__`, from which
mruby infers the platform: the capstone build sets `MRB_WITH_IO_PREAD_PWRITE`
(else `IO#pread`/`#pwrite` are left out) and `MRB_STR_LENGTH_MAX=0` (else strings
are capped at 1 MiB where the native build has no cap). `POOL_ALIGNMENT=16` in
both builds: the parser pool's cells hold pointers.

## Prior art

- `ports/musl-capstone/mruby-probe/` on `c128/3-musl` (2026-08-20, e.g.
  `80e0bf1ae874`, `53b732469525`): mruby built under the gp-captable ABI with
  LTO, to find where that ABI stopped. This port is the dev-ABI successor.
- The CheriBSD purecap mruby port (`xlang/cheri/mruby-port`) widened the embedded
  length as 0001's first version did, for an mruby without a coderange field.
