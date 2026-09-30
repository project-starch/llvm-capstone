# mruby in a Capstone musl domain

mruby, pinned, built as a Capstone domain on musl with this repo's port runtime,
and its own test suite (`mrbtest`) run in the domain under QEMU against the same
configuration built natively. The goal is a real allocator-heavy interpreter in a
domain, as the ground on which the Sublet corpus rows for mruby can later run on
the real program rather than on extracted reproducers.

## Delegated result with stdlib-io (2026-09-30)

The build takes mruby's whole `stdlib-io` gembox: at head `mruby-socket`,
`mruby-env`, `mruby-signal` and `mruby-process` join `mruby-io`,
`mruby-errno` and `mruby-dir`; 4.0.0-rc2's gembox adds `mruby-socket`.
Patch 0007 is gone. mrbtest in the domain against the same configuration
natively (`results/2026-09-30/mrbtest-stdlib-io.json`):

| | Total | OK | KO | Crash | Skip |
|---|---:|---:|---:|---:|---:|
| head, native | 2858 | 2761 | 0 | 0 | 74 |
| head, domain | 2858 | 2760 | 0 | 1 | 74 |
| 4.0.0-rc2, native | 1682 | 1673 | 0 | 0 | 9 |
| 4.0.0-rc2, domain | 1682 | 1673 | 0 | 0 | 9 |

Every test but one has the status it has natively. The one is
`Process.kill passes the pid selectors on`: `Process.kill(0, 0)`, signal 0
to the caller's own process group, is refused with EPERM, because the
launcher confines `kill` to the task, its children and its parent. The
control, head with the previous configuration on the same compiler and
runtime, has 2710 tests, 2617 OK, 70 skips in both builds: `socket()` is a
delegated call now, so `FileTest.socket?` passed with 0007 still applied,
and the patch no longer fired. `mruby-process` makes no child of its own
(pid, ppid, waitpid, kill; its tests make children with `IO.popen`, patch
0009), and `mruby-signal` is a table of signal names that installs no
handler.

## Delegated result (2026-09-29)

The current ABI-v2 recipe passes the regular head suite with 2616 OK,
zero failures/crashes and 71 skips (23 optional stress cases not run).
The 4.0.0-rc2 suite has 1638 OK, zero failures/crashes and 10 skips.
Process IO accounts for the newly passing tests. The protected GC/popen
smoke passes; the larger GC stress still exhausts revocation nodes at
both 65,536 and 262,144. See the
[migration qualification](../../common/application/README.md#verified-migration-2026-09-29).

## Historical result (2026-09-26)

mrbtest, `MRB_NO_BOXING`, switch dispatch, `-O2`, gems: stdlib, stdlib-ext,
math, metaprog, mruby-io, mruby-errno, mruby-dir, mruby-pack
(`results/2026-09-26/mrbtest.txt` has the lines):

| | Total | OK | KO | Crash | Skip |
|---|---:|---:|---:|---:|---:|
| native (x86-64, gcc) | 2710 | 2617 | 0 | 0 | 70 |
| Capstone domain (QEMU) | 2710 | 2604 | 0 | 0 | 83 |

The 13 extra skips in this v1 run are the whole difference: 12 need a child
process (`IO.popen`, backticks, `FileTest.pipe?` on a popen pipe,
`IO#close_write` on a popen stream) and 1 needs a UNIX socket
(`FileTest.socket?`). The build disables process IO with `MRB_NO_IO_POPEN`;
the exit report lists `socket` (198, three calls) as the only unserved syscall.

The current recipe builds ABI v2 exclusively. Patch 0009 replaces the POSIX
IO HAL's fork path with delegated `posix_spawn`, including pipe closure and
standard-stream redirection. Process IO is enabled. The historical counts
above remain evidence for the old image, not measurements of the new recipe.
See [the shared application build and runner](../../common/application/README.md).

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
  0001 and 0003 are rewritten for its code, 0002 is head's. There is no
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
    python3 ../../common/application/run.py --state "$VM_STATE" \
      --result "$MRBD_ROOT/mrbtest.json" "$MRBD_ROOT/src/mruby/build/capstone/bin/mrbtest" -v

`build-mruby-domain.sh` builds musl and the application SDK privately, clones
mruby at `ad98f216eb47` (and Prism at `c0e37816e97e`), applies `patches/`, and
runs rake with `build_config.rb`. It produces a native reference and the
Capstone application using the SDK's compiler driver. `MRUBY_MIRROR=<local
clone>` avoids the network. Use a new root after a patch-set change.

The common runner supplies ordinary argv, environment, cwd and streams.
There are no argument side files or private entry/HostCall helpers.

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

These were services of the HostCall v0 runtime, removed on 2026-09-30; under the
delegated runtime every one of them is Linux's own call. What the port needed then:

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
