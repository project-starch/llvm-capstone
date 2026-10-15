# mruby in a Capstone musl domain

mruby, pinned, built as a Capstone domain on musl with this repo's port runtime,
and its own test suite (`mrbtest`) run in the domain under QEMU against the same
configuration built natively. The goal is a real allocator-heavy interpreter in a
domain, as the ground on which the Sublet corpus rows for mruby can later run on
the real program rather than on extracted reproducers.

## mruby-task, and the GC-slot corpus (2026-09-30)

The build also takes `mruby-task`, at 4.0.0-rc2 through `hal-posix-task` and at
head through the gem itself, the port layer carrying its HAL. The gem was left
out as a service a domain does not have; what its POSIX HAL actually uses is
`sigaction(SIGALRM)` with `setitimer(ITIMER_REAL)` for the tick and
`clock_gettime` with `nanosleep` for a sleeping task. All four are delegated,
and none of them is a thread, so the scheduler runs in a domain.

That matters beyond the gem. The [GC-slot
corpus](../../../bug-corpora/mruby/gc-slot-repros) is for the defects that
reuse a GC object slot without the allocator seeing a release, and its survey
on `corpus/mruby-gc-slot-reuse` names this gem as the reason four candidates
cannot be reached: mruby #6870, #6886, #6872 and #6887, the reports carrying
that corpus's `MRB_TT_FREE` assertion exactly, all go through
`mrb_task_mark_all`, and the gem was in none of the port's gemboxes. It is in
all of them now, so those four are in range of the port.

mrbtest with the gem (`results/2026-09-30/mrbtest-task.json`):

| | Total | OK | KO | Crash | Skip |
|---|---:|---:|---:|---:|---:|
| head, native | 2936 | 2838 | 0 | 0 | 74 |
| head, domain | 2936 | 2837 | 0 | 1 | 74 |
| 4.0.0-rc2, native | 1710 | 1701 | 0 | 0 | 9 |
| 4.0.0-rc2, domain | 1710 | 1701 | 0 | 0 | 9 |

4.0.0-rc2 matches native test for test. head's one difference is the
`Process.kill(0, 0)` crash the next section describes, which the gem does not
introduce. Three task workloads also run to completion in the domain.

A reproduction of #6886 was attempted and does not yet stand up; the record
holds the arms and the control. The defect is live at the pin -- upstream
`456a8687a` adds `mrb_task_mark_all` to `final_marking_phase`, and the pin
calls it from `root_scan_phase` only -- but no Ruby workload written to that
mechanism fires, while a control with task marking compiled out does. The
corpus reached the same conclusion for its `realloc-vmstack` rows: this shape
needs a C-level case rather than a script.

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

## Patch 0010: envadjust, and what it unblocked (2026-09-30)

At the 4.0.0-rc2 pin, `stack_extend_alloc()` hands the old VM stack to
`mrb_realloc()` and then calls `envadjust()`, which moved every frame's pointer
with `ci->stack += delta` -- pointer arithmetic on a pointer `realloc` has
already freed. On an ordinary allocator that computes the right address; with
`MRBD_HEAP=sublet` or `sublet-gc`, where `free` revokes, the result is derived
from a revoked capability, carries no tag, and the next write through it faults
with cause 24. Both revoking arms died at ~40 frames of Ruby recursion, shallow
enough that `scripts/smoke.rb` stopped at M8, so neither arm could report
anything.

Upstream fixed the same defect in `e5c82761f` (2026-07-24), found when another
memory-safe C implementation trapped on that write; `patches/4.0.0-rc2/0010`
backports it. Only this pin needs it -- `head` already carries it.

With 0010 all three arms complete `scripts/smoke.rb` and `deep(500)`, and the
[release corpus](../../../bug-corpora/mruby/release-differential) runs in each:
`sublet` catches one case the control completes, and `sublet-gc` catches four,
three of which revoke-on-free cannot see.

## Every GC object slot a Sublet lifetime (`MRBD_SUBLET=1`, virtual profile)

Patch 0008 (4.0.0-rc2, `MRB_CAPSTONE_SUBLET`) makes every GC object slot a
child lifetime of its heap page. Each page keeps a lifetime over its slots
(`CDERIVE` of the page) and, per slot, the child it handed out; `mrb_obj_alloc`
returns the object as a child bounded to its slot, and the sweep revokes that
child (`CREVOKE`) once `obj_free` has freed the object, so a stale reference to
a collected object faults instead of reaching the slot's next occupant. The GC
works on its own pointers, derived from the page; where it is asked about a
pointer that may be dead (`mrb_object_dead_p`, and `obj_free`'s walk of a dying
fiber's frames) it finds the slot by address. Without the define `gc.c`
preprocesses as without the patch. 61 lines of code.

It runs on the virtual profile: build with `MRBD_SDK` naming a virtual SDK, and
`malloc` is that SDK's musl mallocng, which bounds and retires every object it
hands out -- mruby's bodies and its heap pages -- while the patch covers the
slots inside a page. mrbtest on both images, 2026-10-11: 1,710 tests, 1,700 OK,
the same one failure (`File#path`, a file-I/O test the guest answers empty),
9 skipped, no capability fault; the
[release corpus](../../../bug-corpora/mruby/release-differential) reads 19
caught without the patch and 22 with it, the three it adds being the GC-slot
rows.

Until 2026-10-11 this port also carried two physical-domain arms on the Sublet
heap (`runtime/sublet_heap.c`): `MRBD_HEAP=sublet`, every body and page on the
buddy heap, and `MRBD_HEAP=sublet-gc`, the GC's pages carved from a second
linear grant with every slot taken and given back through the region API (the
old patch 0008, design `docs/design/mruby-gc-sublet-port-plan.md`). Their
mrbtest runs and corpus readings stay in `results/2026-09-26` and the corpus's
`results/20261006`.

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
