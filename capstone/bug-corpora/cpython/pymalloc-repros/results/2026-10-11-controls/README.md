# CPython pymalloc: parent, sibling and successor controls, 2026-10-11

CPython 3.13.7's row of `capstone/docs/plans/2026-10-11-nested-lifetime-controls.md`. The nested
allocator is pymalloc (`PyObject_Malloc`/`PyObject_Free`). The protected arm is the interpreter
with patch 0014 (`CPY_SUBLET=1`, with the F1 fix c6b7ba55); the release arm is the same source
without 0014. The harness is `controls/nested-lifetime/` (built-in C module `_pymctl`, build
script, runner); `results.json` holds every result line.

## Result

**Every prediction held, on both arms, in each of three boots per arm.**

| control | protected (`CPY_SUBLET=1`) | release |
|---|---|---|
| 1 parent | completed: pool free-list head = A, A's link = the old head, A's last byte 0xa1, write 0x5a reads back 0x5a | completed, the same readings |
| 2 sibling | completed, B intact (0 of 48 bytes differ), write/read-back of B[47] 0x5a | completed, the same |
| 3 successor | C = A on the first allocation; 48 bytes written 0xc3 through C and read back | the same |
| stale read through A | **fault, cause 25**, pc at `pymctl_stale_read`, fault address = A | **completed**, read 0xc3, C's data |

Positive control: the stale variant completes on release and faults at the labelled probe on
protected, so the rejection is the patch's.

### Per run

Addresses are guest virtual. pcs are runtime; the image pc is `pc - code start + link base`.

| arm | boot | parent A | sibling A / B | successor A = C | stale A = C | stale outcome |
|---|---|---|---|---|---|---|
| protected | 1 | 0x3f95ba24b0 | 0x3fad7224b0 / 0x3fad723140 | 0x3f96863140 | 0x3f89b624b0 | fault cause=25 pc=0x3f87475120 (image 0x485120 = `pymctl_stale_read`) |
| protected | 2 | 0x3fbe5624b0 | 0x3f905a24b0 / 0x3f905a3140 | 0x3fa8de3140 | 0x3faf9a24b0 | fault cause=25 pc=0x3fad475120 (image 0x485120) |
| protected | 3 | 0x3fb59224b0 | 0x3fb09a24b0 / 0x3fb09a3140 | 0x3fafea3140 | 0x3f8e9624b0 | fault cause=25 pc=0x3f8c475120 (image 0x485120) |
| release | 1 | 0x3f9caa24b0 | 0x3f83d224b0 / 0x3f83d23140 | 0x3fbd9a3140 | 0x3f9b6a24b0 | completed, read 0xc3 |
| release | 2 | 0x3fa6b224b0 | 0x3f933624b0 / 0x3f93363140 | 0x3f8c3e3140 | 0x3f8ada24b0 | completed, read 0xc3 |
| release | 3 | 0x3fbb9224b0 | 0x3f8dae24b0 / 0x3f8dae3140 | 0x3fbd723140 | 0x3f80b224b0 | completed, read 0xc3 |

The parent, sibling and successor controls completed with exit status 0 in all six boots; the
stale run exited 139 on protected (the fault) and 0 on release. The
fault lines have the form `domain fault cause=25 pc=... address=<A> ...`, and the fault address
is A's every time. Each fault resolves to `pymctl_stale_probe+20`, which is exactly the
`pymctl_stale_read` label (0x485120 in the protected image). In every boot A sat at offset 9392 in
a 48-byte (szidx 2) pool of arena 0, and the pool still held 337 blocks after the free. The
successor took one allocation each time (LIFO free list). After every boot `capstone-vexec
--stats` reported 0 live domains, 0 live regions and 0 live bytes.

## The parent reference

Control 1 reaches A's former range through `allarenas[i].arena_ptr`: the capability mallocng
returned for the 1 MiB arena (patch 0009, `_Py_ARENA_PTR`). It forms `arena_ptr + offset`, applies
`POOL_ADDR`, and adds A's offset in the pool. This is the pointer `pymalloc_free` uses after its
`CREVOKE`: `arena_inside(ao, p)`, then `POOL_ADDR`. In the protected build it is the grandparent of
A's revoked child (`arena_ptr` → `arena_life` by `CDERIVE` → A by `CDERIVE`), not A itself and not
`arena_life`. The release build takes the same path. The pointer-sized read through it returns the
free-list link that `pymalloc_free` wrote into A: the pool's previous head, at 0x...3140 in each
boot.

## Probes (`controls/nested-lifetime/_pymctl.c`)

| probe | file:line | image address protected / release |
|---|---|---|
| `pymctl_read` (`lbu`) | `_pymctl.c:54` | 0x484c0c / 0x4849d8 |
| `pymctl_write` (`sb`) | `_pymctl.c:63` | 0x484c40 / 0x484a0c |
| `pymctl_stale_read` (`lbu`) | `_pymctl.c:72` | 0x485120 / 0x484eec |
| `pymctl_link_probe` (pointer load) | `_pymctl.c:82` | function at 0x484bcc / 0x484998 |

Call sites: parent 164, 170 (pool header), 173 (A's link), 174/179/182 (read), 178/180 (write);
sibling 227/233 (read), 232 (write); successor and stale 284 (write), 288 (read); stale 325 (the
stale read).

## What ran

- **Source.** llvm-capstone `53325bba0d5c` (`ports/nested-lifetime-controls`) plus three
  uncommitted files, `controls/nested-lifetime/{_pymctl.c, build-arm.sh, run-controls.py}`,
  whose sha256 values are in `results.json` under `llvm_capstone`. On the build host, a
  `git archive` of 53325bba (`capstone/`, `compiler-rt/lib/builtins`) plus those files was
  committed as a scratch snapshot. The images were built at snapshot `0668e2a9` and the runner ran
  from `e56a4689`, which differs only by a one-line runner fix (see Deviations). The local files
  hash equal to the snapshot's.
- **Images.** `build-arm.sh protected|release`, which is `CPY_TEST_CAPI=1 [CPY_SUBLET=1]
  build-virtual.sh cpython` plus `_pymctl` in `Setup.local`. The two arms built in parallel with
  `JOBS=12`, `CPY_JOBS=8`, in 273 s and 275 s. Both use compiler_sha256 4c6135feafae20c2...,
  the kit toolchain, and the tarball 6c9d8083....

  | | protected | release |
  |---|---|---|
  | `cpython.dom` sha256 | 4d8f9405cec8de3fbfa5387b47cd11956ab9c43b729e7eb1610c6d43949fb1c9 | 892a55a281d69fbf744b2b684be64f3a7a4a06f090dd2234d1b9e6de73ff0982 |
  | manifest `nested` | cpython | none |
  | patches | 0001-0005, 0007-0017 (16; 0014 sha256 6c2c170e...) | the same without 0014 (15) |
  | `profile`, `heap`, `stack_bytes` | virtual, virtual, 8388608 | same |

- **Platform.** The kit's `kit-sublet`, checked by hash before staging. QEMU af37cc32...,
  `capstone-vexec` 3975314a..., `capstone_vm.ko` a1c6cb6b..., Image e833dfb1..., fw_jump.elf
  44cd6cbc..., rootfs.ext2 62049800.... These are the files of
  `ports/cpython/app/results/2026-10-11-same-answers`. Each boot used
  `runtime/virtual/run-staged.py` with one hart and exact bounds, on its own staged disk, with its
  own QEMU lock. There were six boots in parallel, three per arm. Boots took 129-213 s, and the
  whole run took 237 s.
- **Guest command**, per control, as root with `PYTHONHOME=/mnt/vm/<arm>` and
  `CAPSTONE_EXEC_DIAGNOSTICS=1`:
  `capstone-vexec /mnt/vm/<arm>/cpython.dom -S -c "import _pymctl; _pymctl.<control>()"`. Each
  control is its own process.

## Deviations

- **A first build stopped at a check, not in the build.** Its images linked, but the post-build
  label check piped `llvm-nm` into `grep -q` under `pipefail`, and the resulting SIGPIPE failed
  the check. The check was rewritten without a pipe, and both arms were rebuilt fresh. Those first
  images got only one unreported smoke boot. That boot also exposed a runner bug: it read
  `CTL sibling completed=1 intact=1` as no completion, because it compared whole lines. The fix,
  a prefix match, is the one-line difference between snapshots 0668e2a9 and e56a4689. None of
  that smoke run is in this result.
- **The manifests say `runtime_dirty: true`.** The only untracked path in the snapshot was
  `capstone/bug-corpora/tools/__pycache__/`, which the concurrent smoke run's import of
  `verdicts.py` wrote. No source file differed.
- **Staged stdlib.** The staging copies the build's `Lib/` without `test/`, `idlelib/` and
  `tkinter/`, which the controls do not import. The same-answers staging kept them.
- **Repeats.** The plan does not fix a number of repeats. This run used three boots per arm.
- **The images are not the security table's images.** They carry the `_pymctl` module in
  addition. The protection is patch 0014 as committed at 53325bba, which includes c6b7ba55.
