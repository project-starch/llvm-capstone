# wmem block_fast lifetime controls on virtual Capstone, 2026-10-12

This is Wireshark 4.6.8's row of `docs/plans/2026-10-12-nested-lifetime-controls-extended.md`,
which was pre-registered at `5d336c73931c`. The controls are
[`../../controls/lifetime-block-fast/`](../../controls/lifetime-block-fast/README.md). They run on
wmem's `block_fast` allocator (`WMEM_ALLOCATOR_BLOCK_FAST`), and the release operation is
`wmem_free_all`, as the pre-registration fixes.

* **protected**: wmem with patch 0001 (`WM_SUBLET=ON`).
* **release**: wmem as released (`WM_SUBLET=OFF`). This is the pre-registration's "stock" build.

Both arms are built from the same snapshot. Per arm there were three boots, alternating release
and protected, with three programs per boot. `results.json` holds every boot's record.

## Result

Every program matched its prediction in every boot, 18 of 18. Control 2 (sibling) is **not
applicable** on block_fast and was not run. It is not counted as a pass.

| control | arm | outcome (3 boots each) | A | C / parent refs | access probes |
|---|---|---|---|---|---|
| 1 parent (`00`) | protected | completed ×3 | `0x..fe0a0`, bounds [A, A+64) | block `0x..fe050`, bounds = the 2 MiB block; `life` and `generation` the same bounds | `lifetime.h:124` load / `:132` store. Via block from `case.c:40,46,49,52,53`, via the new generation from `case.c:61,64`, via life from `case.c:73,76`. Read back through block at `case.c:65,77` |
| 1 parent (`00`) | release | completed ×3 | `0x..fe080`, bounds = the block | block `0x..fe050` | as above, via block only (no `generation` or `life` without the patch) |
| 2 sibling | both | **not applicable** | | | |
| 3 successor (`01`) | protected | C == A ×3 (first try), completed ×3 | `0x..fe0a0` | C = A, bounds [C, C+64) | `lifetime.h:124/132` from `blockfast.h:149,150` |
| 3 successor (`01`) | release | C == A ×3 (first try), completed ×3 | `0x..fe080` | C = A | same |
| 3 stale through A (`02`) | protected | **fault ×3: cause 25 at `wm_defect_probe`** (link pc `0x21e40`, `wm_probe`+20, `shared/driver.c:53`), address = A | `0x..fe0a0` | C = A | C's accesses done; stale read from `case.c:23` not done |
| 3 stale through A (`02`) | release | completed ×3; all 64 bytes read through A are C's | `0x..fe080` | C = A | stale read `shared/driver.c:53` from `case.c:23`, done |

`0x..` abbreviates the high bits. Each boot maps the block at a different address, and
`results.json` has the full values for every boot. A is the block's first chunk: block header
plus one 16-byte chunk header (64 + 16 = `0x50` on protected, where the header carries patch 0001's
two lifetimes; 32 + 16 = `0x30` on release). The protected per-boot faults were:

| boot | runtime pc | code start | fault address = A = C |
|---|---|---|---|
| protected-1 | `0x3f91c91e40` | `0x3f91c80000` | `0x3f910fe0a0` |
| protected-2 | `0x3fb3991e40` | `0x3fb3980000` | `0x3fb2dfe0a0` |
| protected-3 | `0x3f82691e40` | `0x3f82680000` | `0x3f81afe0a0` |

Each runtime pc, minus the code start, plus the image's link base, equals the address of the
`wm_defect_probe` label in the image's own symbol table (`0x21e40`). The launcher's line for
protected-1 was:

    capstone-exec: domain fault cause=25 pc=0x3f91c91e40 address=0x3f910fe0a0 entry=0x3f91c80890 code=0x3f91c80000-0x3f91d00000 last=0x42 preparing=0x42 sha256=5fb4e364f37fb90ef788c1d0b859dff122f9c8ea457800e1b3c5c98b715595f1 image=/mnt/vm/02-successor-stale.dom

It came after `WM_CTL access read stale-A via=A addr=0x3f910fe0a0 len=64 probe=wm_defect_probe
site=case.c:23` and `WM_DEFECT case=2 ready`, with no `done` line.

### Verdicts against the pre-registration

* **Control 1, parent (protected): AS PREDICTED.** After `wmem_free_all`, the block reference read
  back what the allocator left in A's former range:
  * the block header's `pos` was rewound to the header size (`block-pos pos=64 ok`);
  * A's chunk header still held A's length (`A-chunk-header len=64 ok`);
  * all 64 bytes still held A's pattern (`A-former-range-after-reset bytes=0..63 mismatches=0 ok`),
    because block_fast's reset writes nothing into chunks.

  Writes and read-backs then completed through three references. The first is the block pointer
  on `block_list`, which is mallocng's capability for the 2 MiB block and the one the
  pre-registration names. The second is patch 0001's `generation` after the reset, the new
  generation the next object is derived from. The third is `life`, the parent of the revoked
  generation and the reset's CREVOKE source.
* **Control 2, sibling: NOT APPLICABLE.** block_fast's `wmem_free` is a no-op on both builds, and
  `wmem_free_all` ends every object at once. No release leaves a sibling live, so nothing was run.
* **Control 3, successor (protected): AS PREDICTED.** After the reset, wmem returned A's address
  on the first request, and the access through C completed. Before the reset, a second object
  had moved `pos` past A, so the equal address comes from the reset.
* **Stale access through A (protected): AS PREDICTED.** It was rejected with cause 25 at the
  labelled probe.
* **Release build: AS PREDICTED.** All controls completed, and the stale read through A
  completed and returned C's bytes (`stale-A-reads-C bytes=0..63 mismatches=0 ok`). The protected
  rejection is therefore the patch's.

## Inputs

* **llvm-capstone revision.** HEAD `5d336c73931cd87a3c50262648c88ac9688409d3`
  (`ports/nested-controls-extended`) plus the uncommitted files of `controls/lifetime-block-fast/`
  listed with their sha256 in `results.json` → `revision`. No tracked file was modified. The
  snapshot was `git archive HEAD capstone compiler-rt/lib/builtins` with those files added, and an
  empty `llvm/`, which the environment script requires as the fork-root marker. The two README
  files were written after the run and are not inputs. The programs include the 2026-10-11
  controls' `controls/lifetime/lifetime.h` unchanged (HEAD's copy) for the labelled accesses,
  patterns and build check, so its probe lines are the same as last night's.
* **Platform.** These are the kit's files, hashed by the runner at each boot. They are the same
  as on 2026-10-11:

  | component | sha256 prefix |
  |---|---|
  | QEMU | `af37cc32` |
  | launcher `capstone-vexec` | `3975314a` |
  | module `capstone_vm.ko` | `a1c6cb6b` |
  | kernel | `e833dfb1` |
  | firmware | `44cd6cbc` |
  | rootfs | `62049800` |

  The machine was `virt-capstone`, `-smp 1`, with `x-capstone-u-mode=true` and
  `x-capstone-exact-bounds=true` (`run-staged.py`, sha256 `d6770b8b`).
* **Build.**
  * SDK: `capstone/runtime/virtual/build-sdk.sh` over musl 1.2.5 with the port's patches, from this
    snapshot. The compiler is clang at sha256 `4c6135fe`, the same compiler as on 2026-10-11.
  * Port: the wmem port's `capstone-application` preset (Release), with
    `-DWM_CORPUS_DIR=controls/lifetime-block-fast` and `WM_SUBLET` ON or OFF.
  * Patch 0001 sha256 is `5f2ef5f3`.
* **Image sha256, per arm.** Each image was identical across its three boots.

  | program | protected | release |
  |---|---|---|
  | 00-parent-block | `9b3500c7617d` | `707122339426` |
  | 01-successor-reuse | `b1aef8e8eb56` | `3d73f90165c2` |
  | 02-successor-stale | `5fb4e364f37f` | `b18d74194ba4` |

* **Runner.** `controls/lifetime-block-fast/run-lifetime-controls.py`, sha256 `74370721`. The gate
  script hash was the same for all six boots.
* **Wall time.** Each boot took 7.1 to 7.8 s from boot to the last program, and 13 to 14 s
  including staging. The SDK build took 19 s, and each port build took 5 s.

## The judge can fail

The judge was tested offline on altered copies of these records. It returns DIFFERS for each of
the following:

* the protected stale record judged under the release rule, and the reverse;
* the stale fault with cause 24;
* the stale fault off the label;
* C at an address other than A's (successor protected, stale protected, stale release);
* control 1 with:
  * the `life` check missing;
  * A's range not what the allocator left;
  * `pos` not rewound;
  * a fault;
* a C access with no `done` line;
* a release stale read that does not return C's bytes.

A missing result returns ERROR. Run against the protected build with `--arm release`, the runner
refuses before booting. The programs also refuse a wmem whose bounds do not match the build they
claim (`lifetime.h`, `wm_ctl_arm`).

## Deviations and limits

* There were three boots per arm, where the pre-registration asks for at least two.
* The controls' new code is a header (`blockfast.h`) and three programs. The labelled accesses
  and the build check are the 2026-10-11 header's, included unchanged.
* The printed bounds cannot tell the generation before the reset from the one after it: both
  span the whole block. The evidence that the reset revoked A's generation is the stale fault,
  cause 25, and not the printed references.
* The writes in control 1 overwrite A's former range after the reset. The program allocates
  nothing afterwards.
* Jumbo objects (larger than the block) and blocks other than the first, which the reset frees
  to the system allocator rather than renewing, are not exercised. Each program uses one block.
* The runs used the snapshot's `run-staged.py` and the platform kit's images on the remote
  host, as the 2026-10-11 controls did. They did not use a persistent `capstone_vm` VM.
