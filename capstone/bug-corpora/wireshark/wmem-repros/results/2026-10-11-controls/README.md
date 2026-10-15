# wmem lifetime controls on virtual Capstone, 2026-10-11

This is Wireshark 4.6.8's row of `docs/plans/2026-10-11-nested-lifetime-controls.md`, which was
pre-registered at `53325bba0d5c`. The controls are
[`../../controls/lifetime/`](../../controls/lifetime/README.md). They run on wmem's `block`
allocator, and the release operation is `wmem_free`.

* **protected**: wmem with patch 0001 (`WM_SUBLET=ON`).
* **release**: wmem as released (`WM_SUBLET=OFF`).

Both arms are built from the same snapshot. Per arm there were three boots, one boot per arm per
row of the table, and four programs per boot. `results.json` holds every boot's record.

## Result

Every program matched its prediction in every boot, 24 of 24.

| control | arm | outcome (3 boots each) | A | B / C | access probes |
|---|---|---|---|---|---|
| 00 parent | protected | completed ×3 | `0x..fe0a0`, bounds [A, A+64) | (guard `0x..fe0f0`) | `lifetime.h:124` load / `:132` store, via block from `case.c:51,73,77,78`, via generation from `case.c:86,90`, read back `case.c:91`, guard `case.c:98` |
| 00 parent | release | completed ×3 | `0x..fe080`, bounds = the block | (guard `0x..fe0d0`) | as above, via block only (no `generation` without the patch) |
| 01 sibling | protected | completed ×3, B intact (0 of 64 bytes differ) | `0x..fe0a0` | B `0x..fe0f0`, bounds [B, B+64) | `lifetime.h:124/132` from `case.c:33,35,36` |
| 01 sibling | release | completed ×3, B intact | `0x..fe080` | B `0x..fe0d0` | same |
| 02 successor | protected | C == A ×3 (first try), completed ×3 | `0x..fe0a0` | C = A, bounds [C, C+64) | `lifetime.h:124/132` from `lifetime.h:248,249` |
| 02 successor | release | C == A ×3 (first try), completed ×3 | `0x..fe080` | C = A | same |
| 03 stale through A | protected | **fault ×3: cause 25 at `wm_defect_probe`** (link pc `0x21e54`, `wm_probe`+20, `shared/driver.c:53`), address = A | `0x..fe0a0` | C = A | C's accesses done; stale read from `case.c:23` not done |
| 03 stale through A | release | completed ×3; all 64 bytes read through A are C's | `0x..fe080` | C = A | stale read `shared/driver.c:53` from `case.c:23`, done |

`0x..` abbreviates the high bits. Each boot maps the block at a different address, and
`results.json` has the full values for every boot. The protected per-boot faults were:

| boot | runtime pc | code start | fault address = A = C |
|---|---|---|---|
| protected-1 | `0x3fbbe91e54` | `0x3fbbe80000` | `0x3fbacfe0a0` |
| protected-2 | `0x3f8bf91e54` | `0x3f8bf80000` | `0x3f8adfe0a0` |
| protected-3 | `0x3f83491e54` | `0x3f83480000` | `0x3f822fe0a0` |

Each runtime pc, minus the code start, plus the image's link base, equals the address of the
`wm_defect_probe` label in the image's own symbol table (`0x21e54`).

### Verdicts against the pre-registration

* **Control 1, parent (protected): AS PREDICTED.** After `wmem_free(A)`, the block reference read
  back what the allocator left in A's former range:
  * A's chunk header was marked free with length 80;
  * both free-list links named A's own chunk;
  * bytes 32..63 were unchanged.

  Writes through both parent references completed and read back. These were the block-list
  pointer, which is mallocng's capability for the 8 MiB block, and patch 0001's `generation`,
  which is A's direct CDERIVE parent and the CREVOKE source.
* **Control 2, sibling (protected): AS PREDICTED.** Revoking A left B readable, writable and intact.
* **Control 3, successor (protected): AS PREDICTED.** wmem returned A's address on the first
  request, and the access through C completed. The new child of the same `generation` over the
  same bytes was not reached by A's revocation.
* **Stale access through A (protected): AS PREDICTED.** It was rejected with cause 25 at the
  labelled probe.
* **Release build: AS PREDICTED.** All three controls completed, and the stale read through A
  completed and returned C's bytes. The protected rejection is therefore the patch's.

## Inputs

* **llvm-capstone revision.** HEAD `53325bba0d5c9e6696a1da5c069276abfa5c9088`
  (`ports/nested-lifetime-controls`) plus the uncommitted files of `controls/lifetime/` listed with
  their sha256 in `results.json` → `revision`. No tracked file was modified. The snapshot was
  `git archive HEAD capstone compiler-rt/lib/builtins` with those files added. The two README
  files were written after the run and are not inputs.
* **Platform.** These are the kit's files, hashed by the runner at each boot:

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
  * SDK: `capstone/runtime/virtual/build-sdk.sh` over musl 1.2.5 with the port's patches. The
    compiler is clang at revision `cda7e335`, sha256 `4c6135fe`, the same compiler hash as the
    corpus's 2026-10-11 virtual bundles.
  * Port: the wmem port's `capstone-application` preset (Release), with
    `-DWM_CORPUS_DIR=controls/lifetime` and `WM_SUBLET` ON or OFF.
  * Patch 0001 sha256 is `5f2ef5f3`.
* **Image sha256, per arm.** Each image was identical across its three boots.

  | program | protected | release |
  |---|---|---|
  | 00-parent-block | `4445108a0c11` | `a67fc4447d19` |
  | 01-sibling-intact | `581c6f23d15f` | `06df3762ee44` |
  | 02-successor-reuse | `2f7b0067d53f` | `17ee3aa1fa75` |
  | 03-successor-stale | `24dc0ec4a7d9` | `bca5caf1fcf5` |

* **Runner.** `controls/lifetime/run-lifetime-controls.py`, sha256 `995aa883`. The gate script
  hash was the same for all six boots.
* **Wall time.** Each boot took 13.5 to 19.2 s from boot to the last program, and 24 to 31 s
  including staging. The SDK build took 11 s, and both port builds took 9 s each.

## The judge can fail

The judge was tested offline on altered copies of these records, and it returns DIFFERS for each
of the following:

* the protected stale record judged under the release rule, and the reverse;
* B corrupted;
* the stale fault with cause 24;
* the stale fault off the label;
* C at an address other than A's.

A missing result returns ERROR. Run against the protected build with `--arm release`, the runner
refuses before booting. The programs also refuse a wmem whose bounds do not match the build they
claim (`lifetime.h`, `wm_ctl_arm`).

## Deviations and limits

* There were three boots per arm, where the pre-registration states no N.
* The free-list links in control 00 are read with pointer loads, as the allocator reads them, and
  not at a labelled probe. The `access` line records them as `probe=pointer-load`.
* The writes in control 00 overwrite the free-list links in A's former range. The program
  allocates nothing afterwards, so the allocator never uses the corrupted links.
* `block_fast` is not covered. Its free is a no-op, so the only release it has is the reset.
* The runs used the snapshot's `run-staged.py` and the platform kit's images, as the corpus's
  2026-10-11 virtual runs did. They did not use a persistent `capstone_vm` VM.
