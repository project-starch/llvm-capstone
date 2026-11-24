# mruby GC object slots: parent, sibling and successor controls on the one-operand lifetime instructions, 2026-11-11

The controls of [`../2026-10-11-controls/`](../2026-10-11-controls/README.md), built and run again after the lifetime
instructions changed. Patch 0008 (`MRB_CAPSTONE_SUBLET`) now hands every object out as a `CDERIVE` child of its page's
own pointer, and the sweep ends that child's lifetime with a one-operand `CREVOKE` (`capstone_cap_release`). The
intermediate `page->life` is gone, and the driver's three probes through it went with it. The driver, its build, its
runner and the judge are in [`../../controls/`](../../controls/README.md). `results.json` holds every repetition.

**All eight predictions hold, in 3 of 3 repetitions per arm.**

| control | arm | outcome | what was read |
|---|---|---|---|
| parent | protected | completed | A's slot through the page: `tt` = `MRB_TT_FREE`, `issued[i]` = NULL, the freelist link; written and read back |
| parent | release | completed | the same through the page |
| sibling | protected | completed | B intact, written and read back |
| sibling | release | completed | the same |
| successor | protected | completed | C = A after 1 allocation; C read, written and read back |
| successor | release | completed | the same |
| stale | protected | **fault, cause 25** | in `ctl_probe_stale_read_A`+140, `controls.c:176` (image VA `0x1f870`), fault address = A + 0x14 |
| stale | release | completed | the read through A returned C's data |

Each repetition recorded `pages=1 live=720 generational=1` and `FREED A after 1 full gc`, and each transcript arrived
whole. Under 0008 A's capability is bounded to its slot (its base is the slot). On release it is the GC's own pointer
(its base is the page). In the sibling control A is slot 955 and B slot 959 of 1,024, as on 2026-10-11.

## What differs from 2026-10-11

- **Patch 0008 and the driver.** With `page->life` gone, the `parent` control reads and writes A's slot through the
  page pointer only, which under the new patch is the direct parent of A's ended child. The probes `life-read-tt`,
  `life-write-tail` and `life-read-tail` no longer exist, so `stale-read-A` is at `controls.c:176`, where it was at
  line 180.
- **Lifetime instructions in the images.** Release: 6 `CDERIVE` and 3 `CREVOKE` words, the C library's mallocng.
  Protected: 8 and 5. The 2026-10-11 images read 0 / 0 and 8 / 2, on a C library that had none of its own. No image
  has a `CRENEW`.
- **The mruby images are not the 2026-10-11 ones**: other patch, SDK and compiler. They are the images of
  [`release-differential/results/2026-11-11-virtual`](../../../release-differential/results/2026-11-11-virtual/README.md).

## What ran

| arm | `MRBD_SUBLET` | mruby image | `libmruby.a` | driver `controls.dom` |
|---|---|---|---|---|
| release | 0 | `e8593fdae83a8215` | `354a678f0808aef7` | `468283d06fe8609c` |
| protected | 1 (patch 0008) | `0391d7d629e25700` | `dbfa30e06ac286a1` | `a7df2579fca5211d` |

- Tree: branch `one-operand-crevoke-crenew` at 59201bdf2, QEMU at ea57497908. Patch 0008 is 31c512ae806b....
- Builds: `ports/mruby/app/build-mruby-domain.sh` with `MRBD_PIN=4.0.0-rc2`, `MRBD_SDK` naming a copy of the corpus SDK
  (`CAPSTONE_SDK_CORPUS=1`, effective `-fno-builtin`), one copy per arm.
- Platform: QEMU 2d5a71078904..., launcher 0f8e0f903725..., module e47bca62addf... loaded without parameters (pinned,
  one context), kernel 48eb725a40e4..., firmware 5b5c047a0030..., rootfs 62049800ed0f.... Virtual profile, one hart,
  exact bounds, 2 GiB, guest CPU with `x-capstone-test-csrs=true`, which a pinned boot of this kernel needs. Full
  hashes are in `results.json`, per boot.
- Each boot is its own VM through `runtime/virtual/run-staged.py`; each mode is its own `capstone-vexec` process, in the
  order parent, sibling, successor, stale. Boot wall times: 63 to 78 s, with other boots of the same campaign on the
  host.

The protected stale faults:

| repetition | pc | fault address |
|---|---|---|
| 1 | `0x3f9440f870` | `0x3f94872c84` |
| 2 | `0x3f96c0f870` | `0x3f97366c84` |
| 3 | `0x3fb480f870` | `0x3fb4e46c84` |

Each pc is image VA `0x1f870`. N = 3 boots per arm; no run was discarded.
