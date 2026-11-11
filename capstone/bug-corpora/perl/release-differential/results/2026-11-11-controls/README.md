# Perl SV heads: parent, sibling and successor controls on the one-operand lifetime instructions, 2026-11-11

The controls of [`../2026-10-11-controls/`](../2026-10-11-controls/README.md), built and run again after the lifetime
instructions changed. The SV-head patch (`ports/perl/sublet/patches/5.36.3/0001`) now derives each head from the
arena's own pointer and ends the recorded child with a one-operand `CREVOKE` (`capstone_cap_release`). The harness,
its build and its runner are in [`../../controls/`](../../controls/README.md); `results.json` is `collect.py`'s
output.

**Every prediction held on both arms.** N = 1 boot per arm (2026-10-11 had two rounds).

| mode | stock (`PERLD_SUBLET=0`) | protected (`PERLD_SUBLET=1`) | predicted |
|---|---|---|---|
| `parent` | completed, PASS; last probe `parent-read-refcnt` (`svhead-controls.c:136`) | completed, PASS | completes, both |
| `sibling` | completed, PASS; B is the head after A (A + 0x30) | completed, PASS; the same | completes, both |
| `successor` | completed, PASS; C = A | completed, PASS; C = A | completes, both |
| `stale` | completed, PASS; the read through A's old pointer returns C's head | **fault, cause 25**, in `probe_read_flags` (`svhead-controls.c:38`), after the announced probe `stale-read-flags` (`svhead-controls.c:236`) | fault on protected, completes on stock |

The arm that 2026-10-11 calls `release` is named `stock` in this record's logs and JSON.

## What ran

| | stock | protected |
|---|---|---|
| Perl image | `708d23a2ca31ff15...` | `8d744f0291c49bb6...` |
| `libperl.a` | `e41ca384ee4a29f8...` | `effd3c7160f557da...` |
| `svhead-controls.dom` | `8a6f7a7447701dd7...` | `f33ef92501017e2a...` |

- Tree: branch `one-operand-crevoke-crenew` at 59201bdf2, QEMU at ea57497908. The SV-head patch is 822a151fc302....
- The Perl builds are the corpus builds of [`../2026-11-11-virtual/`](../2026-11-11-virtual/README.md) (corpus SDK,
  effective `-fno-builtin`); `controls/build-controls.sh` links the harness against each build's `libperl.a`.
- Platform: QEMU 2d5a71078904..., launcher 0f8e0f903725..., module e47bca62addf... loaded without parameters (pinned,
  one context), kernel 48eb725a40e4..., firmware 5b5c047a0030..., rootfs 62049800ed0f.... `controls/run-controls.sh`
  boots through `runtime/virtual/run-staged.py`, one boot per arm with the four modes as four processes; the guest CPU
  adds `x-capstone-test-csrs=true`, which a pinned boot of this kernel needs.
- Boot wall times: 103 s (stock) and 98 s (protected), with other boots of the same campaign on the host.
