# wmem lifetime controls on virtual Capstone, 2026-11-11

Wireshark 4.6.8's rows of `docs/plans/2026-10-11-nested-lifetime-controls.md` (the `block`
allocator, release by `wmem_free`) and of
`docs/plans/2026-10-12-nested-lifetime-controls-extended.md` (the `block_fast` allocator, release
by `wmem_free_all`), measured again after patch 0001 and the processor model changed their
lifetime instructions. `wmem_free` ends an object with `CREVOKE(object)`. The reset renews the
block's generation with `CRENEW`. The block header's `life` field is gone, so the `block_fast`
parent control has two references left, the block and the renewed `generation`.

* **protected**: wmem with patch 0001 (`WM_SUBLET=ON`).
* **release**: wmem as released (`WM_SUBLET=OFF`).

One boot per arm and allocator (the earlier records have three). Each file here is the runner's
own `result.json` of one boot.

## Result

Every program matched its prediction, 14 of 14.

| allocator | control | protected | release |
|---|---|---|---|
| `block` | 00 parent | completed, via the block and via `generation` | completed, via the block |
| `block` | 01 sibling | completed, B intact | completed, B intact |
| `block` | 02 successor | C == A, completed | C == A, completed |
| `block` | 03 stale through A | **fault, cause 25 at `wm_defect_probe`** (link pc `0x1f0c8`), address = A | completed, reads C's bytes |
| `block_fast` | 00 parent | completed, via the block and via the renewed `generation` | completed, via the block |
| `block_fast` | 01 successor | C == A, completed | C == A, completed |
| `block_fast` | 02 stale through A | **fault, cause 25 at `wm_defect_probe`** (link pc `0x1f07c`), address = A | completed, reads C's bytes |

`block_fast` has no sibling control, as before. A is 64 bytes at offset `0x40` of its block on the
protected build, with bounds [A, A+64), and at offset `0x30` on the release build. The block header
is 48 bytes with patch 0001 and 32 without it.

## The `block_fast` runner's judge

`controls/lifetime-block-fast/run-lifetime-controls.py` still required the checks
`A-former-range-via-life` and `A-former-range-written-via-life` of control 00 on the protected
arm. The control no longer makes those accesses, because patch 0001 has no `life` reference any
more. With that judge the protected control 00 of this boot reads DIFFERS
(`failed_checks=['A-former-range-via-life', 'A-former-range-written-via-life']`) although every
access it makes completes. The two names were removed from the judge. Its other requirements are
unchanged.

## Platform and build

The kit is the tree's virtual platform on the model with the one-operand `CREVOKE` and `CRENEW`;
each file carries the kit's hashes, the image hashes and the probe label addresses. One hart, exact
bounds: `CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`, the
module loaded without parameters. The runner records `exact_bounds: true` unconditionally and does
not pass the option to `run-staged.py` itself, so the property came from that variable.
`wall_seconds` of `block-protected.json` includes 3,190 s of waiting for the QEMU lock.

    cmake --preset capstone-application -S capstone/ports/wireshark/wmem -B <out> -DCAPSTONE_SDK=<corpus SDK> \
        -DWM_CORPUS_DIR=<controls/lifetime or controls/lifetime-block-fast> -DWM_SUBLET=ON|OFF
    controls/<directory>/run-lifetime-controls.py --arm protected|release --build <out> --kit <kit> ...

These are Release builds, as the earlier records. The runner resolves the stale read by the
label's address, so the label lying at `wm_probe+0` in a Release image of this compiler does not
affect it. Patch 0001 is `bec33460c5ed`.
