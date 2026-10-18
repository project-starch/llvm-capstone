# Result: a bundle's `compiler` field names the RUNNER's toolchain, not the compiler that built the images

Found while producing the first Tier 3 prediction (`mpt`) against a deliberately frozen SDK. The
freeze works; the fields anyone would check to confirm it do not say so.

## The mpt prediction, and why it is a cross-check

`t3-mpt/bundle`: 3/3 **CAUGHT**, both controls faulted, exit 0.

| image | case | verdict | cause | symbol | offset | reached | completed | attribution |
|---|---|---|---|---|---|---|---|---|
| `f76b089115a659f4` | 00 | CAUGHT | 25 | `mct_read_probe` | 56 | True | False | probe |
| `c6c3b4490abef1c9` | 01 | CAUGHT | 25 | `mct_write_probe` | 88 | True | False | probe |
| `2126edc5837f79fc` | 02 | CAUGHT | 25 | `mct_read_probe` | 56 | True | False | probe |

**All three identical to the pilot's cells** — and the pilot used a different kit *and* a different
tool checkout (mine), where this used the board lane's `qkit7` and tools at `origin/dev`. Same
image bytes, two independent tool paths, same cells. That is worth more than either run alone.

## The two platform fields, and what they actually name

| field | recorded | what it is |
|---|---|---|
| `qemu` | `5f731f99ec6a19b2` | the **wrapper script's** hash, not the binary `7a4f3a64776f0c88` it execs |
| `compiler` | `ba603a574ce08ea4` | **`--llvm-bin`'s** clang — the runner's symbolisation toolchain |

`compiler` comes from `kit_platform(kit, a.llvm_bin / "clang", HERE)`. `--llvm-bin` supplies
`v.Symbols(...)` and the freshness probe; it is **not** what compiles the cases. The compiler that
built the images is the SDK's `cc`, which `sdk.json` gives as the frozen `a5be1a4d621513eb`.
Measured: `vp2-prep/llvm-bin-v6` and `llvm-dev-current-20261017` are the same clang
(`ba603a574ce08ea4`), and the frozen tree's is `a5be1a4d621513eb`.

**So the freeze held — the images were built by the frozen compiler — and neither bundle field
records it.** A reader checking `compiler` to confirm "both sides used one frozen compiler" would be
reading the wrong toolchain, and would get a consistent answer for the wrong reason, because
`--llvm-bin` happens to be the same on both sides too.

## What does carry the proof

**The image sha256, matched cell by cell** — which is what the campaign already does. An image hash
is downstream of the compiler that produced it, so identical hashes on both sides is the real
guarantee; the platform fields are weaker than they look. For the configuration, the wrapper's
`applied.log` and (on a non-wrapper kit) `cpu.txt` carry the resolved `-cpu` string; the `qemu`
field does not.

Recorded so that no later write-up cites `compiler` as evidence the freeze held.

## A related gap in the arena pin, from the board lane

`pin_ahead` calls `pin_user_pages_remote(FOLL_WRITE)`. An **absent** page is faulted in as a write
and gets D set; a page already **present and clean** has only its `struct page` marked, not its PTE
— `gup.c:624-627`, the same lines the S-24 retraction turns on. So the pin can leave
pinned-but-clean pages, and under `svadu=false` a store there still faults; on silicon an STC there
is the S-25 ghost. That is a plausible account of why the pin did not carry `sqlite_write` in boot
5, and it is checkable on QEMU as a count of faulting capability stores inside that step under the
pin. Theirs, not mine, and recorded as a mechanism to check rather than a claim.
