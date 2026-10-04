# Probe case 39: the capability arm for FFmpeg pool-repros case 3, measured (2026-10-04)

**Question.** `bug-corpora/ffmpeg/pool-repros/03_5c66a3ab51_vvc_nonref_output_releases_tabs` was the
**only one of the 22 upstream cases** across memcached, tshark and FFmpeg with no measured capability
arm: its `spatial` and `sublet` arms read `"status": "not written"`. Does it discriminate?

**Pre-registration.** `PREREG.md` in this folder, pushed as **`a9a5df563658`** on
`lane/ffmpeg-case3-probe39` **before** the domain build and before any boot.

## Verdict

**Both cells exactly as registered.** `run.sh` exited 0; the runner raises on the first non-passing row,
so a silent pass was not available.

| mode | meaning | expected | result |
|---:|---|---|---|
| **0** | bounds only, no lease | completed | **completed** — `passed: true`, `runner_exit: 0` |
| **2** | a Sublet lease on every pool get, revoked when the entry returns | fault | **fault, cause 24**, `pc = 0x101a11720`, `expected_pc = 0x101a11720` |

**One binary for both cells.** `domain_sha256 = ac4473aa27353f9e…` in *both* rows — the mode is a
runtime argument, so the pair differs in **exactly one thing**. `SHA256SUMS` records the domain image
and the Linux-side loader every boot ran.

**The fault is attributed, not merely present.** `pc == expected_pc`, where `expected_pc` is read from
the domain's **own** printed capability sites after the stage marker, so it is the address the running
program reported for `ff2_probe_read` rather than a symbol address I derived. `site = 0` is correct
here: the runner computes `writing = case in (2, 4, 6, 38)`, and 39 is a read. Exactly one fault, and
`FF2 return=42044` absent.

## This is NOT the `sublet-port` arm — the distinction matters

Case 3 declares two protected Capstone arms and they are different things:

- **`spatial` / `sublet`** — the **component** port, `ports/ffmpeg/buffer-pool`, probe slot 39. **This
  bundle.** Measured.
- **`sublet-port`** (fixtures 46/47) — the **app** port, `ports/ffmpeg/app`, FFmpeg's own pools under
  the Sublet allocator. **Still a prediction**, and blocked: the app SDK's ABI gate
  (`capstone/runtime/application/CMakeLists.txt:16`) refuses every toolchain on this host for want of
  the intcap extensions (PR #120 / C-72). Do not read this bundle as closing that arm.

Slot 39 is exactly what case 3's own note said was missing: *"spatial and sublet need a probe case in
`ports/ffmpeg/buffer-pool` (36-38 are taken by cases 0-2)"*. With it, cases 0-3 all carry the same
measured pair, 36-39.

## Why the probe needed care — the LIFO trap

Releasing **two** entries is what distinguishes this probe from tests 5/6/7, which release one. The
free list is LIFO — `pool_return_entry` pushes onto `pool->available_entries` and
`refstruct_pool_get_ext` pops the head — and `ff_vvc_unref_frame` releases `tab_dmvr_mvf` first and
`rpl_tab` second, so the entry handed back is **`rpl_tab`**. Watching only `tab_dmvr_mvf` reads "no
reuse", which **a correct allocator and a defect that does not exist produce identically**; it cost a
void native run on 2026-10-03. So the probe asserts the order (CHECK 485) *and* picks `held` by
comparison, and both tables get a distinct known fill (`0xA0`, `0xB0`) so a stale read of zero cannot be
confused with uninitialised memory.

**VVC's pools are the simple all-NULL-callback form** (`vvc/dec.c:387,394`), and the probe uses
`av_refstruct_pool_alloc(64, 0)` to match, so the recycling here is the same code path the decoder's is.

## The image was verified to contain case 39 before the boots

Checked in the disassembly, not assumed from the build, and the check was proven able to report:

- all seven new CHECK codes `480`-`486` present as `0x1e0`-`0x1e6`, once each;
- **positive control** — test 38's codes `470`/`471`/`473` (`0x1d6`/`0x1d7`/`0x1d9`) also present;
- **negative control** — a bogus code (`0x270f`) absent;
- the case-39 dispatch immediate `0x27` present.

The first attempt at this check grepped for the codes in **decimal** and returned 0 for the new codes
*and for the control*, which is why the control is in the list: it showed the instrument was broken
rather than case 39 missing.

## Toolchain, and one limit taken from it

- **compiler:** `clang 22.0.0git`, project-starch/llvm-capstone **`b7b31421e9fa`** — recorded as a
  commit identity, since a path is not an identity.
- **Behavioural note, stated rather than hidden:** this clang **fails** the app SDK's direct-call check
  — the gate's two-call probe shows the target computed once and reused with no `delin`, the C-46
  shape. The component port does not invoke that gate, and the probe's own direct calls are not what is
  under test, but the image was built by a compiler with that shape and the result should be read with
  that known.
- emulator: `CAPSTONE_QEMU_BINARY` from the pinned `capstone-qemu` build; `CAPSTONE_REV_NODES=1048576`;
  `rounds=70000`.
- The `linux-guest` preset had to be configured with `-DCAPSTONE_BUILDROOT_DIR` pointed at the main
  clone: this ran from a worktree, and **a worktree has no submodules**, so the toolchain file resolved
  the buildroot cross-gcc into an empty directory and configure failed until it was overridden.

## What this does and does not say

- **It does** give case 3 a measured, discriminating capability pair: the unprotected arm completes the
  stale read and the Sublet arm faults at the labelled access.
- **It does not** make case 3 live at the pin. `5c66a3ab51` is an ancestor of `n9.0.1`, so the case
  re-introduces the reverse of the fix, as all four in this corpus do.
- **It does not** close `sublet-port`, PoisonCap, CheriBSD, `backing` or `native-detect` for this case.
- **It does not** say anything about silicon. Cause 24 here is capstone-qemu reloading a revoked
  capability untagged; the deployed silicon lets a stale data access retire (ISSUES Q-11), so no trap is
  expected there.
- **N = 1 per cell**, two boots.

Files: `PREREG.md` (the predictions, pushed first), `result-lines.txt`, `SHA256SUMS`.
