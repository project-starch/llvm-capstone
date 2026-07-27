# xlang — short TODO to proceed (2026-07-27)

Phase-1 is in and merged to `capstone-bootstrap`. Nothing below needs our compiler,
our QEMU fork, or the board — all stock toolchain, so none of it can be blocked by
the churn on the silicon track.

**Priority order. 1–3 protect claims the paper will make; 4 is the one that saves us
real time later.**

1. **Rows 1 & 2 — corpus fidelity.** Both need a vendored patch to build on any
   post-2020 rustc, and row 1's patch **touches the destructor under test** (chosen to
   preserve the double-drop). Either find a patch that leaves the defect path
   untouched, or write two or three lines in `1/target.md` stating exactly what was
   changed and why the defect is still the upstream one. "Unpatched upstream source"
   is a claim we would like to keep making.

2. **Confirm CVE-2026-1979 against NVD.** The #6701 mapping came from the upstream
   commit message and could not be checked offline. Needs network.

3. **Rows 6 and 11 — write the reclassification, don't re-argue it.** The finding
   (both are spatial, not temporal) is accepted. What's needed is the paper-facing
   sentence for each: what the defect actually is, and why bounds rather than
   revocation are what stop it. Two short paragraphs in the row READMEs.

4. **Phase-2 harness skeleton — the highest-value item.** For each reproducing row,
   factor its allocate → free → use sequence into a small, toolchain-agnostic shim so
   the capability version becomes a drop-in rather than a rewrite. Keep it building
   and passing under the stock toolchain exactly as it does today. This is the grindy
   half of Phase 2 and it is fully decoupled from our in-flux compiler/ABI, so it can
   proceed in parallel with the silicon work.

**Not now:** more corpus rows, and open-ended hunting for new cross-language defects.
Corpus size is not what the paper is short of — the capability half is.

**Board:** not needed for any of the above. The board is a single shared physical
resource and is serialized across everyone working on it, so if an xlang-on-silicon
step ever becomes worthwhile, coordinate a window first rather than assuming one.

**Reporting:** keep `xlang/README.md` and the dated state note in `history/` as the
single sources of truth; per-row detail stays in `<row>/target.md`.
