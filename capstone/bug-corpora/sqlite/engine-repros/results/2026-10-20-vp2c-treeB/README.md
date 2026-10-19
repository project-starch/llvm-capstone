# sqlite/engine-repros on vp2 silicon, tree B (caplifive_vp2c_8de3c48cf, board 2), 2026-10-20

The board bundles of arm q15: vp2-nested-pools, memsys5 with the Q-15 repair (dev f1d522d7fff1 images, SDK t3-freeze-nb).
They are scored cell by cell against the QEMU run C, results/2026-10-20-q15-repair/run-C. The board records are in
tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/README.md (arm q15).

- `vp2-nested-pools/`: subsets a, b and c merged (cases 00-17), giving CAUGHT 12 and MISSED 6. All 18 cells equal run C.
  The subsets' own bundles sit beside it, as `vp2-nested-pools-q15{a,b,c}/`.
- Tree A's subsets d-f are under ../2026-10-20-vp2c-treeA/. Case 31's per-platform note (R-60) is there.
- Case 31 has not yet run on tree B. Its decode path is identical to tree A's by source, and a tree-B boot of subset f is
  queued.
