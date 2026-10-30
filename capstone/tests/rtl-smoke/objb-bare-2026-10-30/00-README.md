# objb-bare-2026-10-30

ObjBounds as bare JTAG arms. The ObjBounds bitstream is capstone-ariane objbounds-b 603528b8b: bitstream 3
(e7af48457) with CAPSTONE_NO_LIFETIME_CHECK defined and LDC/STC's node query removed from the DYN unit. Capability
loads, stores and fetches take no lifetime verdict, and everything else is bitstream 3's (`docs/plans/2026-10-30-objbounds-variant.md`). The kit does two things:

- **Grades the variant's identity by value on its first boot.**
  - `objb-stale` reads 25 at six stale accesses on bitstream 3. On ObjBounds it reads 0 there, and the stale store
    lands.
  - `objb-pcc` completes on ObjBounds only. On bitstream 3 a revoked PC capability faults in its own trap handler,
    so the image does not return; it is the LAST image of any boot.
  - `objb-pcc-ctrl`, the same stream revoking a data node instead, reads the same on both bitstreams. It is the
    boot's control.
  - `objb-stale-w` (PREREG addendum 2) is objb-stale plus one block: TIGHTEN, whose node query ObjBounds keeps, on
    each revoked node, so it reads 25 there on both bitstreams if the revocations took effect. It is a boot of its
    own, after the ObjBounds arm, and the only test here that is not in capstone-ariane (`tests/objb-stale-w.S`).
- **Reads the per-access lifetime check's cost on silicon,** shape by shape (`objb-perf-s1..s7`), on both
  bitstreams. These images print cycle counts. The pre-registration predicts the PATTERN per shape: equal, or
  higher on bitstream 3. Their simulated values are under memory delay 12 and are not silicon values.

Siblings: R-43 (the cache and replay that shapes 1 and 5 exercise), R-35 (why the cache exists), and
`sublet-bare-2026-10-15`'s `stale-access` (the same source under an older wrapper). The vector it read on silicon,
on the Sublet bitstream, equals this kit's bitstream-3 vector, which board 2 read on e7af48457 on 2026-10-30 (addendum 1).

`build.sh` rebuilds the images from `tests/` (capstone-ariane verif/tests/custom/capstone/ at 0af4307fc, unchanged at 603528b8b, checked
against `VP_SRC` when given) with `va-alloc-bare-2026-10-23`'s wrapper (inc/). `images/SHA256SUMS` pins them.
`PREREG.md` holds the vectors, simulated from THESE images on objbounds-off-b 1782d20bd (the control: e7af48457 by
preprocessing, and by the hashes of its generated Anvil units) and on 24e588199 (603528b8b's tokens), and the grading rule. Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG` = the image, `SUP_REC_ADDR` from its `.rec`),
the bare-boot path only.

`grade/` holds the two graders the record's Silicon section was graded with.
- `objb-grade.py <run dir> <bs3|objb> [<other arm's dir>]` grades one arm. With the other arm's directory it also
  grades the cost pattern.
- `objb-wgrade.py <image dir>` grades the witness against addendum 2.
- Each hashes the image its run.log names, reads the resident name and the SR/SV lines, and exits 0 only on the
  pre-registered verdict. It exits 2 on NO DATA.
- Negative controls: objb-grade.py on the bitstream-3 directory with `objb` reads STOP, and objb-wgrade.py on a
  fabricated live-node vector reads FINDING. Both exit 1.
