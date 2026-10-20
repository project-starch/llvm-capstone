# Q-15 placement repair on the board pass's line (nomovc-cw): pre-registration, pushed before any run (2026-10-20)

Supersedes the pb13 re-run of q15-placement 57215acb2dbb, which is NOT scored: that pre-registration named defect-00
6ab1d1d0148d and defect-08 47c236c38400, and the images its run read were rewritten 34 s after it (5e50e709, 593563a3).

## Images (all built before this file, from committed clean trees; run roots under /mnt/hdd/<user>/capstone/)

- pb13 (memcached allocator-repros, virtual-nested-pools): t3-images-cw/q15/prebuilt/b13, tree q15-placement-cw
  d893ec627057 (vp2-tier12 851265d49299 + 57215acb2dbb's memcached 0006, postgres 0003, driver-virtual.c, and
  capstone/representable.h 7c41a409acb7a897), SDK t3-freeze-nb-nomovc-cw (sdk.json 64bc9086452fc30a).
  slabs_q15_note is in exactly the images that link do_cache_alloc (defect-00, -01, -08), as in t3-new8b.
    a644ba44ecc3161a  cases/bin/defect-00
    aac6bf0c4264b7f6  cases/bin/defect-01
    1f180c4cf6f352b4  cases/bin/defect-02
    6ccf86b8cb0d16e9  cases/bin/defect-03
    86c2490a2322441b  cases/bin/defect-04
    3ba945435bfc4e99  cases/bin/defect-05
    f40871226889c621  cases/bin/defect-06
    3ca8fbcfd77c9b62  cases/bin/defect-07
    8b3ab66b3cae57ca  cases/bin/defect-08
    e8a8e55b58795385  controls/bin/defect-90
- pgm1 (postgres mmgr-repros, virtual-pg-pools): t3-images-cw/q15/pg/mmgr-{cases,controls}-ON, same tree and SDK.
    51e4f1a9414b7ec3  mmgr-cases-ON/bin/00-tuplestore-double-free
    7a4677bd87ebda14  mmgr-cases-ON/bin/00-tuplestore-double-free-control
    2d04951fa8a0975a  mmgr-cases-ON/bin/01-vacuum-dead-items-stale-struct
    9f87acd1843c5a7d  mmgr-cases-ON/bin/01-vacuum-dead-items-stale-struct-control
    604d1a6cf20c81ea  mmgr-cases-ON/bin/02-live-parts-stale-alias
    8c3d473bd5832984  mmgr-cases-ON/bin/02-live-parts-stale-alias-control
    0aa07545fbefafeb  mmgr-cases-ON/bin/03-pgoutput-entry-cxt-teardown
    2dde99e0c78a1c96  mmgr-cases-ON/bin/03-pgoutput-entry-cxt-teardown-control
    65896b08f6b40f46  mmgr-cases-ON/bin/04-reorderbuffer-spec-insert-lsn
    73b314b754d3956f  mmgr-cases-ON/bin/04-reorderbuffer-spec-insert-lsn-control
    d892f764f5208d17  mmgr-cases-ON/bin/allocator-example
    59ed98d9c6516fd1  mmgr-cases-ON/bin/client-allocset-capstone-application
    c43598271ddc2e39  mmgr-cases-ON/bin/client-bump-capstone-application
    165bf913a12aad0f  mmgr-cases-ON/bin/client-generation-capstone-application
    157aea137b3a91d9  mmgr-cases-ON/bin/client-slab-capstone-application
    230b77c5934607fe  mmgr-cases-ON/bin/contexts-native
    464c1c642655940b  mmgr-cases-ON/bin/replay
    0a3c8acdbe83c38b  mmgr-controls-ON/bin/00-uaf-chunk
    2233f8956918957a  mmgr-controls-ON/bin/allocator-example
    6eb764f9c45cd151  mmgr-controls-ON/bin/client-allocset-capstone-application
    6369194abeec799f  mmgr-controls-ON/bin/client-bump-capstone-application
    6874f12faf1c4374  mmgr-controls-ON/bin/client-generation-capstone-application
    663239d10f13e161  mmgr-controls-ON/bin/client-slab-capstone-application
    e49dd9088bae704c  mmgr-controls-ON/bin/contexts-native
    c4aac0f58d240c3d  mmgr-controls-ON/bin/replay
- pg-pools (postgres sql-repros lt1): t3-images-cw/q15/postgres-lt1, tree runtime-5b3e-cw-q15 a62c3fd755c9
  (runtime 8f132305 = 5b3e6b79 + 232d13b2's heap.c + the MOVC switch, built OFF; + 0003), d9787ac2's recipe.
    fdfb66b57baff0ac  postgres-lt1/image/postgres.dom

## Step 1: exact bounds (qkit7-eb), against the exact-bounds records

Runners from wt-q15cw, invocations as run-q15-eb.sh (pb13: run-virtual-cases.py --prebuilt; pgm1: mmgr run-defects.py;
sql: sql-repros run-arm.py --image, fixture pgfixture/cluster). Comparator: per cell verdict, cause, symbol, controls,
notes, infra.

| arm | record | prediction |
|---|---|---|
| pb13 | t3-eb-pb13 | 8 of 9 identical; case 08 (ascii_all_spaces_scan_past_rbuf) reads NO-READING: control-failed, pad-masked (first_oob_offset 0 < pad_bytes 16; the read buffer 16400 -> 16416 at its 32-byte granule): CONTROL-FAILED 785, the all-spaces scan stopping in the tail pad, the mechanism the case's oracle records for silicon |
| pgm1 | t3-pgm1 | 5 of 5 identical |
| pg-pools | t3-sql-eb-virtual-pg-pools | 9 of 9 identical |

Masked rows: 1 (pb13/08, as above, counted as no reading with the pad as its named cause; not MISSED).
Any other difference is reported with its console line and blocks step 2. These images are built without
+movc-keeps-integer-source and the records with it, so a difference elsewhere is attributed (flag vs repair) before
anything else is concluded.

## Step 2: compact, on the current-bitstream kit qkit7-tB-mimic (and plain qkit7-nf2 beside it)

The same images and invocations. Predictions: every cell identical to its step-1 reading, on both kits; 0 lines
matching `cause=29` in each run's FILE raw/guest/serial.log (count printed with the file's line count; the counter's
control is t3-q15-c-pb13's serial, which reads 4). Q15_PAD lines are reported per workload.
