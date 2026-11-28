# B5b-2 on silicon: pre-registered 2026-11-24, before any board run

**The bitstream:** B5b-2's full build, capstone-ariane b5b2-step6 0c0458149, sealed by the board lane:
`caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-f2-hpmtag-lrscfix-tagoverlap6c-pagedtbl-r76-r75d-b5b2_0c0458149.bit`,
sha256 `b44291bacb220b8ae6dd02ff113cb68bcfff231893fae70741261c087bb53b9a`. Its full-build keys and readings are in
llvm-capstone capstone/docs/plans/2026-11-24-b5b2-trap-route.md.

**The session:** one board, flashed with that seal, its name read back after the power cycle. Then the 16 images in
the table's order, the control first. The runner power-cycles before each image, and every image is expected to
RETURN (print SR and its vector, then spin), so the ordering rule is met.

**Each vector** is the recorder's words at `board_rec`: the count SR at board_rec, then the values at
board_rec + 8·(1..SR), as the UART prints them (`SR`, then one `SV` line per value).
- Each was taken by simulating the image itself on 0c0458149's RTL through the Anvil prologue: `rtl_dirty=no`,
  `S12_MEM_DELAY=12`, `+time_out=3000000`, no XDEF (s2pre-wt `.sup/b5k2`).
- rc=2 is the image's CAP_PASS spin running to the cycle limit after the report.
- `expected-vectors.json` carries the same table for graders.

**Checks the simulation passed before this was written:**
- Every image's vector equals the in-tree run of the same test on the same core, with one explained exception (below).
- Every image wrote exactly 59 + 21·SR characters to the UART: the prologue's banner, the SR line, one SV line per value
  and the END line. The prologue's divisor write lands on the same address, hence 59. On the first pass (before
  RVTEST_PASS was mapped to the report) the same check read 25 for every B5b-2 image, so it fires.
- The plain recorder (`out/xcheck/b5b2-illegal-plain`, simulation only) reads b5b2-illegal's 73 values.
- No B5b-2 image loads memory that is neither part of the image nor stored earlier in its own run, so none reads
  uninitialised DRAM. The same scan flags sublet-life's pool dump at 0xbff00000, its positive control. Limit: the
  lifetime unit's own memory writes are not in the core trace; sublet-life's exclusion below rests on source.

**The exception, b5b2-raw position 16 (reading 15, M4):** it compares the CSTRAW'd word 0 of a source whose lane was
destroyed with a plain ld of the tagged source.
- Under Sublet a tagged word 0 is `bounds[25:0] << 38 | address[37:0]` (ariane_pkg.sv:966-967). CLDRAW carries that raw
  word, and a plain ld projects only the cursor.
- So the reading is 1 only when bounds[25:0] is 0. With the tree's link (data at 0x80004000) it is 1. In this image
  data sits at 0x80005000: CLDRAW reads 0x0004000080005000, bounds[12] = 1, and the reading is 0.
- It is a property of the layout, not of the hardware; the image's own 0 is the prediction.

## Identities and vectors (on 0c0458149, graded)

| image | sha256 | board_rec | SR | retired (sim) |
|---|---|---|---|---|
| `sublet-life.bin` | `991f87887e3f40ba15b5a5fa6a8ef22701617e8cd3f0f1e3db382dd139a316d5` | 0x80003100 | 35 | 1476795 |
| `b5b2-illegal.bin` | `d76a286aa87a65aa4035feeb148b353f82bef75489f8a018678ac76514479e91` | 0x80005000 | 73 | 1466572 |
| `b5b2-illegal-runv.bin` | `1e6591f0f0b5a8ae540c8b4f3c5f51b319fd8776e598977f69e5e40bce8281bc` | 0x80003000 | 4 | 1482450 |
| `b5b2-uprot.bin` | `08285ab00036d92f52de8756364e7256f72fac4ed340b2f2b2288471cd6a0a22` | 0x8000c000 | 19 | 1372974 |
| `b5b2-uprot-mis.bin` | `b51889b8266d654db26747ee0f65347f5eb52e781fdc484a73f1be081d57cdf0` | 0x8000c000 | 23 | 1371967 |
| `b5b2-rule2.bin` | `748635029c20760ff9e72eb51f40704ef2ef8678f35ed20294fed2ba19d7a074` | 0x80010000 | 14 | 1322137 |
| `b5b2-rule2-pair.bin` | `408e7d85b84dcb3c35a1cf014ab736487858ed038ddef0acaf2e7d20bdf60f9a` | 0x80010000 | 21 | 1320254 |
| `b5b2-sref.bin` | `624a531562bab3a4ac7d2cecca532f037a5c947c748842926e6d574990a4a10f` | 0x8000c040 | 20 | 1370221 |
| `b5b2-sref-gen.bin` | `6fb462b500969cd1a23b5c7f6ae716ec78a1ec1bef503a4a699983cadcea8be7` | 0x8000c040 | 20 | 1370050 |
| `b5b2-scratch.bin` | `12adb230ffbdc8e1a8955f0a10fec32eb982c0403611f368985f46168325d82e` | 0x8000c240 | 54 | 1368514 |
| `b5b2-mwrite.bin` | `17975ed92c56eb21c79f99c457b9963317390a1dad83ea908a9f193100b679eb` | 0x8000c240 | 8 | 1374423 |
| `b5b2-raw.bin` | `50ed437fdb47d25b626335101cb319b5309836033d6ca223065b846c85fa0643` | 0x8000c240 | 38 | 1370227 |
| `b5b2-rawlcc.bin` | `e4764971b93d9e0bb4799a89901975dfe10a77e95541513cb243293f3940b878` | 0x8000c240 | 5 | 1374151 |
| `b5b2-cptag.bin` | `8e74b5918604fd8103a1fee378f57b12ded8e9af5ae970c0355b4c7251cd4b29` | 0x8000f240 | 24 | 1331141 |
| `b5b2-cptag2.bin` | `ecb129125c952ce3a66e12b53748a3e2a278885da61c30c649efa73adc9d637f` | 0x8000e240 | 13 | 1343336 |
| `b5b2-nsedge.bin` | `b75706e200fb76e2a308b6b683af6f254bcba136c4fa735aa41244d3c7a319c6` | 0x80011280 | 17 | 1247630 |

**sublet-life** (SR 35):
```
0x0 0x0 0x0 0x19 0x0 0x0 0x19 0x0 0x0 0x0 0x19 0x19
0x2 0x2 0x8000000000010000 0x5 0x300000002 0x7 0xc00000000 0x100000000 0x1000000000 0x400000000 0x0 0x200000000
0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x6
```
**b5b2-illegal** (SR 73):
```
0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1
0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1
0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1
0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1
0x2 0x1 0x1 0x2 0x1 0x1 0xff 0x2 0x0 0xff 0x2 0x0
0xff 0x2 0x0 0xff 0x2 0x0 0xff 0x2 0x0 0xff 0x2 0x0
0x12
```
**b5b2-illegal-runv** (SR 4):
```
0xff 0x2 0x0 0x0
```
**b5b2-uprot** (SR 19):
```
0x0 0x77 0x19 0x18 0x1c 0x19 0x0 0x99 0x0 0x77 0x0 0x77
0x19 0x0 0x77 0x1c 0x0 0x77 0x6
```
**b5b2-uprot-mis** (SR 23):
```
0x0 0x77 0x19 0x18 0x1c 0x19 0x0 0x99 0x18 0x1c 0x4 0x19
0x0 0x77 0x0 0x77 0x19 0x0 0x77 0x1c 0x0 0x77 0xa
```
**b5b2-rule2** (SR 14):
```
0xd 0xf 0x0 0x77 0x0 0x77 0xd 0x0 0x77 0x0 0x77 0x0
0x77 0x3
```
**b5b2-rule2-pair** (SR 21):
```
0xd 0xf 0x0 0x77 0x0 0x77 0xd 0x0 0x77 0x0 0x77 0x1
0xd 0x0 0x0 0x77 0x1 0xf 0x0 0x77 0x5
```
**b5b2-sref** (SR 20):
```
0x0 0x0 0x1 0x1 0x0 0x0 0x0 0x1 0x4 0x3 0x1c 0x6
0x6 0x0 0x1 0x0 0x2 0x18 0x0 0x3
```
**b5b2-sref-gen** (SR 20):
```
0x0 0x0 0x1 0x1 0x0 0x0 0x0 0x1 0x4 0x3 0x1c 0x6000000080000006
0x6 0xc0000001 0x1 0x0 0x2 0x18 0x0 0x3
```
**b5b2-scratch** (SR 54):
```
0x35 0x1 0x0 0x1 0x1 0x7 0x1 0x1 0x1 0x1 0x1c 0x1
0x7 0x1 0x7 0x1 0x7 0x1 0x7 0x1 0x7 0x7 0x7 0x1
0x7 0x1 0x1 0x7 0x1 0x2 0x1 0x1 0x1 0x1 0x1 0x1
0x1 0x2 0x1 0x1 0x1 0x7 0x7 0x7 0x1 0x1 0x1 0x1
0x1 0x2 0x0 0x1 0x1c 0x6
```
**b5b2-mwrite** (SR 8):
```
0x7 0x1 0x19 0x18 0x1 0x0 0x77 0x2
```
**b5b2-raw** (SR 38):
```
0x25 0x0 0x7 0x0 0x1 0x1 0x1 0x0 0x1 0x7 0x1 0x1
0x0 0x7 0x1 0x0 0x7 0x1 0x7 0x18 0x1 0x18 0x4 0x6
0x4 0x0 0x7 0x1 0x1 0x0 0x1c 0x0 0x1 0x2 0x2 0x18
0x3 0x9
```
**b5b2-rawlcc** (SR 5):
```
0x4 0x1 0x1 0x1 0x1
```
**b5b2-cptag** (SR 24):
```
0x17 0x0 0x0 0x0 0x1 0x1 0x0 0x1 0x0 0x5 0x1 0x5
0x1 0x5 0x1 0x5 0x1 0x0 0x0 0x2 0x1 0x1 0x2 0x6
```
**b5b2-cptag2** (SR 13):
```
0xc 0x1 0x1 0x7 0x0 0x0 0x1 0x1 0x0 0x1 0x1 0x1
0x0
```
**b5b2-nsedge** (SR 17):
```
0x10 0x0 0x19 0x0 0x0 0x0 0x19 0x0 0x19 0x0 0x19 0x0
0x0 0x19 0x19 0x0 0x6
```

## The grading rule

- **Before each image:** its sha256 and board_rec must equal the table. The resident name read back must be the seal's.
- **The control:** sublet-life must read SR 35 with positions 1-24 and 35 exact.
  - **Positions 25-34 are not graded.** They are pool slots 5-9, above the header's next, which nothing in the boot
    writes. Simulation zeroes them and DRAM does not: csmint-bare-2026-11-23's amendment, boot 1450, the same test.
  - A control that misses makes the whole session VOID: no B5b-2 image is graded from it.
- **Each B5b-2 image:** AS PREDICTED when SR and every value equal the table. Any other reading is a MISS, reported
  with its positions and values, with no re-interpretation before a re-read.
- **An image that prints no SR line** is NO READING, classified by the board-run skill. It is never read as a pass.
- **No pre-B5b-2 vector is graded.** The resident name and the seal's sha256 are the identity check.
