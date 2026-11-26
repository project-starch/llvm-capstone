# The next B5b-2 bitstream on silicon: pre-registered 2026-11-25, before any board run

**The bitstream:** the full build on capstone-ariane b5b2-step9 904cd9c6b (8a + 8b + the CCSRRW deletion + step 9),
sealed by the board lane as `caplifive_…-r75d-b5b2n_904cd9c6b.bit`, 11,443,722 B, sha256
`36e70855a55de3692a359817b5043cc6f6f2cc92d4f41f0317df7861223a1b08` (re-hashed by the rtl lane, equal to the seal verdict;
added at the seal, before the flash); the full build's keys are in llvm-capstone capstone/docs/plans/2026-11-24-b5b2-trap-route.md. The tests are at
2bed0dfe7 (904cd9c6b + the kit re-key, tests only), so every image sim ran on the synthesised RTL.

**The session:** one board (2 or 3), the seal flashed and its name read back after the power cycle, then the 22 images
in the table's order, the control first. The runner power-cycles before each image; every image is expected to RETURN.

**Each vector** is the recorder's words at `board_rec`: SR, then SR values, printed as `SR` and one `SV` per value.
- Taken by simulating the image itself on 904cd9c6b's RTL WITHOUT the simulation-only scaffold (as silicon):
  `rtl_dirty=no`, `S12_MEM_DELAY=12`, `+time_out=3000000`. rc=2 is the report spin.
- The 14 images byte-identical across the kit's two builds keep their first run (s2pre-wt `.sup/b5kn`); the 8
  rebuilt ones come from `.sup/rk-img`.
- For every image, the bytes the simulation loaded equal the board image, by sha256.
- Every image wrote exactly 59 + 21·SR UART characters. This kit's report and recorder are plain M stores; the old
  report minted its UART capability with CAPCREATE, which step 9 makes illegal.
- No image holds a custom-3 encoding except b5b2-illegal's six deletion arms, the build's own check and its control.

## Identities and vectors (graded)

| image | sha256 | board_rec | SR | image-sim run (s2pre-wt .sup/) |
|---|---|---|---|---|
| `sublet-life.bin` | `ecbfe9171c0dbbbf9e32db48b6eb4c0c2f2b99eae0b4529c5feb46660e7227a9` | 0x80003200 | 35 | b5kn |
| `b5b2-illegal.bin` | `41c88fb5f347aab4b121ec66a22e8579f2edf7d7dad0518d4f4118a5bd55fa8d` | 0x80005000 | 76 | b5kn |
| `b5b2-illegal-runv.bin` | `e8136c724fbd7a9cf70e63c28dd4ea0c3a83116cd6fce65b72bfe3ed09f6fdb2` | 0x80003000 | 4 | b5kn |
| `b5b2-illegal-csup.bin` | `648c758ac5e8c2ba29e69ae190749fccf10a3e5de6d70af58874afdb1475d070` | 0x80004000 | 46 | b5kn |
| `b5b2-uprot.bin` | `d4734d0fa2fa859e04fae164ff9326a3e4c14a164081f6447cb782e2ea2771c5` | 0x8000d000 | 19 | b5kn |
| `b5b2-uprot-mis.bin` | `28bb4f2bea4a0f80fe7ae0817ff1026394ffdc064d9b9e80b0c0df475cc0149f` | 0x8000d000 | 23 | b5kn |
| `b5b2-rule2.bin` | `2fc07cf51430030489e1727de2e2ae5be65942f8fec73531a6693121157fcc08` | 0x80011000 | 14 | b5kn |
| `b5b2-rule2-pair.bin` | `71c3cf7f91addd1c193d6d2bcb6791624ec1f10ed1b1ba57657a5c18b8695981` | 0x80011000 | 21 | b5kn |
| `b5b2-sref.bin` | `cc6b5db7c544779758cb888464186bfa9454e05e8ab62450919df22c496f8c97` | 0x8000e040 | 20 | rk-img |
| `b5b2-sref-gen.bin` | `6d449998da2ac55330bc2f5fd94d2c70d068f0c245dd6f6ab111a0536a83cb2d` | 0x8000e040 | 20 | rk-img |
| `b5b2-scratch.bin` | `974c067471dfcb84bc3dfcca8cdf2d0057d945d17b6db8d34f569a60ad68db64` | 0x8000d240 | 54 | b5kn |
| `b5b2-mwrite.bin` | `c3649488c6fa4a71a0bd1c12b1449442c9807c69682fcd2272170de1a36862d5` | 0x8000d240 | 8 | b5kn |
| `b5b2-raw.bin` | `6a54102b7dd3d13a5f9a85962366c320514403cd39487d589af8dc50b505ccfb` | 0x8000d240 | 38 | b5kn |
| `b5b2-rawlcc.bin` | `3b6dd196ab5d701be8b8b786a5d8ae77a5f04f912ff5c8ebb40e827e1e86c7de` | 0x8000d240 | 5 | b5kn |
| `b5b2-cptag.bin` | `8d65c0e54c888ab1574567f57c0cd4ba0c172e97881536602f458d2bbc4237ea` | 0x80010240 | 24 | b5kn |
| `b5b2-cptag2.bin` | `1a9ebadff577a347a445fdaad2c60be0b196bbb1f367b46863483a7c1061f498` | 0x8000f240 | 13 | b5kn |
| `b5b2-nsedge.bin` | `58f34a2d73763cc5ee8240e5c251f4a5af30da05561a9ba33e51be2dfdc766fe` | 0x80012280 | 17 | rk-img |
| `b5b2-ctxport-f4.bin` | `181314f837d650ae5ee30f8bcf98c903f89a59238df10789d5a874460bda64e8` | 0x80016000 | 6 | rk-img |
| `b5b2-ctxport-f4ctl.bin` | `516551fc534ef04e9bb9334d6fae4278ab9e1b0a6e4e5843707b9ec2aa76b81f` | 0x80016000 | 6 | rk-img |
| `b5b2-ctxport-widelo.bin` | `a63b405dc6afe16dd8a9464918707ccdb9fe51a509e5f1ebf5ee2eed9a8cfde7` | 0x80016000 | 6 | rk-img |
| `b5b2-ctxport-widehi.bin` | `6cc824e6251f24358fe14c2643b84989da0ed40858a0b4a5299a9544e0ec87bf` | 0x80016000 | 6 | rk-img |
| `b5b2-ctxport-edgem.bin` | `eda7f42aa4b036d5df2a95ff934ce0a5635671b5183f80dedf772798cf0b2221` | 0x80016000 | 6 | rk-img |

**sublet-life** (SR 35):
```
0x0 0x0 0x0 0x19 0x0 0x0 0x19 0x0 0x0 0x0 0x19 0x19
0x2 0x2 0x8000000000010000 0x6 0x300000003 0x8 0x0 0x0 0x1000000000 0x100000000 0x1400000000 0x400000000
0x0 0x200000000 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x6
```
**b5b2-illegal** (SR 76):
```
0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1
0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1
0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1
0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1
0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1
0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1
0x2 0x1 0x1 0x19
```
**b5b2-illegal-runv** (SR 4):
```
0x2 0x1 0x1 0x1
```
**b5b2-illegal-csup** (SR 46):
```
0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1
0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1
0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1 0x2 0x1 0x1
0x2 0x1 0x1 0x2 0x1 0x1 0xff 0x2 0x0 0xe
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
0x0 0x0 0x1 0x1 0x0 0x0 0x0 0x1 0x5 0x4 0x1c 0x7
0x7 0x0 0x1 0x0 0x2 0x18 0x0 0x3
```
**b5b2-sref-gen** (SR 20):
```
0x0 0x0 0x1 0x1 0x0 0x0 0x0 0x1 0x5 0x4 0x1c 0x6000000080000007
0x7 0xc0000001 0x1 0x0 0x2 0x18 0x0 0x3
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
**b5b2-ctxport-f4** (SR 6):
```
0x19 0x0 0x0 0x1 0x0 0x1
```
**b5b2-ctxport-f4ctl** (SR 6):
```
0x0 0x0 0x77 0x1 0x0 0x0
```
**b5b2-ctxport-widelo** (SR 6):
```
0x0 0x0 0x80001000 0x8000206 0x8000208 0x0
```
**b5b2-ctxport-widehi** (SR 6):
```
0x0 0x0 0x80001000 0xfffffe06 0xfffffe08 0x0
```
**b5b2-ctxport-edgem** (SR 6):
```
0x0 0x0 0x0 0x0 0x0 0x3
```

## Differences from the in-tree runs, predicted before the board run (the semantics owner's condition 3)

The in-tree runs are capstone-ariane 2bed0dfe7 / 904cd9c6b with the scaffold (`.sup/s9`, `.sup/reg-s9`,
`.sup/rk-it`). Each difference is attributed; any other difference on silicon is a finding.
- **b5b2-illegal**, positions 55-69 (sites 19-23), 73-75 (site 25) and 76 (the count): those sites trap 2 / 1 / 1,
  and the count is 0x19. Site 24 (positions 70-72) is unchanged. That is step 9's
  deletion with no scaffold. In-tree, the scaffold keeps them retiring and adds site 25's own capprint line. This is
  the semantics owner's condition 2, read in a build without the define.
- **sublet-life**, positions 16-26, the pool dump: CSMINT's root takes pool slot 3, so next is 6 and the records move
  up one. Positions 25-34 stay ungraded (DRAM above next; csmint-bare-2026-11-23's amendment).
- **b5b2-sref / -gen**:
  - 9: 0x5 and 10: 0x4, the ids of x27 and x26 one up;
  - 12-15: a6 at slot 7 (sref-gen: 0x6000000080000007, generation 0xc0000001), its slot 7, its generation, flag 1;
  - the re-key moved the free record to slot 7.
- **b5b2-raw** position 16: the link layout (b5b2-bare-2026-11-24's PREREG explains it).
- **b5b2-ctxport-widelo / -widehi** position 3, data[8]: 0x80001000, since _start sits behind the board prologue.
- Every other image reads exactly its in-tree vector: illegal-runv, illegal-csup, uprot (both), rule2 (both), scratch,
  mwrite, rawlcc, cptag, cptag2, nsedge, ctxport-f4, ctxport-f4ctl and ctxport-edgem.

## The kit's three findings, and the re-key (KIT findings: tables keyed to CAPCREATE-era slots)

The first image sims (tests at 904cd9c6b) found three slot-keyed tests reading as if their subject were absent:
- b5b2-nsedge read exactly its NS_SIM_NOFLUSH mutant vector;
- ctxport-f4 and f4ctl faulted 25 at the cderive;
- sref and sref-gen's a6 arm read 0 / 0xffffffff with its flag 0.
CSMINT's root takes pool slot 3, so every pool id after it moves up one, while the tables were keyed to the
CAPCREATE-era slots. The semantics owner ruled a re-key for all three, with conditions, and 2bed0dfe7 carries it:
- **Re-key:** s8 is a plain address, the lists are re-keyed one up, and the stale and free records move with them.
- **Key check:** word 1 of the minted root's granule must show {0,3}; otherwise the run prints the sentinel
  0x6b657962616420 alone. The deliberately skewed builds (`out/skew`, -DKIT_ROOT_SLOT=4, simulation only) read the
  sentinel alone: nsedge-skew, sref-skew and ctxport-skew.
- **Discrimination, read again:**
  - nsedge's kit image reads the pass vector, and the SAME image under NS_SIM_NOFLUSH (`.sup/rk-mut`; the define
    confirmed in the model's file list) reads the mutant vector `0 0 0 0 0 | 0 0 | 0x19 x4 | 0 0 0x19 0x19 | 6`;
  - sref-gen's a6 reads generation 0xc0000001 with the flag at 1;
  - ctxport-f4 carries the reached-F4 marker (data[16] = 1: the derive produced a capability; nothing between it and
    the ldc), and f4ctl passes with the same marker.

## The hashes behind "byte-identical, readings stand"

| image | first build (904cd9c6b tests) | rebuild (2bed0dfe7 tests) |
|---|---|---|
| `b5b2-cptag.bin` | `8d65c0e54c888ab1574567f57c0cd4ba0c172e97881536602f458d2bbc4237ea` | `8d65c0e54c888ab1574567f57c0cd4ba0c172e97881536602f458d2bbc4237ea` |
| `b5b2-cptag2.bin` | `1a9ebadff577a347a445fdaad2c60be0b196bbb1f367b46863483a7c1061f498` | `1a9ebadff577a347a445fdaad2c60be0b196bbb1f367b46863483a7c1061f498` |
| `b5b2-illegal-csup.bin` | `648c758ac5e8c2ba29e69ae190749fccf10a3e5de6d70af58874afdb1475d070` | `648c758ac5e8c2ba29e69ae190749fccf10a3e5de6d70af58874afdb1475d070` |
| `b5b2-illegal-runv.bin` | `e8136c724fbd7a9cf70e63c28dd4ea0c3a83116cd6fce65b72bfe3ed09f6fdb2` | `e8136c724fbd7a9cf70e63c28dd4ea0c3a83116cd6fce65b72bfe3ed09f6fdb2` |
| `b5b2-illegal.bin` | `41c88fb5f347aab4b121ec66a22e8579f2edf7d7dad0518d4f4118a5bd55fa8d` | `41c88fb5f347aab4b121ec66a22e8579f2edf7d7dad0518d4f4118a5bd55fa8d` |
| `b5b2-mwrite.bin` | `c3649488c6fa4a71a0bd1c12b1449442c9807c69682fcd2272170de1a36862d5` | `c3649488c6fa4a71a0bd1c12b1449442c9807c69682fcd2272170de1a36862d5` |
| `b5b2-raw.bin` | `6a54102b7dd3d13a5f9a85962366c320514403cd39487d589af8dc50b505ccfb` | `6a54102b7dd3d13a5f9a85962366c320514403cd39487d589af8dc50b505ccfb` |
| `b5b2-rawlcc.bin` | `3b6dd196ab5d701be8b8b786a5d8ae77a5f04f912ff5c8ebb40e827e1e86c7de` | `3b6dd196ab5d701be8b8b786a5d8ae77a5f04f912ff5c8ebb40e827e1e86c7de` |
| `b5b2-rule2-pair.bin` | `71c3cf7f91addd1c193d6d2bcb6791624ec1f10ed1b1ba57657a5c18b8695981` | `71c3cf7f91addd1c193d6d2bcb6791624ec1f10ed1b1ba57657a5c18b8695981` |
| `b5b2-rule2.bin` | `2fc07cf51430030489e1727de2e2ae5be65942f8fec73531a6693121157fcc08` | `2fc07cf51430030489e1727de2e2ae5be65942f8fec73531a6693121157fcc08` |
| `b5b2-scratch.bin` | `974c067471dfcb84bc3dfcca8cdf2d0057d945d17b6db8d34f569a60ad68db64` | `974c067471dfcb84bc3dfcca8cdf2d0057d945d17b6db8d34f569a60ad68db64` |
| `b5b2-uprot-mis.bin` | `28bb4f2bea4a0f80fe7ae0817ff1026394ffdc064d9b9e80b0c0df475cc0149f` | `28bb4f2bea4a0f80fe7ae0817ff1026394ffdc064d9b9e80b0c0df475cc0149f` |
| `b5b2-uprot.bin` | `d4734d0fa2fa859e04fae164ff9326a3e4c14a164081f6447cb782e2ea2771c5` | `d4734d0fa2fa859e04fae164ff9326a3e4c14a164081f6447cb782e2ea2771c5` |
| `sublet-life.bin` | `ecbfe9171c0dbbbf9e32db48b6eb4c0c2f2b99eae0b4529c5feb46660e7227a9` | `ecbfe9171c0dbbbf9e32db48b6eb4c0c2f2b99eae0b4529c5feb46660e7227a9` |
| `b5b2-ctxport-edgem.bin` | `93777fa4883a3001e810a136f8f3e21ed0776e55178f2cd33cea459301c79234` | `eda7f42aa4b036d5df2a95ff934ce0a5635671b5183f80dedf772798cf0b2221` **(rebuilt)** |
| `b5b2-ctxport-f4.bin` | `dc265d25bb37637fea87029d634b6a4b2b166d66816a40b8e15221e0edf54663` | `181314f837d650ae5ee30f8bcf98c903f89a59238df10789d5a874460bda64e8` **(rebuilt)** |
| `b5b2-ctxport-f4ctl.bin` | `788e0f38d38d9391c5383cf57cf2e6b3afa88adcfb537216638b0e861fafb298` | `516551fc534ef04e9bb9334d6fae4278ab9e1b0a6e4e5843707b9ec2aa76b81f` **(rebuilt)** |
| `b5b2-ctxport-widehi.bin` | `e63ff0bfaa4021ac3642b44cb043f85a67643210586c85efa02b41b8d5a7009d` | `6cc824e6251f24358fe14c2643b84989da0ed40858a0b4a5299a9544e0ec87bf` **(rebuilt)** |
| `b5b2-ctxport-widelo.bin` | `5e975a144d38410246e9bdfd98e4d691bd99fa0eed6cb33ecf9d277b190ac0fa` | `a63b405dc6afe16dd8a9464918707ccdb9fe51a509e5f1ebf5ee2eed9a8cfde7` **(rebuilt)** |
| `b5b2-nsedge.bin` | `f96ff6b8aef9621aa7370c7e8b803f1d668de5c48461b1a3db36947920cc8cfc` | `58f34a2d73763cc5ee8240e5c251f4a5af30da05561a9ba33e51be2dfdc766fe` **(rebuilt)** |
| `b5b2-sref-gen.bin` | `2c2bbeb93b36fd58ef669ac566f2f1419dc687d68f6cbdb9660545365099a82c` | `6d449998da2ac55330bc2f5fd94d2c70d068f0c245dd6f6ab111a0536a83cb2d` **(rebuilt)** |
| `b5b2-sref.bin` | `a953d02c70c89f1135ccb90949e91084ec8f5483edfb884e0202e90aababfdfa` | `cc6b5db7c544779758cb888464186bfa9454e05e8ab62450919df22c496f8c97` **(rebuilt)** |

## Amendment (2026-11-25, after board 2's first run and before any diagnostic run): b5b2-cptag reading 21 and its diagnostic

**What was read.**
- Board 2 on the seal (run-b2-20261124T2213): 21 of 22 AS PREDICTED, the control exact. The one MISS is b5b2-cptag
  position 22, reading 21, which read 0 where 1 was predicted.
- Repeats on board 2 read 1, 1, 0 (with Tier 1: 0, 1, 1, 0).
- The same image on board 3 (the B5b-2 seal, 0c0458149) read 1, 1, 1.
- Reading 21 is a cycle comparison: a load after CPTAG may cost at most 2 cycles more than the same load before it.
- In the image sims both windows contain a first-time I$ line fetch (before 70 cycles, after 54). So the reading
  compares two fetch misses more than the D$ line at rs1. Not attributed.

**The diagnostic, `b5b2-cptag-diag.bin`:** a separate image, the original untouched. It is the test's -DCPT_DIAG build,
capstone-ariane c6195bea5.
- After reading 21 it records:
  - the cold windows' raw deltas;
  - the same before / CPTAG / after windows run twice, CPTAG on every pass, and the warm pass's raw deltas and flag;
  - a warm-code D$ miss's cost, the load of a never-touched line at 0x80200500.
- Identity: sha256 `2e0bfc3cf3d5e0cb031bc119a30a22bc948d185f159c435abfb32259cc5d75ec`, board_rec 0x80010240, SR 30. Simulated on 904cd9c6b's RTL with no scaffold
  (`.sup/cpdiag2`):
```
0x1d 0x0 0x0 0x0 0x1 0x1 0x0 0x1 0x0 0x5 0x1 0x5
0x1 0x5 0x1 0x5 0x1 0x0 0x0 0x2 0x1 0x1 0x46 0x36
0x4 0x4 0x1 0x65 0x2 0x6
```
- **Graded exactly:** positions 1-21, 29 and 30.
- **Positions 22-28 are timing, decided by this rule** (in the sim: 0x1, 70, 54, 4, 4, 1, 101):
  - 22: the original reading 21, recorded;
  - 23 and 24: the cold before and after deltas, recorded;
  - 25 and 26: the warm before and after deltas;
  - 27: the warm flag;
  - 28: the warm-code D$ miss cost.
  - **VALID** only if position 28 >= position 25 + 10, so the miss scale did miss.
  - **H-fetch** (the instruction-fetch / layout reading): 26 − 25 < 3 and 27 = 1. The CPTAG pass leaves the warm load
    unchanged, so the original's 0 came from the cold fetch windows.
  - **H-line** (CPTAG disturbs the D$ line at rs1): 26 − 25 >= (28 − 25) / 2, at least half this board's own miss
    penalty.
  - Anything else is UNDETERMINED and reported as such.
- The sim reads H-fetch: 4 / 4, flag 1, miss cost 101.
- The proposed load: the control, then the original b5b2-cptag ×1 and the diagnostic ×3, on board 2 (904cd9c6b) and on
  board 3 (0c0458149) alike. The diagnostic runs there too, since 0c0458149 decodes CSMINT
  (decoder.sv:1337-1344).

## The grading rule

- **Before each image:** its sha256 and board_rec must equal the table. The resident name read back must be the seal's.
- **The control:** sublet-life must read SR 35 with positions 1-24 and 35 exact; positions 25-34 are not graded. A
  control miss makes the session VOID.
- **Each other image:** AS PREDICTED only if SR and every value are exact. Anything else is a MISS, reported with its
  positions. An image that prints the sentinel 0x6b657962616420 is a KEY-CHECK FAILURE, never a reading. No SR line is
  NO READING.
- No earlier bitstream's vector is graded.

## Addendum 2026-11-25: R-80's seal a5c45a46d, pre-registered before its sessions

The same 23 images (byte-identical: SHA256SUMS unchanged; every ELF in `out/` reproduces its pinned `.bin`) were simulated on capstone-ariane b5b2-ldcfix a5c45a46d, rtl_dirty=no (s2fix-wt `.sup/finn`), with this kit's settings. Their vectors are in `expected-vectors-a5c45a46d.json`, which has the same fields as `expected-vectors.json` plus `sim_rtl`.

**Every image reads exactly this kit's vector, in every position.** The grading rule above applies unchanged on that seal, with `expected-vectors-a5c45a46d.json`'s `vector` as the column. The kit cannot tell a5c45a46d's seal from 904cd9c6b's by values; the flash is shown by the readback and by the sibling kits whose columns differ (crevoke4b, wapflu).

## Addendum 2026-11-25: the final candidate b42c8c644 (R-80 + R-77's PRE fix), pre-registered before its seal

The same 23 images (byte-identical: SHA256SUMS unchanged) were simulated on capstone-ariane b5b2-prefix b42c8c644 (core = 7861abcfa), rtl_dirty=no (s2fix-wt `.sup/r77n`), with this kit's settings. Their vectors are in `expected-vectors-b42c8c644.json`, which has the same fields as `expected-vectors.json` plus `sim_rtl`.

**Every image reads exactly this kit's vector, in every position.** The grading rule above applies unchanged on that seal, with `expected-vectors-b42c8c644.json`'s `vector` as the column.
