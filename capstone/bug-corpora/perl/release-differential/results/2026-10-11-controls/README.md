# Perl 5.36.3: parent, sibling and successor controls at the SV head arena, 2026-10-11

The Perl row of the pre-registered controls,
[`docs/plans/2026-10-11-nested-lifetime-controls.md`](../../../../../docs/plans/2026-10-11-nested-lifetime-controls.md)
(committed in 53325bba0d5c before any harness existed). Cases: [`../../controls/`](../../controls/README.md).
Data: [`results.json`](results.json). Scripts: [`build.sh`](build.sh) (the two Perl builds),
[`run.sh`](run.sh) (link the controls, boot both arms, collect).

**Every prediction holds, on both measured rounds.** The stale read through A is rejected with
cause 25 at its labelled probe on the protected build, and on the release build it completes and
reads C's head: the positive control fired.

## What ran

- Allocator: Perl's SV head arena. A head is taken with `newSViv`/`newSV` (`new_SV` -> `uproot_SV`)
  and released with `SvREFCNT_dec` to 0 (`sv_free2` -> `del_SV` -> `plant_SV`), the release
  operation patch 0001 revokes at (CREVOKE in `plant_SV`).
- Harness: `controls/svhead-controls.c`, libperl embedded in C (`perl_alloc`, `perl_construct`,
  `perl_parse` of `-e 0`, `perl_run`, then the control), linked against each build's own
  `libperl.a` with its own CFLAGS and SDK. One process per control.
- Parent reference: the arena's own pointer to A's former head. On the protected build it is
  `Perl_sv_arena_head(A)` from patch 0001, which the harness checks against its own walk of
  `PL_sv_arenaroot` (same address); the release build has no such function, so the walk alone
  (the same code) gives it.
- llvm-capstone 53325bba0d5c (`ports/nested-lifetime-controls`), snapshot by `git archive` of
  `capstone` and `compiler-rt/lib/builtins`; uncommitted: this directory's scripts and
  `controls/` (sha256 of each in `results.json`). The Perl builds use committed files only.

| | release | protected |
|---|---|---|
| configuration | `PERLD_SUBLET=0`, port patches 0001-0011 | `PERLD_SUBLET=1`: 0001-0011 + `sublet/patches/5.36.3/0001`, `-DPERL_CAPSTONE_SUBLET` |
| perl image | 312d398e6071b38f... | 98d67b3dca5e6985... |
| libperl.a | b9f40d1a63c66e5c... | dd3f9afb0b444566... |
| controls image (run) | **65cb418d7908b01b...** | **be3ae1ed2bf254c5...** |
| Perl build wall | 357 s | 358 s |

Platform as registered: QEMU af37cc32, launcher 3975314a, module a1c6cb6b, firmware 44cd6cbc,
kernel e833dfb1, rootfs 62049800, virtual profile, exact bounds, one hart
(`runtime/virtual/run-staged.py`). One boot per arm per round, both arms in parallel with their
own QEMU locks; the image hash is rechecked inside the guest. Boot walls 26-29 s; each control
process 4.4-6.2 s (guest uptime). Compiler clang 4c6135fe, SDK `capstone-cc` 8e4bd500.

## Results (measured round 1, tag 20261011-231325)

Probe lines are the call sites in `controls/svhead-controls.c`, announced on stdout before each
access; the fault's pc is symbolized on the image (its `-g` line table).

| control | arm | outcome | addresses | probes | reading |
|---|---|---|---|---|---|
| 1 parent | release | completed | A 0x3fb7e2a550; arena 0x3fb7e29590; parent ptr 0x3fb7e2a550 | :130-:136 | flags 0xff (SVTYPEMASK), link 0x3fb7cfd9a0 = the old `PL_sv_root`, `PL_sv_root` = A; refcnt write 0x5a5a5a5a read back, restored |
| 1 parent | protected | completed | A 0x3f957ee510; arena 0x3f957ed550; parent ptr 0x3f957ee510 (`Perl_sv_arena_head` = walk) | :130-:136 | the same: flags 0xff, link 0x3f957c18b0 = the old `PL_sv_root`, write read back |
| 2 sibling | release | completed | A 0x3fa3f256e0, B 0x3fa3f25710 (one arena, B - A = 48) | :178-:182 | B flags 0x1101, IV 0x0b0b0b02 intact, refcnt 1; write read back, `SvIV`/`sv_setiv` agree |
| 2 sibling | protected | completed | A 0x3f940fa4e0, B 0x3f940fa510 (one arena, B - A = 48) | :178-:182 | the same |
| 3 successor | release | completed | A = C = 0x3fba1a6550, first `newSV` | :222-:225 | C IV written and read back |
| 3 successor | protected | completed | A = C = 0x3fa58d2510, first `newSV` | :222-:225 | the same |
| 3 stale (own run) | release | **completed** | A = C = 0x3f92a9e550 | :236, :237 | through A: flags 0x1101, IV 0x0c0c0c05 = **C's head** |
| 3 stale (own run) | protected | **fault, cause 25** | A = C = 0x3fa0c02510 | last announced :236 (`stale-read-flags`) | pc 0x3f9f8153b8 = `probe_read_flags`+20, `svhead-controls.c:38`; fault address 0x3fa0c02524 = A + 0x14, A's `sv_flags` |

Measured round 2 (tag 20261011-231440, the same images): every row the same outcome, result,
probe and cause; protected stale again cause 25 in `probe_read_flags` (`:38`), fault address
0x3f8fc86524 = A + 0x14; C again at A's address on the first `newSV`. Addresses differ per
process and are in `results.json`.

What the bounds show (the `PTR` lines): on the release build every head pointer carries its
whole arena's bounds (A: base 0x3fb7e29590, end 0x3fb7e2a580, 4080 bytes); on the protected build
A, B and C are each bounded to their own 48-byte head, while the arena pointer spans the arena
plus patch 0001's issued[] trailer (5424 = 4080 + 84 x 16 bytes).

## Prediction verdicts

| prediction | release | protected |
|---|---|---|
| control 1: completes, reads what the allocator left there | holds | holds |
| control 2: completes, B intact | holds | holds |
| control 3: C's address equals A's, access through C completes | holds | holds |
| stale access through A | completes and reads C's data: holds | rejected, cause 25, at the labelled probe: holds |

## Deviations and caveats

- **A first run is superseded, not counted** (tag 20261011-231151, `results.json` → `first_run`).
  Its sibling mode took B as the next heads after A and required B in A's arena. A was the last
  free head of its arena, so none of 4096 `newSViv` landed there, and the mode ended before any
  access, on both arms (no probe announced; `RESULT sibling FAIL no B in A's arena`). The mode now
  takes A and B as two consecutive heads that share an arena (3 tries on both arms). That run's
  other three modes gave the outcomes above. Both measured rounds use the changed source.
- The Perl images are not byte-identical to tonight's glob-fix images built from the same tree
  (a5efdd17 / 09e87eae). Build paths differ; that this is the only cause was not checked.
- The release arm's parent reference is the harness's walk of `PL_sv_arenaroot`, the code of
  `Perl_sv_arena_head`, since the release build has no such function; on the protected build the
  patch's own function is used and agrees with the walk.
- N = 2 rounds per arm, the same images. The build host's clock was 26 minutes behind UTC (measured against this host);
  the times in the logs are its own.
