# Controls: parent, sibling and successor at the SV head arena

The Perl cases of the pre-registered controls
([`docs/plans/2026-10-11-nested-lifetime-controls.md`](../../../../docs/plans/2026-10-11-nested-lifetime-controls.md)).
They ask the opposite question to the corpus cases: does the lifetime patch
(`ports/perl/sublet/patches/5.36.3/0001`) leave the accesses Perl is allowed to make alone?

The corpus's cases are Perl scripts; these drive the allocator from C. `svhead-controls.c`
embeds libperl and takes heads through Perl's own API (`newSViv`, `newSV`), releasing them with
`SvREFCNT_dec` to 0, which reaches `plant_SV`. One mode per process:

| mode | what it does | expected, protected | expected, release |
|---|---|---|---|
| `parent` | release A, then read (flags, free-list link, refcnt) and write (refcnt, then restore) A's former head through the arena's own pointer, `Perl_sv_arena_head` | completes; flags SVTYPEMASK, link = the previous `PL_sv_root` | the same |
| `sibling` | A and B consecutive heads of one arena; release A; read and write B through B | completes, B intact | the same |
| `successor` | release A; `newSV` until a head at A's address comes back (C); read and write through C | completes, C = A | the same |
| `stale` | `successor`, then read through the old pointer to A (`stale-read-flags`) | fault, cause 25, at that probe | completes, reads C's head |

Each access goes through a `noinline` `probe_*` function and is announced first as
`PROBE <label> svhead-controls.c:<line>`, so a fault's pc and the last announced line both name it.

- `build-controls.sh PERLD_ROOT OUT.dom` links the harness against one build of
  `ports/perl/musl/build-perl-domain.sh` (virtual profile, `PERLD_SUBLET=0` or `1`), with that
  build's CFLAGS and SDK.
- `run-controls.sh ARM IMAGE KIT WORK` runs the four modes in one boot on the registered platform.
- `collect.py` reads the serial logs into one JSON (outcome, addresses, probes, fault cause, pc,
  function and source line), and refuses a log whose in-guest image hash differs.

Results: [`../results/2026-10-11-controls/`](../results/2026-10-11-controls/README.md).
