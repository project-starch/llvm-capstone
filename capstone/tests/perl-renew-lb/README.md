# perlbench CRENEW on Line B: the experiment arms and their twin runner (nested lane, 2026-11-26/27)

These are the instruments behind the section "perlbench item 3 on Line B: CRENEW for heads, bodies and hash entries,
failing closed, on the trap-route twin" in `capstone/docs/ref/nested-allocator-workloads.md`. They cover its
pre-registration and its Result, which found that the variant costs 4,250 image instructions where a saving was
predicted.

**They are experiment arms, not port scripts.** The port is in
`capstone/ports/spec-cpu2017/own-allocators/500.perlbench_r/`, and these scripts edit a tree that the port's scripts
have already produced:
- `heads-renew.py` is the 5.36 heads variant in the 5.22 port's shape. Under `PERL_CAPSTONE_SUBLET_HEADS_RENEW` the
  arena renews a head's issued lifetime instead of revoking it, and on a failed renewal it falls back to the port's
  revoke.
- `bodies-renew-lb.py` and `he-renew-lb.py` are the bodies and hash-entry variants, failing closed. A failed
  renewal revokes and returns the segment's own pointer.
- `renew-shim.h` is force-included. It defines `PERL_CAP_RENEW`, and under `PERL_CAPSTONE_RENEW_FORCE_FAIL` every
  renewal reads as failed without executing CRENEW.
- `build-renew.sh` builds the arms from B5TL1's own tree:
  - perl-fullren;
  - perl-fullren-ff;
  - perl-fullren0, which is byte-identical to perl-full.

  `inputs.txt` lists the compiler and the inputs it hashed.
- `build-renew-fixtures.sh` builds the fixtures with the port's committed `make-*-fixture.py`.
- `twin-renew.sh` runs one stage per boot on the trap-route twin, booting through `boot-plugin.py`. That is the
  runtime lane's `boot-long.py` with two changes: it adds `-plugin`, and it waits for QEMU's own exit.
- `opcount.c` is the TCG plugin that counts executed opcode-0x5B instructions by region.

The diffs of the edited perl sources are not committed, because their context lines are licensed SPEC source.

The copies replace the account's paths with `<hdd>/` and `<home>/`, so they do not run as committed.
`RAN-SHA256.txt` gives the sha256 prefixes of the files as they ran. They match the hashes the pre-registration
quotes.
