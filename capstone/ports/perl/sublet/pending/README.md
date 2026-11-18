# Patches measured but not applied by the builds

`ports/perl/musl/build-perl-domain.sh` applies every patch in `../patches/<version>/` when `PERLD_SUBLET=1`. A patch
here is not applied by any build in this tree: it would change every such image, and the results recorded for those
images name the patches they were built with.

| patch | applies after | what it changes | measured |
|---|---|---|---|
| `5.36.3/0002-sv-arena-lookup-by-index.patch` (sha256 `04540913…`) | `patches/5.36.3/0001` | `S_sv_arena_of` finds a head's arena in an address-sorted array by bisection and reads no arena; 0001 walks `PL_sv_arenaroot`, one load through each arena | `suns.pl` on silicon (bitstream P1): the protected interpreter costs +16.05 % in-context cycles with it and +208 % without (`docs/ref/nested-allocator-workloads.md`, 2026-11-18) |

Before it moves to `patches/`: it has run `suns.pl` on the board with stock's output and, ported to Perl 5.22.1, SPEC's
`test.pl` (`ports/spec-cpu2017/own-allocators`). It has not been run through this port's own suites
(`bug-corpora/perl/release-differential`), and those results would then need their images rebuilt.
