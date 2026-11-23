# Patches measured but not applied by the builds

`ports/perl/musl/build-perl-domain.sh` applies every patch in `../patches/<version>/` when `PERLD_SUBLET=1`. A patch
here is not applied by any build in this tree: it would change every such image, and the results recorded for those
images name the patches they were built with.

None is pending now.

`5.36.3/0002-sv-arena-lookup-by-index.patch` (sha256 `04540913…`) was here until 2026-11-24 and is now
`../patches/5.36.3/0002`. It moved after this port's own suite ran on an image built with it:
`bug-corpora/perl/release-differential/results/2026-11-22-virtual-0002/` (0001 and 0001+0002 agree in all 11 cases
and all 47 qualify files). Images recorded under earlier results were built without it and name the patches they
were built with; they are not rebuilt.
