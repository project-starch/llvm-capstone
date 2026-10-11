# Eight cases on CheriBSD under PoisonCap, paired, 2026-09-22

**Mode 0 completes all eight; mode 1 faults on all eight at the address the
supervisor resolved for `apr_defect_read`** -- SIGPROT, si_code 2, one fault
per arm, every pair paired. The same binary in both arms; only the mode
argument differs.

    matrix.tsv    one line per arm: verdict, exit, fault PC, resolved probe address, adapter counters
    inputs.json   binaries, platform fingerprint, the two platform controls

Mode 0 publishes every node and every bucket piece as an exactly bounded
alias without poison authority and invalidates nothing: the allocators as
upstream ships them, under exact bounds, `sweeps=0` on every arm. Mode 1
poisons and sweeps a node when APR files it (`aprp_node_release`), which is
where cases 2, 3, 5 and 7 end (a pool destroy) and where cases 0, 1, 4 and 6
end (a bucket allocator's blocks returning to APR, at its destroy or at the
handback the reduced consumer declares); a bucket piece is poisoned at its
individual `apr_bucket_free`, a transition none of the eight reaches and the
port's fixtures cover. Case 4 is the class-3 case: no free, so this
mechanism sees it only because the reduced consumer ends the allocator's
tenancy at the handback -- the same declaration the sublet arm needs, acted
on from the memory side.

Platform: the PoisonCap work tree's image with the local libc fix the CPython
corpus documents (`libc 6726fdb0…`), guest libc revocation on and read back
by the ABI probe. This is the second attempt of the day: the first aborted
after the first pair because the runner declared a report output for the
faulting arm, which dies before writing one; no verdict was lost, the suite
simply did not continue. Raw guest logs are archived outside the repository
with the corpus's at `~/artifacts/httpd/20260922-bucket-corpus/raw-campaign.tar.gz`.
