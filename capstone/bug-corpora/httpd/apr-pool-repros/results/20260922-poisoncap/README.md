# The httpd/APR case on CheriBSD under PoisonCap, paired, 2026-09-22

**Mode 0 completes; mode 1 faults at the address the supervisor resolved for
`apr_defect_read`** -- SIGPROT, si_code 2, one fault, the pair paired. The
same binary in both arms. Mode 0 publishes every node as an exactly bounded
alias without poison authority; mode 1 poisons and sweeps the node when APR
files it at `apr_pool_destroy`, so the handle mod_watchdog kept across the
destroy is dead by the time the next pool takes the node.

    matrix.tsv    one line per arm: verdict, exit, fault PC, resolved probe address, adapter counters
    inputs.json   binaries, platform fingerprint, the two platform controls

The port's PoisonCap adapter is `ports/apr/pools/src/cheribsd/node-poison.c`
(`-DAPRP_POISONCAP=ON`), built through `shared/build-cases.sh poisoncap` and
run by `runners/poisoncap/run-defects.py`. Platform: the PoisonCap work
tree's image with the local libc fix the CPython corpus documents, guest libc
revocation on and read back by the ABI probe. Raw guest logs are archived
outside the repository with the bucket corpus's campaign.
