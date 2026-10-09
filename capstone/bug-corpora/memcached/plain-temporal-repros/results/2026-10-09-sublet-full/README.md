# memcached/plain-temporal-repros: the program's whole Sublet configuration (`sublet-full`), 2026-10-09

**Question.** Column 3 of the per-bug table ("Sublet in nested") for a plain case, by the lead's
choice: run the case in the program's full Sublet configuration -- the Sublet heap AND the program's
nested-allocator port, live -- rather than leave the cell empty.

**Build.** Each case is linked with the slab/cache Sublet port in mode 1 (tools/full-config/memcached.c, the allocators port's leases.c/metadata.c/authority.c; SDK HEAP_LOG 27), whose constructor brings the port up before
`main()` and drives one issue and one revoke through it; every run must print its `FULLCONFIG ...
live` line or `tools/run-capstone-domain.py --arm sublet-full` refuses the run (exit 75).


**Result: 3 of 3 caught, 3 of 3 as pre-registered (aeba7ddefb4d)** --
the same reading as the `sublet` arm on every case. The case's objects come straight from malloc,
so the live port is off the bug's path; the run shows it changes nothing for a direct-allocation
bug. Controls in the same boot as every arm of this tool (clean, oob, uaf, subobj) behaved.
