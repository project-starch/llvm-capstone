# memcached/plain-heap-repros: the program's whole Sublet configuration (`sublet-full`), 2026-10-09

**Question.** Column 3 of the per-bug table ("Sublet in nested") for a plain case, by the lead's
choice: run the case in the program's full Sublet configuration -- the Sublet heap AND the program's
nested-allocator port, live -- rather than leave the cell empty.

**Build.** Each case is linked with the slab/cache Sublet port in mode 1 (tools/full-config/memcached.c, the allocators port's leases.c/metadata.c/authority.c; SDK HEAP_LOG 27), whose constructor brings the port up before
`main()` and drives one issue and one revoke through it; every run must print its `FULLCONFIG ...
live` line or `tools/run-capstone-domain.py --arm sublet-full` refuses the run (exit 75).


**Result: 9 of 9 caught, 9 of 9 as pre-registered (aeba7ddefb4d)** --
the same reading as the `sublet` arm on every case. The case's objects come straight from malloc,
so the live port is off the bug's path; the run shows it changes nothing for a direct-allocation
bug. Controls in the same boot as every arm of this tool (clean, oob, uaf, subobj) behaved.

**What this is, and is not.** A non-interference result: every reading is unchanged with the port linked and live, and the port is off these bugs' path, so no row here is a catch by the port. The image carries the slab port's Sublet layer (the ledger, its metadata heap and the Sublet authority), driven directly by the constructor -- slabs.c and cache.c themselves are not linked. The SDK is a HEAP_LOG 27 SDK (the port lends a 64 MiB payload), not the HEAP_LOG 22 one the sublet arm used, so the heap size differs from the `sublet` arm's as well as the port.
