# Native mallocng qualification

ABI v4 on `virtual-musl-policy` uses unmodified native musl 1.2.5 mallocng
through the trusted launcher. The capability SDK supplies ABI wrappers;
Linux core and firmware are unchanged. The custom virtual allocator and its
65,536-block ceiling are removed. Object/page records grow dynamically.

| Gate | Result |
|---|---|
| malloc contract and spatial/temporal safety | 17/17 |
| shared-mm pthreads | 9/9 |
| SQLite persistence, mruby, Perl and 17-section Perl smoke | PASS |
| default-profile M1 / virtual-runtime / U-access | 60/60 / 69/69 / 69/69 |
| all eight native mallocng C/header files vs verified archive | identical |
| changed UNIT and disabled in-place realloc policy controls | both rejected |
| ordinary memcpy instead of tag transfer | tag-loss fault detected |
| omitted retirement on in-place realloc | stale access survives; gate detects failure |

The malloc gate covers native-reference behavior, in-place and moving
realloc, internal mremap, failed-realloc rollback, alignment, zeroing, exact
bounds after store/load, non-linear and linear capability transfers, 70,000
simultaneously live objects and 200,000 lifetimes with actual node collection.
Negative probes require the exact cause and instruction, including old
pointers after actual reuse of the same virtual address. Allocator offset
cycling is allowed to choose reuse; the test does not force allocator policy.

Rebuild the adapter with `build-adapter.sh`, and the virtual SDK and probes
with its generated `capstone-cc`. Compile `malloc-contract.c` also against the
native musl sysroot built by the adapter. Run `run-malloc.py` with `--adapter`,
`--application`, `--native`, `--qemu`, `--images` and a new `--work` directory.
Run `run-pthreads.py --exact-bounds` for threads. Application recipes are the
SQLite/mruby/Perl section of `run.py`, staged using
`run-staged.py --exact-bounds`. Each JSON records binary/input and recipe
hashes; capture logs remain outside the repository.

**Prototype storage contract:** `x-capstone-exact-bounds=true` makes physical
shadow bounds authoritative beside the compressed 128-bit payload. This is
additional metadata, not a one-bit-tag or RTL implementation. The profile
is opt-in; the instruction regression gates exercise the default profile.
No performance claim is made: each allocation crosses the supervised service
boundary. The existing compiler and Linux images were reused; the compiler
freshness check reports incomplete target coverage. The remaining application
ports and complete bug corpora have not been requalified for ABI v4.
