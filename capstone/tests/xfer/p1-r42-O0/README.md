# P1 -O0 pair for `caplifive_r42_6cbdaeeb4` — TRANSFER BRANCH, NOT FOR MERGE

Compiler-capable host → the board lane on apollo, which cannot compile the SQLite domain TU
(`capstone/docs/design/hosted-libc-os-analysis.md:24-34`). Cite by hash, never by filename.

## THESE ARE E2's BYTES. NOTHING WAS REBUILT.

Both images are the *exact* programs of record from boots sw75 and sw79, recovered from this host
and verified **by content**:

| arm | image | sha256/16 | `movc` |
|---|---|---|---|
| cell ⑤ — memsys5 + pool, lookaside `1200,40`, 2 MiB static heap | `cell5-memsys5-O0.dom` | `e6ee5255c896aa21` | 6,755 |
| cell ⑥ — Sublet | `cell6-sublet-O0.dom` | `ceeded2533a74bce` | 6,808 |

`movc` density confirms **-O0** — an -O1/-O2 image of this workload runs ~17k.

**Why not a rebuild, stated because it is the load-bearing decision here.** The building host is 408
commits behind dev with a toolchain built 2026-09-17, and the compiler lane confirms **fifteen
commits touch `llvm/`/`clang/` since**, seven of them distinct codegen fixes — C-52 (local stack-slot
base as a capability) changes register pressure and C-58 (MachineLICM speculating capability
instructions) changes what may be hoisted. Both move cycle counts. A rebuild today is therefore a
**different program**, and any comparison against E2's numbers would be two-variable. "Same geometry
as E2" means E2's bytes, so E2's bytes are what this branch carries.

## The runs, and which log serves which gate

| file | run | key line |
|---|---|---|
| `cell5.serial.log` | cell ⑤ at its sw75 geometry — `--speedtest1 --testset main --size 1 --verify`, **no** `--arena`/`--tables` | `SPEEDTEST1-CYCLES 678572868 HIGHWATER n/a HEAP 2097152 DROPPED 0 RC 0` |
| `cell6-2mib-arena.serial.log` | cell ⑥ at `--arena 2097152 --tables 1750285` — **this is `board-b79.sh:40`'s `$QEMU_LOG_CELL_2MIB`** | `SPEEDTEST1-CYCLES 692392094 HIGHWATER n/a HEAP 1344064 DROPPED 0 RC 0`, `sublet: split=5568 mrev=37966 delin=32565 revoke=37966 init=5401` |
| `cell6.serial.log` | cell ⑥ at the **default** arena (1,419,584) — included for completeness, **not** the b79 denominator | `... HEAP 911104 ...`, `sublet: split=5481 mrev=37874 …` |

Both cell ⑥ logs are the same image; arena and tables are host arguments and never enter the image.
`cell5.qemu-pass.record` / `cell6.qemu-pass.record` are the pass records keyed by image sha256; note
cell ⑥'s record points at the **default-arena** run, which is why the 2 MiB log is supplied
separately.

All arms carry the oracle `Verification Hash: 112006 38bb59fd…3925d8518`, `DROPPED 0 RC 0`.

## Three things a later reader needs, which the files alone do not say

**1. The invocation, because the pass record's `args=` line is NOT it.** Arena and tables are
separate argv elements BEFORE the quoted benchmark string; the host lifts them out of argv before
the join, so `--arena` written inside the quotes is a speedtest1 flag and not a region request:

    --speedtest1 --arena 2097152 --tables 1750285 '--testset main --size 1 --verify'

Cell ⑤ takes no `--arena`/`--tables` at all. A reader who copies the `args=` line out of a
`.qemu-pass` record reproduces the DEFAULT-arena run, which is not the b79 denominator.

**2. There is one pass record per IMAGE, and it is the default-arena run's.** The record system
keys on the image sha256, so `cell6.qemu-pass.record` exists and satisfies a gate that wants
`qemu-pass/<full sha256>` — but its `args=` and `log=` describe the default-arena run. The 2 MiB run
has a log here and no record of its own. That is the record system's shape, not a missing artefact.

**3. How the 2 MiB log is tied to THIS image.** The log carries no image hash, so the link is
argued rather than read off. Both arena runs came from one sweep 47 seconds apart, and the sweep's
default-arena arm produced `SPEEDTEST1-CYCLES 690051663 … HEAP 911104` with `sublet: 5481/37874` —
byte-for-byte the numbers in this image's own recorded pass. The emulator is deterministic and those
numbers are image-specific, so the sweep ran this image. If you want the chain closed by a hash
rather than by determinism, re-run the 2 MiB arena over `cell6-sublet-O0.dom`: it must reproduce
692392094 exactly.

## The 2 MiB log is the ORIGINAL sw79 pre-run, and identifying it needed provenance, not content

Six files on the building host contain the exact string
`SPEEDTEST1-CYCLES 692392094 HIGHWATER n/a HEAP 1344064`. **Content matching cannot pick the right
one**, and three of the six are 55–129-byte synthetic fixtures written two days later to
negative-test the gate. One of them, `rec-wrongtables.log`, reads:

    == Sublet: pool 2097152 bytes (arena, REV_BORROWED), tables 2523136 bytes
    SPEEDTEST1-CYCLES 692392094 HIGHWATER n/a HEAP 1344064

— the pool-raised wrong-tables configuration, carrying a byte-identical gate line. That is the gate
defect the board lane fixed, preserved as a fixture.

The file transferred here is the 79,204-byte capture from 2026-09-14 20:21, which carries a real
boot banner, the 2 MiB sublet counters `5568/37966/32565`, and the oracle hash. It was chosen on
**size, timestamp and the presence of a full serial capture**, not on containing the gate line.

## Redaction

The three `.log` files and both `.record` files are **not byte-original**: identifier substitutions
only, nothing else — no cropping, line counts identical to the originals. Three literal byte
substitutions were applied over each whole file: the operator's account name in a `<account>@<host>`
kernel build banner, the same account as a `/home/<account>` path component, and the same account
inside the harness's encoded scratchpad path segment `-home-<account>-dev-…`. **The removed strings
are not reproduced here** — writing the account name into a committed file is what the redaction
exists to prevent, and `precommit-scan.sh` blocks a README that quotes it.

Pre-redaction sha256, so anyone holding an original can diff and see every byte that moved:

    cb1467447640f5a5…  cell5.serial.log
    15dc9d271d14f201…  cell6.serial.log
    0c68d755aace42fa…  cell6-2mib-arena.serial.log
    a924f903610821ec…  cell5.qemu-pass.record
    5df66d5ec1853898…  cell6.qemu-pass.record

The two `.dom` files are **verbatim** — no transformation, and their hashes above are the hashes of
record. `SHA256SUMS` covers every file **as committed**, i.e. logs post-redaction, so
`sha256sum -c` passes as-is.

**Redaction cannot disturb the gates, and that was checked rather than assumed** — re-run after the
substitution: the b79 2 MiB line rc=0, cell ⑤'s sw75 line rc=0, the 2 MiB sublet counters rc=0, the
oracle hash rc=0. Negative control: `HEAP 1344064` is **absent** from `cell6.serial.log` (rc=1),
confirming the two cell ⑥ logs are different runs rather than copies.

## Geometry, as agreed

Cell ⑤ runs at its sw75 shape with no `--arena`/`--tables` — it has a compiled-in 2 MiB static heap
and takes neither flag. Cell ⑥ runs `--arena 2097152 --tables 1750285`. **Two boots, one SQLite
image each**, so both may stay at entry VA `0x410000`; two distinct SQLite images in one boot would
need distinct VAs or R-3 hangs the second.

Nothing here is a board result. These are emulator records and the images they belong to; the
silicon verdict is the board's.
