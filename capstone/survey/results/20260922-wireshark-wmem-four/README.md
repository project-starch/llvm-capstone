# One manager, four allocators, three orders of magnitude apart

The four `wmem` levels on Wireshark 4.6.8, the revision
`capstone/ports/wireshark/wmem` pins. The survey had no Wireshark target at
all before this, and these four allocators carry twelve of the defect
corpus's cases.

## The four, repetition 1

| level | processes | allocations | reuses | before release | distinct addresses |
|---|---:|---:|---:|---:|---:|
| `block` | 163 | 7,714,525 | 15,040 | 100.0 % | **7,677,929** |
| `block_fast` | 162 | 9,270,146 | **9,003,932** | 100.0 % | **266,233** |
| `simple` | 2 | 8 | 0 | — | 8 |
| `strict` | 0 | — | never created | — | — |

`block` hands out 7.7 million objects over 7.68 million distinct addresses,
so it almost never gives one back out: 15,040 reuses in the whole run.
`block_fast` hands out 9.3 million over 266,233 addresses and 97 per cent of
its allocations are reuses. Same manager, same captures, both reusing before
any backing release, and three orders of magnitude between them in how often
an address comes round again.

That is the port's own description of the two, measured rather than
repeated: `block_fast` keeps its first block and rewinds, `block` keeps
every block and rebuilds its free lists.

`simple` is created twice and serves eight objects. `strict` is never
created, which is not a gap in the measurement but the corpus's own point:
upstream reaches these defects only by substituting `strict` through
`WIRESHARK_DEBUG_WMEM_OVERRIDE`, so an ordinary run has none.

## The oracle is a hash

Both arms dissected the same captures with `-V` and the two outputs hash to
`d2cc12fd…`. Equal hashes are what make the instrument inert, and they are
stronger than a `cmp` verdict because the number travels with the bundle.

Those outputs are 169 MiB each and reproducible from the pin, so `run.sh`
hashes them as they stream rather than writing them down. The first run left
2.4 GB in the bundle of which 4.6 MB was evidence, which is the mistake this
records so it is not made twice.

## One file, four levels

`wsutil/wmem/wmem_core.c` owns the switch that installs a type's function
pointers, so that is the one place a wrapper sits and the four allocator
files stay pristine. `A1_WMEM_LEVEL` picks the allocator at startup, because
the counting core carries one custom level per process and four levels would
otherwise mean four builds of Wireshark. The container is `private_data`,
which is per pool, so two pools of one type stay apart.

## Provenance

| | |
|---|---|
| Wireshark | 4.6.8, sha256 `c0f1ccf2…9d0` |
| pin | matches `capstone/ports/wireshark/wmem/upstream.json` exactly, a second independent record |
| workload | the 163 captures the pinned tree ships that tshark can read, `-V` |
| capture choice | decided by asking tshark, not by extension: the tree also carries compressed captures and text records |
| build | cmake Release, tshark only, no pcap, no lua, no plugins |
| installed for it | libpcap-dev, libgcrypt20-dev, libc-ares-dev |
| compiler | gcc 13.3.0 |
| raw | two reports per level, the oracle hashes and the provenance, hashed in `raw/SHA256SUMS` |

## Limits

One workload. A small excess of frees over allocations, four on `block` and
240 on `block_fast`, is the counting core's own rule that a bulk death
counts as a free as well as in the bulk count.
