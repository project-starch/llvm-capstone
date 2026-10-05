# Native allocator lifetime survey

This experiment measures object lifetimes and address reuse in the eight QE
applications on an ordinary Linux/glibc host. It supplies evidence for the
problem statement. It does not measure temporal protection, exploit detection,
Capstone overhead, CHERI overhead or native application performance.

The branch starts from `dev` at `c03acf684f8715c5a15b96f8091748a59a530133`.
Application sources match the port versions and remain in scratch. The binaries
use upstream allocation policies with observation-only source hooks. No
Sublet adapters, sanitizers, quarantine or delayed free are enabled.

## Workloads and their scope

The exact parameters are in [workloads.json](workloads.json). Every profile has
three fresh baseline processes and three fresh observed processes. Each of the
six Wireshark inputs is a separate profile. All selected profiles are retained.
The first campaign has 16 profiles, eight applications and 96 processes.

| Application | Selected work | Why this input | Functional oracle |
| --- | --- | --- | --- |
| SQLite 3.22.0 | Upstream `speedtest1`, main, size 10. Default heap and a separate 128 MiB memsys5 configuration | SQLite uses this benchmark for its own engineering measurements | Independent `integrity_check` and canonical final database dump |
| PostgreSQL 17.5 | The upstream pgbench `tpcb-like` transaction SQL, scale 1, 1,000 transactions | Established transactional SQL pattern, adapted to the port's single-user scope | All SQL results and independently calculated balance and history totals |
| CPython 3.13.7 | pyperformance 1.14.0 `json_loads` and `json_dumps`, fixed iteration counts | Existing interpreter benchmarks with stable datasets | Round trips, payload checksums and exact call counts |
| mruby 4.0.0-rc2 | Upstream AO renderer, width 32 | The project's allocation-heavy application benchmark, already in the port's scope | Complete PPM output and dimensions |
| Perl 5.36.3 | Sequential Benchmarks Game binary-trees, depth 12 | Established allocation-heavy interpreter workload | Analytical tree-check totals and exact output |
| FFmpeg 9.0.1 | Two pinned FATE MPEG-4 inputs, Xvid and resolution change | Upstream decoder regression inputs with different buffer behavior | Every frame's MD5 and expected frame counts of 20 and 150 |
| Wireshark 4.6.8 | Six preselected upstream captures for HTTP, DNS, TCP, SIP/RTP, DHCP and TLS | Protocol diversity from the project's regression corpus | Complete decoded protocol trees and packet counts |
| memcached 1.6.45 | memtier 2.2.1, 1,000 keys, 256-byte values, 1,000 warmup sets and 50,000 mixed operations | Established cache traffic generator with repeated overwrites | Exact server operation counters, zero misses and readback of all final keys |

The PostgreSQL SQL body is extracted from its pinned `pgbench.c`. The harness
instantiates variables using a fixed Python PRNG seed. It does not run the
pgbench client or reproduce its concurrency and random stream. This is an
adapted pgbench SQL workload, not a TPC-B result or a pgbench TPS measurement.

The Python benchmark functions and datasets are unchanged. A fixed-count harness
replaces pyperf's time calibration. FFmpeg runs the complete CLI decoder on FATE
inputs, with its native frame output checked against the baseline. These are
adapted FATE input runs, not official FATE scores. Binary Trees and AO are
allocation-heavy workloads, not representatives of all interpreter use.

The current window includes process initialization and teardown. memcached also
includes warmup and final readback. Wireshark initialization can dominate small
captures. Those limits remain attached to the results. The input selection does
not establish deployment prevalence or vulnerability frequency.

Primary sources:

- [SQLite measurement methodology](https://www.sqlite.org/cpu.html)
- [PostgreSQL 17 pgbench](https://www.postgresql.org/docs/17/pgbench.html)
- [pyperformance benchmarks](https://pyperformance.readthedocs.io/benchmarks.html)
- [mruby benchmark directory](https://github.com/mruby/mruby/tree/9d523e2f74f2e63ca02840937523de61398a617d/benchmark)
- [Binary Trees specification](https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/binarytrees.html)
- [FFmpeg FATE](https://ffmpeg.org/fate.html)
- [Wireshark test execution](https://www.wireshark.org/docs/wsdg_html_chunked/ChTestsRun.html)
- [memtier benchmark](https://github.com/redis/memtier_benchmark/tree/2.2.1)

## What the counters mean

For each allocator family and process:

- `alloc` is a successful object issue. Failed allocations do not contribute.
- `free` is an observed retirement, including objects retired by a bulk reset.
  `bulk_free` is its subset retired by reset or destruction. Objects freed
  individually are removed from the live list before a later reset.
- `reuse` is an issue at an address previously retired in that allocator family.
  It uses the exact object start, not an overlap of byte ranges.
- `inside` is the subset whose old and new object belong to the same observed
  backing generation. That system allocation stayed live across the reuse.
- `outside` has two known, different backing generations. It does not imply
  that any particular protection mechanism would have detected a bug.
- `unknown_reuse` has an unknown backing at either endpoint. It never counts
  as `inside`. The campaign refuses incomplete backing coverage.
- `inside / reuse` states what fraction of observed same-start reuse remained
  inside a live backing allocation. It is undefined when `reuse` is zero.
- `inside / alloc` includes first uses in the denominator. Always report it
  alongside `inside / reuse` and the raw counts.

The backing boundary is the process's libc allocation interface, supplemented
by explicit CPython mmap-arena acquisition and release hooks. It is not the
kernel's `mmap` or `brk` boundary for all allocations. A successful libc realloc
starts a new backing generation, conservatively including an in-place realloc.
Unknown libc-internal allocation paths are not inferred from addresses.

The recorder uses monotonically increasing backing generations. Releasing a
block and acquiring another at the same numeric address produces `outside`,
not `inside`. A range must contain the full requested object extent. SQLite's
memsys5 and lookaside observations can therefore share an ultimate libc root
without counting an intermediate allocator release as a libc release. The two
families remain separate, so summing their event counts would double-count
different levels of the same allocation hierarchy.

Object histories are keyed by family and start address. Allocator instances
have separate generation identifiers and allocation clocks. Destruction retires
an instance generation. Reuse across instances is counted but has no local
reuse gap. Process address spaces are never combined.

The reuse gap is the current successful-issue clock minus the retirement clock
in the same instance. A gap of one means the next successful issue. Bin `b`
covers gaps from `2**b` through `2**(b+1)-1`. `gap_inside` contains within-backing
reuse. `gap_other` includes known outside and unknown reuse. State the fraction
of `inside` events with a local gap before interpreting its distribution.

The live-object peak is a diagnostic for this instrumented process. Byte
counters sum each object's last observed issue or resize size. Some internal
in-place resize paths do not cross the selected hooks, so these byte counters
are not complete requested-memory measurements. They are not used as results.
Neither counter measures RSS, backing overhead or quarantine cost.

## Hook boundaries

| Family | Birth and retirement |
| --- | --- |
| SQLite lookaside | Successful slot issue and return. Reconfiguration and connection destruction retire the instance |
| SQLite memsys5 | Successful unsafe allocation and free, under the original allocator lock |
| PostgreSQL AllocSet, Generation, Slab, Bump | Method-table alloc, free, realloc, reset and delete. Aligned redirects are not counted again |
| CPython pymalloc | Only allocations and frees actually served by pymalloc. Raw fallbacks remain libc objects |
| mruby GC slots | `mrb_obj_alloc` and actual `obj_free` during sweep or teardown |
| Perl SV heads | `uproot_SV` and `plant_SV`. SV bodies and other pools are outside this family |
| FFmpeg AVBufferPool and AVRefStructPool | Successful pool lease and final-reference return. Merely taking another reference is not a new lifetime |
| Wireshark simple, strict, block, block_fast | Public wmem calls, realloc and bulk lifecycle. No-op block_fast frees are counted separately and do not retire retained storage |
| memcached slabs and object cache | Real issued items and returned items, excluding initial slab carving. Both object-cache retention and free-to-libc branches retire the object |

An instrumented family that a workload never exercises is not a measured
allocator. The result table must preserve that distinction. A moved realloc
retires the old object only when that allocator actually releases it. A failed
realloc preserves it. A successful in-place realloc is not address reuse.

## Reproduction

Run from the repository root. Docker supplies native Linux/glibc on the host's
architecture. There is no Capstone or CHERI execution. The repository's usual
environment script may warn that the Capstone compiler build is absent. This
experiment uses the container's native compiler.

```sh
source capstone/tests/capstone-test-env.sh
python3 capstone/experiments/native-lifetime-survey/fetch.py
python3 capstone/experiments/native-lifetime-survey/prepare.py
python3 capstone/experiments/native-lifetime-survey/prepare_inputs.py
docker build -t qe-native-survey:ubuntu22 capstone/experiments/native-lifetime-survey
docker run -d --name qe-native-survey \
  -v "$PWD:/repo" -v "$CAPSTONE_TMP_ROOT:$CAPSTONE_TMP_ROOT" \
  -e CAPSTONE_TMP_ROOT="$CAPSTONE_TMP_ROOT" \
  qe-native-survey:ubuntu22 sleep infinity
docker exec qe-native-survey python3 /repo/capstone/experiments/native-lifetime-survey/test_observer.py
docker exec qe-native-survey useradd -m -u 1000 survey
docker exec qe-native-survey python3 -m pip install \
  --target "$CAPSTONE_TMP_ROOT/native-survey/python-deps" --no-deps pyperf==2.9.0
docker exec -w "$CAPSTONE_TMP_ROOT/native-survey/tools/memtier_benchmark-2.2.1" \
  qe-native-survey sh -c 'autoreconf -ivf && ./configure --disable-tls && make -j4'
```

For each of `sqlite postgresql cpython mruby perl ffmpeg wireshark memcached`,
build both variants with `build.py APPLICATION baseline` and
`build.py APPLICATION observed` inside the container. Perl builds on the
container's case-sensitive filesystem at `/var/tmp/native-survey-build`.
CPython's executable is `python.exe` on the macOS bind mount. The harness
resolves that distinction.

```sh
docker exec qe-native-survey python3 \
  /repo/capstone/experiments/native-lifetime-survey/campaign.py \
  --output "$CAPSTONE_TMP_ROOT/native-survey/campaign-new"
```

Output directories must be new. Every attempted process gets a result record,
including a failure. Raw output, input programs and build logs stay in scratch.
Only reviewed counters, input and output hashes, commands, environment metadata
and reproducible figure sources belong in the repository.

`test_observer.py` checks backing generations, bulk retirement, in-place and
failed resize, unknown backing, cross-instance gaps and a fatal overlap control.
The recorder serializes its metadata, never reads retired application storage
and exits with status 86 on inconsistent state or capacity exhaustion. Its
fixed tables consume memory and its locks can affect scheduling. The survey
therefore makes no timing claim and uses one application worker where possible.
Do not rebuild the shared recorder while measured processes are running.
