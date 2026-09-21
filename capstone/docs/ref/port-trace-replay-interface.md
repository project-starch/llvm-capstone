# Trace replay across the ports: what exists, and how much of it is one interface

*Audited 2026-09-22 on branch `audit/1-trace-replay-interface`, across the four
branches that carry port work. Complements the [port catalog](../../ports/README.md)
and the [shared trace tooling](../../ports/common/host/port_trace/README.md); this
document is the cross-port view neither of those gives.*

## The two answers

**Not all ports replay a trace.** Twelve port components exist across four branches.
**Six** replay an allocator trace; five of those go through the shared layer and one
(nginx) predates it and is invisible to it. Of the other six, three run a workload
(SQLite, MicroPython, musl), two run compiled-in corpus cases through a trace-shaped
protocol that carries no trace (memcached, APR pools), and one only establishes that a
library compiles (the APR census).

**The common interface is real on the host and a convention in C.** There is one
shared host layer — `port_trace` for the wire formats and `port_support` for staging
and verdicts — and all five shared-layout ports use it. Below it, the C side has the
*same shape* in every port but *no shared definition*: each port declares its own
header struct, its own event struct, its own magic, its own success sentinel, and
carries its own near-identical copy of the guest loader. Nothing enforces the
agreement; it is maintained by copying.

Two ports (memcached, APR pools) adopt the byte layout without adopting the meaning:
their trace carries a corpus case number, not allocator events. Same interface at the
byte level, different interface at the semantic level — see "The shape without the
substance" below.

## Matrix

| Component | Branch | Wire format (magic) | Where traces come from | Replay engine | In `port_trace` | `host/run-qemu.py` | Domain sentinel |
|---|---|---|---|---|---|---|---|
| `ffmpeg/buffer-pool` | integration | `ffmpeg.buffer-pool` v2, 128/128 B | recorder in-repo (`host/record.sh`, patched libavutil) | `src/shared/replay-engine.c` | yes | yes | 42044 |
| `postgres/memory-contexts` | integration | `postgres.a11` v1, 80/40 B | recorder in the paper repo (`experiments/a11/postgres`), per `a11trace.h`; fixtures generated in-repo | `src/shared/replay-engine.c` | yes | yes | (own loader protocol) |
| `cpython/pymalloc` | integration | `cpython.pymalloc` v1, 96/32 B | recorder in-repo (`host/record.py`, patched CPython) | `src/shared/replay.c` | yes | yes | 42049 |
| `whisper/ggml-context` | integration | `whisper.ggml-context` v1, 128/48 B | recorder in-repo (`host/record.py`, patched whisper.cpp) | `src/shared/replay.c` | yes | yes | 42050 |
| `wireshark/wmem` | `ports/17` | `wireshark.wmem` v1, 128/48 B | **no recorder** — traces synthesised by tests/fixtures | `src/shared/replay.c` | yes | yes | 42060 |
| `nginx` | integration | `NGXTRACE` v1, 80/40 B | recorder in the paper repo (`experiments/a11/nginx`), per `ngxtrace.h` | `port/ngx_replay.c` | **no** | no (`run-nginx-replay.sh`) | n/a |
| `memcached/allocators` | `ports/16` | `MCSLABS1`, 96/32 B — **case selector, not events** | n/a | corpus macro only | no | no | 42047 |
| `apr/pools` | `ports/18` (also on 16, 17) | `APRPOOL1`, 96/32 B — **case selector, not events** | n/a | corpus macro only | no | no | 42046 |
| `apr` (census) | integration | — | — | — | — | — | — |
| `sqlite` | integration | — (workloads: speedtest1, sqllogictest) | — | — | — | — | — |
| `micropython` | integration | — (interpreter/GC workloads) | — | — | — | — | — |
| `musl-capstone` | integration | — (libc functional tests) | — | — | — | — | — |

## What is genuinely shared

* **`common/host/port_trace/`** — one reader API over all registered formats, with a
  `Format` base class (`model.py:31`) and an explicit registry
  (`formats/__init__.py:8`). `validate` / `summary` / `inspect`, plus the
  `--replay-input` contract that refuses a capture still carrying measured results.
* **`common/host/port_support.py`** — `stage_run`, `run_guest` (serialized QEMU),
  `record_trace`, and `write_replay_verdict` (`:82`) writing
  `capstone.replay-verdict/v1`.
* **`common/cmake/Port.cmake`** and the three toolchains, with the same preset names
  (`native`, `capstone-domain`, `linux-guest`, `cheribsd`) in every shared-layout port.
* **`common/host/cheribsd/components.json`** — one registry, one `build.py`/`run.py`
  for the purecap arm.
* **The domain ABI**: four shared regions arrive in a fixed order — report, metadata,
  trace, payload — then one `call_dom`; the payload grant is linear, the rest are not.
* **The runner CLI**: `run-qemu.py <trace> <output> --protection {spatial,sublet}`.

## What only looks shared

* **No common C header.** Every port declares its own `struct <p>_header` / `struct
  <p>_event` and its own `<p>_replay()` — `pym_replay` (`port.h:50`), `wg_replay`
  (`:37`), `wm_replay` (`:35`), `aprp_replay` (`:65`), `mcp_replay` (`:86`),
  `ff2_replay_run` (`replay-engine.h:9`), `replay_run` (postgres, `:21`). The headers
  say so in prose ("the pymalloc port's shape with APR's names"), which is exactly the
  kind of agreement a compiler cannot check.
* **The guest loader is copied per port.** After normalising identifiers and stripping
  comments, `src/linux-guest/domain-loader.c` is 81–95 % line-identical across
  pymalloc, ggml, wmem, memcached and APR (ggml↔wmem: 94.6 %; memcached↔APR: 89.1 %).
  FFmpeg's is the same design in a different house style; PostgreSQL's is a different
  program (221 vs ~57 normalised lines) with its own argv protocol
  (`--report-file`, `--scratch`) and its own success marker.
* **Six success sentinels, hard-coded twice each** — 42044 / 42046 / 42047 / 42049 /
  42050 / 42060, once in `src/capstone-domain/entry.c` and once in the loader that
  checks it. A seventh port means picking a seventh number by hand.
* **Test presets diverge.** FFmpeg and PostgreSQL register `qemu-replay` and
  `qemu-security` test presets; pymalloc and ggml register only `native`. There is no
  single command that runs "the replay test" for all of them.
* **Two replay concepts share one symbol name.** For pymalloc the corpus *also*
  supplies `pym_replay` (`bug-corpora/cpython/pymalloc-repros/shared/corpus.h:148`),
  overriding the trace interpreter with a defect case. So `<p>_replay` means "execute
  the trace" in one build and "execute case N" in another.

## The shape without the substance

memcached and APR pools adopt the 12-word header, the 4-word event record and the
`<p>_replay(input, out)` seam — and then never carry a trace. Their corpus macro
(`bug-corpora/memcached/allocator-repros/shared/corpus.h:137`,
`bug-corpora/httpd/apr-pool-repros/shared/corpus.h:127`) requires `input->count == 1`
and `e->id == <case number>`: the trace region is a one-record case selector.

This is consistent — the file says it is the pymalloc shape with different names — but
it means "follows the common interface" is true of the bytes and false of the
semantics. A tool that inferred "has a trace region ⇒ can replay a trace" would be
wrong on two of the seven ports that declare the seam. Neither has a `replay` binary
at all: their built trees hold `allocator-example`, `domain-loader` and (memcached)
`defects.dom`, and no trace reader.

## Verified today, with controls

Everything below was run on 2026-09-22 against the pre-existing native builds under
`/tmp/capstone/*/build/native/`.

| Check | Result |
|---|---|
| `common/tests` unittest suite (30 tests) | OK (1 skipped) |
| `port_trace validate` on a hand-built `cpython.pymalloc` trace | accepted |
| …six malformed variants (truncated, unknown op, no end, end not last, report fields set, free with payload) | **6/6 refused**, each with its own error |
| `bin/replay` (pymalloc) on that trace / on the truncated and unknown-op ones | `completed=4` / exit 3 / `failed=310` |
| `cpython/pymalloc/tests/native/test-replay.py` (incl. double-free, live-at-end, invalid-id controls) | pass |
| `whisper/ggml-context/tests/native/test-replay.py` | pass — 2607 events, malformed + exhaustion controls |
| `wireshark/wmem` native replay test (branch 17) | pass — 1661 events, malformed controls |
| `postgres` `check-native.py` on both committed fixtures | pass — replay reproduces the recorded backend's block counts (72/68/1/12 and 49/48/0/7) exactly |
| `ffmpeg` `bin/replay` on a real 720p/300 s recording | `status=0`, 630 309 events |
| `port_trace validate --replay-input` on that recording's `commands.bin` | accepted (sha256 `722825fc…`) |
| …on its `recorded.bin` (measured outcomes still present) | refused: "strip measured outcomes before replaying FFmpeg commands" |

**Instrument note.** `ctest --test-dir /tmp/capstone/<port>/build/native` fails
instantly on all four ports with `can't open file
'~/llvm-capstone-cpython/…'`: those build trees were configured from
worktrees that no longer exist, so CTest still points at deleted source paths. That is
a stale-configuration fault, not a port failure — the same tests pass when invoked
directly, as above. Reconfigure before reading any `ctest` result from those trees.

**Not verified here:** no domain or QEMU replay was run, so this says nothing about
the `capstone-domain` or `linux-guest` arms beyond the fact that their sources build
the protocol described above.

## Gaps, in the order they cost something

1. **nginx is outside the shared layer.** `ngxtrace.h` is deliberately "the same shape
   as `a11trace.h`", 80/40 bytes, and the two are read by two unrelated readers. It is
   the cheapest adapter to add (`formats/nginx.py`), and until it exists nginx traces
   get none of the framing, replay-input or provenance checks the other five get.
2. **`wireshark/wmem` has no recorder.** Its traces are written by its own tests, so
   "real allocator, model consumer" holds for the allocator but the *request
   distribution* is invented. The other four record from the real program.
3. **memcached and APR pools have the protocol but no trace path.** Adding
   `src/shared/replay.c` plus a format adapter to each is mechanical; the header,
   loader and domain entry already exist.
4. **The C protocol has no single definition.** A `common/include/port-protocol.h`
   carrying the header/event layout, the region order and a sentinel range, plus one
   parameterised `domain-loader.c`, would remove five near-copies and make the
   agreement checkable. PostgreSQL would stay outside it; it has real reasons to.
5. **No cross-port replay entry point.** `ctest --preset qemu-replay` exists for two
   ports out of five.

## Integration hazard

Branches `ports/16-memcached-poisoncap` and `ports/17-wireshark-wmem` both append a
sixth entry to `common/host/cheribsd/components.json` at the same position, and both
extend `common/host/cheribsd/README.md`. They will conflict textually; the resolution
is a union, not a choice. Otherwise all three unmerged branches touch `common/` only
additively — `ports/17` adds `formats/wmem.py`, the registry line, a README row and
tests, and nothing else in the shared layer differs from the integration branch.
