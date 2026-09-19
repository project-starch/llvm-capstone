# Shared allocator trace tooling

`port_trace` reads the four existing allocator trace formats through one host
API. It does not change their bytes, execute allocator operations or reinterpret
an allocator's ownership rules. Python 3.11+ and the standard library suffice.

From the repository root:

```sh
export PYTHONPATH="$PWD/capstone/ports/common/host${PYTHONPATH:+:$PYTHONPATH}"
python3 -m port_trace validate /path/to/trace.bin --replay-input
python3 -m port_trace summary /path/to/trace.bin
python3 -m port_trace inspect /path/to/trace.bin --limit 10
```

The magic selects the adapter. `--format postgres.a11`, for example, also
requires that particular format. All commands validate the **entire** file
before writing success JSON; `--limit` only bounds the inspection preview.
Errors produce JSON on stderr and exit 1. Argument errors exit 2. The input is
read-only, and no build or guest is needed.

## Formats and boundaries

| Format ID | Wire version | Header / record bytes | Semantics retained by the adapter |
|---|---|---|---|
| `ffmpeg.buffer-pool` | 2 | 128 / 128 | Calls, nested callbacks, distinct lease/backing identities and measured outcomes |
| `postgres.a11` | 1 | 80 / 40 | Context kinds/hierarchy, bulk reset/delete, old/new realloc identities and process-prefix references |
| `cpython.pymalloc` | 1 | 96 / 32 | alloc/calloc/realloc/free; a replay-table ID can persist through realloc |
| `whisper.ggml-context` | 1 | 128 / 48 | Separate descriptor/buffer identities, buffer ownership and allocation epochs |

The wire definitions remain in each port's C headers. The Python adapters
implement their current versions explicitly; an incompatible future layout
needs a distinct version/magic and an adapter. Unknown formats, versions or
operations are refused. These readers cover the current little-endian formats.
They do not assume native Python/C structure alignment.

`validate` establishes framing and schema: header/record sizes and counts,
known operations, format-specific fixed fields, and exactly one terminal
record at physical EOF. It checks the bytes read for concurrent file changes.
It **does not** establish allocator correctness, valid object lifetimes,
replay resource capacity, successful capture instrumentation, or replay success.
Those remain the port's recorder controls and native/domain oracles.

`--replay-input` adds the existing replay-input contract:

- FFmpeg measured outcome/report fields must have been stripped by its
  `host/memory/analyze.py commands` step. They cannot become allocator choices.
- PostgreSQL traces must already be flattened: unresolved parent-process
  prefixes are preserved by inspection but refused for standalone replay.
- CPython input headers must not contain result fields.
- Whisper input headers must not contain result fields and must declare the
  supported recorded native object-header geometry.

A capture can be structurally inspectable yet unsuitable for replay. Failed
result headers remain visible during inspection; `complete: true` only means
that the **trace file** completed structural validation.

## Python API

```python
from port_trace import TraceReader, inspect_trace

summary = inspect_trace(path, expected_format="cpython.pymalloc", replay=True)
with TraceReader(path) as trace:
    for event in trace.events():
        consume(event.index, event.operation, event.role, event.fields)
    digest = trace.sha256
```

The reader is single-pass. Exhausting `events()` performs the final checks;
closing it early does not establish validity, and `sha256` is unavailable
until completion. Each event preserves its port-specific field names and
values. `index` is zero-based. Roles are `command`, `callback`, `observation`
and `end`. For FFmpeg the observation role describes an outcome position even
when its measured fields have been zeroed for a command stream.

The common summary has schema `capstone.trace-inspection/v1`. Its `trace`
object records format, wire version, byte order, byte/record counts and the
SHA-256 of exactly the bytes inspected. `header` retains format-specific
metadata; PostgreSQL phase bytes are hex encoded. `operations` and `roles`
count wire records **including the terminal record**. The counts are not
normalized allocation or lifetime metrics and should not be compared as such.
`recording_provenance` is null because these binary formats do not establish
which source revision or workload produced them. Join existing recorder
manifests by trace checksum; never substitute the current checkout's pin for
an unknown historical recording revision.

## Port integration and results

The four normal QEMU replay launchers validate the staged input before guest
execution and retain `trace.json` beside the existing run manifest. Failed
validation retains the input and a `capstone.trace-error/v1` sidecar; the VM
is not started. Dedicated PostgreSQL lifetime fixtures do not interpret their
input as a trace, and FFmpeg's explicit nonzero `--expected-status` controls
retain their deliberate invalid-input path. These paths skip trace preflight.

The launchers use `port_support.write_replay_verdict` for
`capstone.replay-verdict/v1`: existing `passed`, `runner_exit` and port-specific
fields remain, with an optional `trace` identity from the validated sidecar.
No trace association is invented for non-trace fixtures or deliberate rejection
controls. A failed trace sidecar cannot be promoted to a passing result.
The port still decides its own oracle: a trace validator never decides whether
an expected capability fault occurred at the right access instruction.

FFmpeg and PostgreSQL memory analysis use the shared wire readers while keeping
their existing independent analysis. CPython and Whisper validate completed
captures before promoting `.partial` files. CPython imports the inspector only
after capture ends, so its imports do not perturb the recorded workload.

## Adding an allocator

1. Define and document its wire format, completion marker and capture scope.
2. Add an adapter under `formats/`, register it explicitly, and assign a unique
   format ID and distinguishable wire magic/version.
3. Preserve domain-specific identities and operations. Do not turn borrowed
   descriptor release, pool return or bulk context deletion into generic free.
4. Test valid bytes and corruption: truncation, unknown operations/versions,
   premature/missing footer and inappropriate replay inputs. Check a real
   recorder artifact against an independent native replay or reference reader.
5. Call `record_trace` on the staged input and `write_replay_verdict` only after
   the port-specific oracle has decided the result.

This first shared layer intentionally has no universal allocator API or writer.
Normalized lifetime/memory views can later be added as separate adapter outputs
with explicit units, identity generations, ownership and unknown-value handling.
Historical captures and their original hashes remain the reference artifacts.

Tests are included in the common port-support CTest. Direct invocation:

```sh
source capstone/tests/capstone-test-env.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s capstone/ports/common/tests -p 'test_*.py'
```
