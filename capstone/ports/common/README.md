# Shared port build and run support

This directory supplies infrastructure to the CMake component ports listed in
the [port catalog](../README.md). It is not an application or allocator.

| Path | Responsibility |
|---|---|
| `cmake/Port.cmake` | Platform selection, external-build guard and support test registration |
| `cmake/Workspace.cmake` | Repository discovery and environment/cache path defaults |
| `cmake/Upstream.cmake` | Shared source preparation helpers |
| `cmake/toolchains/` | Capstone domain and guest Linux compilation |
| `host/port_support.py` | Verified source preparation, run staging, provenance and serialized guest execution |
| `tests/` | Source, staging, runner and oracle regression tests |

Component roots include `../../common/cmake/Port.cmake` before `project()` and
call `port_check_platform()` afterward. They provide their own source manifest,
patch sequence, replay targets and workload oracles. Keep source archives and
generated variants under `$CAPSTONE_TMP_ROOT`, normally `/tmp/capstone`.

Shared capability instructions and Sublet lifetime helpers live in
[`capstone/runtime`](../../runtime/CMakeLists.txt), exposed as
`Capstone::Runtime`. Do not add allocator policy here or duplicate those
headers in a component. Optional fault recovery and emulator prerequisites are
tracked in the [integration plan](../../docs/plans/port-stack-integration.md).

Run the support checks from the repository root:

```sh
source capstone/tests/capstone-test-env.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s capstone/ports/common/tests -p 'test_*.py'
```
