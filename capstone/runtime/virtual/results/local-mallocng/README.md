# Local mallocng qualification

The allocator is compiled for Capstone and runs with the application. The
launcher supplies actual Linux VM/syscall slow paths. Upstream allocation
policy is retained; capability representation and lifetime hooks are ported.
See [the implementation contract](../../mallocng.md).

- Allocator: 19/19, including 200,000 lifetimes, 70,000 live objects, exact-site
  spatial/temporal denials, nonlinear/linear tagged realloc and real Linux
  mremap grow/shrink. The directed local-loop oracle detects an inserted WAIT.
- Runtime: 42/42, including linear loans, same-mm threads, VM protections,
  stale/Bounds denials, SQLite persistence, mruby and Perl's 17-section smoke.
  Its two long sparse recycling cases were explicitly omitted.
- Pthreads: 10/10, including concurrent mapping realloc in two threads.
- Other normal ports: 16/16 for CPython, PostgreSQL, FFmpeg and tshark,
  compared with independent native workload outputs.
- Processor: 60 M1, 69 virtual-runtime, 69 U-access and four exact-bounds
  checks, on the same QEMU binary.
- Policy: 22 focused source comparisons; three deliberate changes are rejected.
  Virtual musl compiles 1,361/1,361 C files. The physical survey retains its
  six expected mallocng-layout failures.

The initial memcached corpus did not trigger fixture 17: immediate address
reuse is not mallocng policy. A bounded reissue search changes the fixture's
setup, retaining its original temporal-fault prediction. Native reference
transcripts are reused from a passing gate with matching binary and harness
hashes because host socket creation is restricted in this environment.
The [final memcached record](memcached.json) passes 73/73 checks, including
all 20 applicable fixtures on four workers.

The [manifest](qualification.json) records build scope, source hashes and
remaining limits. The compiler and Linux/firmware artifacts are reused;
this is not a fresh full LLVM build or RTL validation. Historical buddy-heap
and native-heap results do not qualify this allocator.
