# Compact launch storage experiment

`CAPSTONE_RUNTIME_COMPACT_LAUNCH=ON` shares the argument and environment
pointer-vector allocation using the launch ABI's combined 1,024-string limit.
It reserves both NULL terminators and page-aligns the vector and string
storage. Count validation precedes formation of the environment cursor;
the existing unpacker validates the complete block before publishing pointers.
The launch wire format and 128-bit capability ABI are unchanged.

The switch defaults to OFF, requires the virtual profile, and is recorded as
`compact_launch` in `sdk.json`. It does not change eager pinning of the
supervisor's meta pages. Signal initialization touches all four of those
pages, and asynchronous host publication needs admitted backing.

Run the host semantic checks from the repository after sourcing
`capstone/tests/capstone-test-env.sh`:

```sh
python3 capstone/runtime/tests/application/run-launch-layout.py --work "$out"
```

`out` must name a fresh directory. Both layouts must pass maximum argument
count, maximum environment count, mixed count, invalid counts, invalid string
offset and truncated-buffer cases. The overlapping-vector control must fail.
These host checks do not exercise capability bounds or guest execution;
qualification also requires the virtual application gates and completed
benchmark validation with images built from the selected SDK.
