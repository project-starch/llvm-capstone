# Perl packed-pointer provenance

This adapter targets the SPEC CPU2017 `500.perlbench_r` Perl 5.22.1 source
with the existing thirteen-file `capstone` port. It is a new source alternative,
`capstone_provenance_v2`; it does not replace the earlier package or its frozen
objects. `baseline.json` pins all twenty files in the union of the old port and
this adapter. Other Perl versions and source baselines are unsupported.

Perl's `pack("p", ...)` and `pack("P", ...)` serialize pointer representations.
Byte reversal, unaligned placement, and UTF-8 conversion cannot preserve a
hardware capability tag. This adapter retains the original typed capability in
private metadata owned by the packed scalar. `unpack` requires a complete field,
matching representation, and this retained authority. It never reconstructs a
capability from an integer, an address lookup, or byte equality alone. A complete
pointer carrying a real hardware tag is also accepted; host tests cannot exercise
that fallback.

The supported in-process transfers include whole scalar assignment, array/hash
storage, function returns, concatenation and append, join, repetition, complete
pointer fields in substrings and substring replacement, `a/A/Z` packing and
unpacking, and Perl's Latin-1/UTF-8 upgrade and downgrade. Offsets follow the
actual encoded bytes. The `p` and `P` forms may share retained authority; normal
template lengths and endianness still apply. The implementation includes explicit
core transfer points for substring aliases and already evaluated magical values.
It does not claim support for every Perl string operation or arbitrary extension
callback.

Mutation revokes affected authority before obtaining writable bytes. Same-byte
writes, partial-field reconstruction, substitution, transliteration, numeric
conversion, and pack cursor truncation cannot restore it by later reproducing
the bytes. Hex, file, pipe, and network byte round trips do not carry metadata.
`utf8::encode` and `utf8::decode` are byte serialization boundaries and discard it.
A non-benign magic getter revokes the previous value's authority before calling
user code; only a fresh, tracked value transfer can supply new metadata. Ordinary
callback evaluation counts remain those of Perl.

Each scalar owns a separate metadata list, including copy-on-write siblings.
Temporary snapshots use Perl's mortal lifetime so exceptions release them. This
does **not** extend the pointed-to string's lifetime: callers retain Perl's
existing responsibility to keep the pointee alive and stable. Native extensions
must obey Perl's mutable-buffer API and pointer-lifetime rules; this is not a
monitor for hostile XS code retaining and writing arbitrary native aliases.
Interpreter thread cloning is outside this port's supported configuration.

The adapter changes no benchmark inputs or comparators and requires no new ISA,
kernel, or SDK behavior. Licensed sources and generated source-alternative
packages belong outside the repository, under `CAPSTONE_TMP_ROOT`.

To reproduce, source `capstone/tests/capstone-test-env.sh`, copy the pinned
existing port source to a fresh external directory, and run:

```sh
python3 capstone/ports/perl/provenance/apply.py "$SOURCE_TREE" \
  --record "$RECORD_DIR/source-audit.json"
python3 capstone/ports/perl/provenance/build.py --tree "$SOURCE_TREE" \
  --log "$ORIGINAL_BUILD_LOG" --cc "$SDK/capstone-cc" \
  --record "$RECORD_DIR/build" --jobs 6
```

The source audit records the pinned before/after hashes and authoring-tool hashes.
The build replays every original compile command, producing a fresh inventory of
all 69 objects plus the linked interpreter. Native flags are unchanged. For a
host diagnostic only, `--host-test --sanitize --cc gcc` selects the SPEC Linux
x86-64 configuration, adds the provenance test define, `-fno-strict-aliasing`,
`-fwrapv`, and ASan/LSan. A host run does not qualify hardware tags or either
native syscall ABI.

The host allocation regression relinks those existing ASan objects with a
standalone `malloc` wrapper; production code has no fault-injection hook:

```sh
python3 capstone/ports/perl/provenance/run-allocation-failure.py \
  --source-tree "$HOST_SOURCE_TREE" --commands "$HOST_RECORD_DIR/commands.json" \
  --lib "$ORIGINAL_INPUT_STAGE/lib" --output "$RECORD_DIR/allocation-failure"
```

It checks both real allocator failure through Perl's fatal OOM cleanup and a
test-only recoverable exception at the same boundary. Metadata replacement must
publish the surviving list before allocation, so cleanup never sees freed nodes.
The latter probe also checks that surviving fields retain authority while the
failed replacement loses it. See [revision07 focused qualification](qualification-07.json)
for the frozen source, package, object inventories, native gates, and host
diagnostic evidence. The complete original revision07 workload and official
comparison are still pending. The preserved [revision06 record](qualification.json)
covers its earlier focused checks; its subsequent complete original workload
failed only `t/op/sort.t` (315 of 316 test files passed).

`scalar-lifetime.pl` checks the core error path for sort callbacks that free
their input elements. Perl 5.22 retains marked scalar headers for that error
path; metadata lookup must reject the freed marker before reading a scalar
body. The fixture also checks recovery and live packed values after sorting.
Run it with the same interpreter and library path as `callback-controls.pl`.

For packaging, copy the twenty final replacement files listed in `baseline.json`
to an external kit's `srcalt/500.perlbench_r/files`, add an external package
`README`, and invoke `tools/make-srcalt.sh` from the SPEC port kit with fifth
argument `capstone_provenance_v2`. Use the pristine original SPEC tree for its
diffs. Keep the generated archive, checksum, complete source inventory, and fresh
object inventory together; never relabel the old frozen objects. Qualification
uses the focused fixtures in both direct and exchange modes and the unchanged
original `t/op/pack.t`, followed by the original full benchmark commands and
comparators.

## Build and run the explicit v2 package

This recipe starts from a licensed SPEC archive and the frozen source package;
it does not need the qualification machine's build directories or `make.out`.
The authoring commands above remain useful when creating a new package.
`srcalt-selection.json` pins the revision07 v2 archive with the focused checks
above and leaves the historical
kit manifest intact. Supply a private package directory containing `packages/`
with that archive. Only selected benchmarks' packages are required; the default
full benchmark selection still installs all three source alternatives.

Use an explicit native SDK whose artifacts match the desired qualification arm.
The new label below avoids the historical heap-comparison arms; the source-only
overlay does not select a compiler, kernel, launcher, or runtime policy.

```sh
source capstone/tests/capstone-test-env.sh
KIT="$PWD/capstone/ports/spec-cpu2017"
PORT="$PWD/capstone/ports/perl/provenance"
HELPERS="$PWD/capstone/runtime/virtual/tests/native-task"
WORK="$CAPSTONE_TMP_ROOT/perl-v2-reproduction"
mkdir "$WORK"                         # must be a fresh directory
# Set absolute paths: SPEC_ARCHIVE, PRIVATE_PACKAGES, SDK.
python3 "$KIT/tools/spec-extract.py" "$SPEC_ARCHIVE" "$WORK/tree" ALL
SPEC_ROOT="$WORK/tree/SPEC17"
CAPSTONE_SPEC_PACKAGES="$PRIVATE_PACKAGES" \
CAPSTONE_SPEC_SOURCE_SELECTION="$PORT/srcalt-selection.json" \
BENCHMARKS=500.perlbench_r \
bash "$KIT/tools/build.sh" "$KIT" "$SPEC_ROOT" "$SDK" "$WORK/build" native_provenance_v2

(
  cd "$SPEC_ROOT"
  source ./shrc
  runcpu --config=capstone-virtual --action=setup --nobuild --size=test --tune=base \
    --define sdk="$SDK" --define kit="$KIT" \
    --define arm_label=native_provenance_v2 \
    --define perl_srcalt=capstone_provenance_v2 500.perlbench_r
  python3 "$HELPERS/prepare-spec-tests.py" \
    --spec "$SPEC_ROOT" --builds "$SPEC_ROOT/benchspec/CPU" --sdk "$SDK/capstone-cc" \
    --frozen "$WORK/build/frozen-native_provenance_v2.json" --require-frozen-objects \
    --label native_provenance_v2 --benchmarks 500.perlbench_r \
    --run-roots "$SPEC_ROOT/benchspec/CPU" --out "$WORK/prepared"
  python3 "$HELPERS/setup-spec-validation.py" --spec "$SPEC_ROOT" \
    --prepared "$WORK/prepared" --cc "$SDK/capstone-cc" \
    --ldecod "$WORK/build/ldecod-host" --out "$WORK/official"
)

# Set QEMU, NATIVE_IMAGES, FIRMWARE, and HUSH to the qualified native platform
# and pinned capability shell. This uses the shared QEMU lock.
python3 "$HELPERS/run-spec-tests.py" --prepared "$WORK/prepared" \
  --comparators "$WORK/official/comparators.json" --benchmarks 500.perlbench_r \
  --qemu "$QEMU" --images "$NATIVE_IMAGES" --firmware "$FIRMWARE" \
  --shell "$HUSH" --null-device --local-tmp --timeout 10800 --out "$WORK/execution"
python3 "$HELPERS/validate-spec-tests.py" --spec "$SPEC_ROOT" \
  --results "$WORK/execution/runs" --comparators "$WORK/official/comparators.json" \
  --out "$WORK/validation.json"
```

The build refuses existing build directories, verifies the twenty selected
source replacements and 69 fresh objects, and records SDK inputs before and
after compilation. Preparation validates source and object hashes before linking;
the following official setup independently compares the original argument lists.
Inputs and comparators are unchanged. This is a complete Perl-only reproduction,
not a new ten-entry or performance claim. The existing full-set preparation still
requires the official generated x264 input and its host-decoder provenance, as
described in the [native qualification guide](../../../runtime/virtual/tests/native-task/README.md#full-spec-test-input-qualification).

If regenerating the source archive, its packaging metadata can change its hash.
Keep the qualified overlay unchanged: make a separate external overlay for the
new archive path/hash while retaining the replacement-file hashes and base-lock
pin. The build records that distinct package identity; prior qualification does
not automatically transfer to it.
