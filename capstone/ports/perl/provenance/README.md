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
failed replacement loses it. See `qualification.json` for the frozen source,
package, object inventories, native gates, and host diagnostic evidence.

For packaging, copy the twenty final replacement files listed in `baseline.json`
to an external kit's `srcalt/500.perlbench_r/files`, add an external package
`README`, and invoke `tools/make-srcalt.sh` from the SPEC port kit with fifth
argument `capstone_provenance_v2`. Use the pristine original SPEC tree for its
diffs. Keep the generated archive, checksum, complete source inventory, and fresh
object inventory together; never relabel the old frozen objects. Qualification
uses the focused fixtures in both direct and exchange modes and the unchanged
original `t/op/pack.t`, followed by the original full benchmark commands and
comparators.
