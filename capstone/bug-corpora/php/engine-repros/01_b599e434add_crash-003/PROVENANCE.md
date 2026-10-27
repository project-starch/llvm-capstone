# CRASH-003: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-003",
 "set": "core-166",
 "trigger": "corpus/CRASH-003.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "engine-core",
 "root_cause_id": "assign-selfref-container-dtor-uaf",
 "c_file_line": "Zend/zend_execute.c:669 (UAF deref of *variable_ptr_ptr); the freeing site is Zend/zend_execute.c:629",
 "upstream_fix_commit": "b599e434add",
 "crashes_on_pristine_build": "False",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "carried-from-4.0.2",
  "origin_stream": "carried-from-4.0.2",
  "history_status": "fixed-by-rewrite",
  "confidence": "high",
  "merged_members": "V5C-048"
 },
 "internal_id": "V5C-003"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-003.php` from that zip, byte for byte (sha256
`280d33b61faabd1941384c73eea4ffdd99fca23a79b0d7c28188dae1d14ab146`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `Zend/zend_execute.c:669` in
`zend_assign_to_variable`, stockalloc `OK`, pristine `OK` (rc 0).
Hence **asan-blind**: ASan reports it only with PHP's allocator patched (cache off, no 8-byte rounding); with zend_alloc.c as shipped the run reads OK. That is the selection, by measurement; which mechanism hides it, and which arm sees it, is what the arms record.

**Where our run and the corpus's record disagree:** nowhere -- ASan class, pristine outcome and the crash flag all agree.

**Upstream fix.** php-src `b599e434add`, the directory prefix `b599e434add`.
