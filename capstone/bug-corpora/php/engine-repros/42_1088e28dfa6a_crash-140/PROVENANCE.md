# CRASH-140: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-140",
 "set": "core-166",
 "trigger": "corpus/CRASH-140.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "engine-core",
 "root_cause_id": "debug-print-backtrace-arg-array-not-reset-per-frame-stale-freed-pointer-uaf-double-free",
 "c_file_line": "Zend/zend_builtin_functions.c:1453 (`zval *arg_array = NULL;` hoisted out of the `while (ptr)` loop at :1486); UAF read at :1433 via :1558; double free at :1559",
 "upstream_fix_commit": "1088e28dfa6a",
 "crashes_on_pristine_build": "False",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5x-core3",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-140"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-140.php` from that zip, byte for byte (sha256
`f5aa929414e46b2ac34e97c15fa975eb886cfbf6f2e504651408f10961a939c5`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `Zend/zend_builtin_functions.c:1433` in
`debug_print_backtrace_args`, stockalloc `OK`, pristine `OK` (rc 0).
Hence **asan-blind**: ASan reports it only with PHP's allocator patched (cache off, no 8-byte rounding); with zend_alloc.c as shipped the run reads OK, so emalloc's cache or rounding hides it from any detector at free().

**Where our run and the corpus's record disagree:** nowhere -- ASan class, pristine outcome and the crash flag all agree.

**Upstream fix.** php-src `1088e28dfa6a`, the directory prefix `1088e28dfa6a`.
