# CRASH-022: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-022",
 "set": "core-166",
 "trigger": "corpus/CRASH-022.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "pure-ext",
 "root_cause_id": "user-sort-comparator-frees-live-bucket-uaf",
 "c_file_line": "ext/standard/array.c:546",
 "upstream_fix_commit": "18a0d46a1b44",
 "crashes_on_pristine_build": "False",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "carried-from-4.0.2",
  "origin_stream": "carried-from-4.0.2",
  "history_status": "historical-known",
  "confidence": "medium",
  "merged_members": "V5C-185"
 },
 "internal_id": "V5C-022"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-022.php` from that zip, byte for byte (sha256
`7b5c327acd23b4f0d464e0bc60faef160330c3f6c6fb03cf6c674abe20c4e3a4`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `ext/standard/array.c:546` in
`array_user_compare`, stockalloc `OK`, pristine `OK` (rc 0).
Hence **asan-blind**: ASan reports it only with PHP's allocator patched (cache off, no 8-byte rounding); with zend_alloc.c as shipped the run reads OK. That is the selection, by measurement; which mechanism hides it, and which arm sees it, is what the arms record.

**Where our run and the corpus's record disagree:** nowhere -- ASan class, pristine outcome and the crash flag all agree.

**Upstream fix.** php-src `18a0d46a1b44`, the directory prefix `18a0d46a1b44`.
