# CRASH-070: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-070",
 "set": "core-166",
 "trigger": "corpus/CRASH-070.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "engine-core",
 "root_cause_id": "begin-class-member-fcall-indirect-static-last_op-stale-after-opcode-array-erealloc-uaf",
 "c_file_line": "Zend/zend_compile.c:1321",
 "upstream_fix_commit": "d8efe626b5359a76c5d631fd67316aef7997db14",
 "crashes_on_pristine_build": "False",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "fuzzing",
  "origin_stream": "fuzz",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-070"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-070.php` from that zip, byte for byte (sha256
`6ee89dc269449ffd1cd3fdb129fbe69fa75950ee42650041e0bdd2798f1d6fb6`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `Zend/zend_compile.c:1321` in
`zend_do_begin_class_member_function_call`, stockalloc `EXIT:255`, pristine `EXIT:255` (rc 255).
Hence **temporal-asan-visible**: a CWE-416 that ASan reports with zend_alloc.c as shipped: efree reached free(), so a detector at the libc boundary sees it; the control population.

**Where our run and the corpus's record disagree:** nowhere -- ASan class, pristine outcome and the crash flag all agree.

**Upstream fix.** php-src `d8efe626b5359a76c5d631fd67316aef7997db14`, the directory prefix `d8efe626b535`.
