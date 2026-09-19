# Provenance

**Tier: LITERAL-traceable allocator and predicate, reduced consumer.**

- **Fix:** `1ee3c984b9` — *"avcodec/snow: ensure current_picture is writable before modifying its data"*, 2020-06-09.
- **File:** `libavcodec/snow.c` and the encoder path.
- **CVE:** `NO VERIFIED CVE`.
- **Live at the pin:** no.
- **Shape:** identical call sequence to the five filter cases; the second holder is the encoder's coded frame rather than a downstream consumer.
