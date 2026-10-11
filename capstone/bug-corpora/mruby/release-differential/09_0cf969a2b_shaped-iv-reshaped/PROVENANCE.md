# shaped-iv-reshaped

Upstream fix `0cf969a2b` (2026-09-18), *variable.c: keep a growing or de-shaping object's IV walk off freed memory*, which is in master and postdates the
4.0.0-rc2 pin. The defect is therefore live in the release this project ports.

**How it was found.** The release sweep extracted the Ruby test that upstream
added with this fix onto a harness and ran it at the pin and at master
(`ledger.json`, `probe/`). `trigger.rb` is that extraction, unmodified.

**Host oracle at the pin.** ASan: heap-use-after-free; with `MRB_HEAP_PAGE_SIZE=1`, which puts
each GC object slot in its own malloc block: heap-use-after-free. On master: no report.

**What the arms read.** See `case.json`. The three Capstone arms are one image
each of the same source and differ only in the heap they link, so a difference
between them is the heap's protection and nothing else.
