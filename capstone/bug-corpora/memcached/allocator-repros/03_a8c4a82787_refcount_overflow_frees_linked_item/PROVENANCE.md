# Provenance

**Tier: LITERAL-traceable allocator, reduced consumer.** `slabs.c` is
upstream's file at the port's 1.6.45 pin, compiled through
[`ports/memcached/allocators`](../../../../ports/memcached/allocators/README.md).
Every allocation decision is upstream's. Reduced: the item layer, to the two
functions that decide whether a chunk goes back to slabs, and the multiget,
to the references it takes.

- **CVE:** **CVE-2018-1000127** — *"memcached version prior to 1.4.37 contains
  an Integer Overflow vulnerability in `items.c:item_free()` that can result in
  data corruption and deadlocks due to items existing in hash table being
  reused from free list."* Published 2018-03-13.
- **Advisories:** Debian DSA-4218, Ubuntu USN-3601-1, Red Hat RHSA-2018:2290.
- **Fix:** `a8c4a82787b8b6c256d61bd5c42fb7f92d1bae00` — *"Don't overflow item
  refcount on get. Counts as a miss if the refcount is too high. ASCII
  multigets are the only time refcounts can be held for so long."*, 2017-05-23,
  `memcached.c` +11/−1. It adds `IT_REFCOUNT_LIMIT 60000` and `limited_get`.
- **Report:** issue [#271](https://github.com/memcached/memcached/issues/271),
  *"Memcached gets a dead loop in func assoc_find"*, with the gdb evidence:

      (gdb) p *it
      $30 = {next = 0x7f101a4fd7a0, prev = 0x7f0ffaeee700,
             h_next = 0x7f101a4fd7a0, ..., refcount = 1, ...}

  `h_next` is the item's own address: the chunk was freed while still linked,
  handed out again, and linked a second time into the same hash chain.
  `assoc_find` then walks it forever.
- **Live at the pin:** no. `IT_REFCOUNT_LIMIT 60000` is at `memcached.c:2185`
  in the pinned tree, reached through `limited_get` and `limited_get_locked`
  from `proto_parser.c`, `proto_bin.c` and `proto_text.c`.

## The defect

`refcount` is an `unsigned short` (`memcached.h:622`) and `refcount_incr` is
`++(it->refcount)` (`memcached.h:1040`) with no ceiling before the fix. An
ASCII multiget takes one reference per occurrence of a key and holds them all
until the response is written, so `get k k k ... k` with 65536 occurrences
adds 65536 to a count that can hold 65535. The stored count wraps to what it
was; the real number of holders does not. The next two releases take the
stored count to zero and `item_free` sends the chunk to `slabs_free` — with
65536 clients and the hash table still pointing at it.

## The reduction, and what is performed rather than asserted

The 65536 references are taken one at a time, in a loop, so the wrap happens
rather than being claimed; the case then checks that the count really does
read 2 again before it proceeds. `item_free` and `do_item_remove` keep their
arithmetic verbatim. Left out: the hash table, the LRU queues, the stats and
the locks, none of which takes part in the decision to free, and the response
writing, which only decides *when* the references are dropped.

One honest gap: after the premature free the fixture stops releasing. In the
server the remaining 65534 occurrences would each release in turn, and each
would be a further use-after-free. Modelling those would add nothing to the
first stale access and would make the labelled probe ambiguous.

## What the two arms show

`unit_reissued` differs between the arms, as in case 2: the premature free is
the defect, so the control cannot show the reuse it enables. The case checks
it before publishing its marker.

What the fixture does not show: the self-linked hash chain of issue #271. It
records the step that produces it — the chunk reaching `slabs_free` while
still referenced — and the first stale read that follows.

## One defect, three front ends, nine years

The same overflow was reachable through the binary protocol until `e3b7d33`
(2026-07-02), whose message says that with it *"you can cause a remote code
execution in the daemon"*; the meta protocol was hardened separately in
`bc080ab` (2020-02-27). This case is the ASCII one, because that is the one
the CVE and the advisories name.
