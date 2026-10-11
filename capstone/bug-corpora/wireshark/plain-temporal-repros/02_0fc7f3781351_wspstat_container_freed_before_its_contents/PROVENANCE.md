# 0fc7f3781351 — Don't free something before freeing some of its contents.

## The defect

The `register_tap_listener` failure path freed the `wspstat_t` container and then tore down the hash table it owns — `g_hash_table_foreach(sp->hash, ...)` and `g_hash_table_destroy(sp->hash)` both load `hash` out of the freed struct. The fix moves the container's free below the teardown.

## Upstream defect

- **Fix:** `0fc7f3781351`, *"Don't free something before freeing some of its contents."*, `ui/cli/tap-wspstat.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The whole path was replaced by a single finish call.

## The vulnerable code, quoted from the fix's parent

```c
		g_free(sp->pdu_stats);
		g_free(sp->filter);
		g_free(sp);
		g_hash_table_foreach( sp->hash, (GHFunc) wsp_free_hash_table, NULL ) ;
		g_hash_table_destroy( sp->hash );
```

## The fix

```c
		g_free(sp->pdu_stats);
		g_free(sp->filter);
		g_hash_table_foreach( sp->hash, (GHFunc) wsp_free_hash_table, NULL ) ;
		g_hash_table_destroy( sp->hash );
		g_free(sp);
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows. The object is a plain allocation because upstream's is
— this code is outside `epan`'s wmem scopes.

**Reduced:** no tap listener and no hash table. The struct's first word stands for the `hash` field and the teardown's load of it is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing. Nor upstream reachability of
the specific sequence chosen here.
