# CRASH-012 -- globals-hashtable-destroy-efree-no-symtable-guard

## The upstream defect

| | |
|---|---|
| class | CWE-590 invalid-free (memory-corruption) |
| where | pure-ext, ext/standard/array.c |
| site, as the source corpus records it | ext/standard/array.c:2060 |
| upstream fix | `1d33a3e95e4e` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`ext/standard/array.c:2060`, from the museum tarball this corpus pins:

```c
 2058  	/* Replace input array's hashtable with the new one */
 2059  	zend_hash_destroy(Z_ARRVAL_P(array));
 2060  	efree(Z_ARRVAL_P(array));
 2061  	Z_ARRVAL_P(array) = new_hash;
 2062  	
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | carried-from-4.0.2 |
| origin_stream | carried-from-4.0.2 |
| history_status | historical-known |
| confidence | high |
| crashes on a pristine build | True |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-012.php`. Nothing
  was removed and nothing was added: it is not a reduction of ours.
- It runs on the whole CLI interpreter, so the defect is reached through PHP's own request
  lifecycle rather than through a driver around the allocator.
- There is no labelled read probe in a PHP script, so attribution rests on `fault_sites`, taken
  from the host-ASan report below and recorded before any Capstone or CheriBSD run.

## Measured at the pin (host ASan, 2026-10-28)

| build | outcome |
|---|---|
| detector patches (cache off, REAL_SIZE unrounded) | ASAN:attempting free on address |
| Zend's allocator as shipped | ASAN:attempting free on address |

Access stack of the first report: `_efree` (Zend/zend_alloc.c:287), `zif_array_splice` (ext/standard/array.c:2060).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
