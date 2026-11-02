# CRASH-154 -- zval-dtor-array-arm-retains-is-array-type-and-live-ht-during-hash-destroy-reentrant-destruction-double-destroys-hashtable

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, Zend/zend_variables.c |
| site, as the source corpus records it | Zend/zend_variables.c:51-53 (missing type-clear before zend_hash_destroy); observed fault frame Zend/zend_execute_API.c:392 in _zval_ptr_dtor called from zend_hash.c:519 |
| upstream fix | `e88cdaa0143a` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_variables.c:51`, from the museum tarball this corpus pins:

```c
   49  				TSRMLS_FETCH();
   50  
   51  				if (zvalue->value.ht && (zvalue->value.ht != &EG(symbol_table))) {
   52  					zend_hash_destroy(zvalue->value.ht);
   53  					FREE_HASHTABLE(zvalue->value.ht);
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | commit-history-reconstruction |
| origin_stream | history-5x-roundE-nofault-reaudit |
| history_status | historical-known |
| confidence | high |
| crashes on a pristine build | True |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-154.php`. Nothing
  was removed and nothing was added: it is not a reduction of ours.
- It runs on the whole CLI interpreter, so the defect is reached through PHP's own request
  lifecycle rather than through a driver around the allocator.
- There is no labelled read probe in a PHP script, so attribution rests on `fault_sites`, taken
  from the host-ASan report below and recorded before any Capstone or CheriBSD run.

## Measured at the pin (host ASan, 2026-10-28)

| build | outcome |
|---|---|
| detector patches (cache off, REAL_SIZE unrounded) | ASAN:heap-use-after-free |
| Zend's allocator as shipped | ASAN:SEGV |

Access stack of the first report: `_zval_ptr_dtor` (Zend/zend_execute_API.c:392), `zend_hash_destroy` (Zend/zend_hash.c:519).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
