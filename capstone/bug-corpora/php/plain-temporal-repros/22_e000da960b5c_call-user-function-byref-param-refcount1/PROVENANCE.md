# CRASH-141 -- call-user-function-byref-param-refcount1-skips-separation-binds-userland-reference-into-engine-owned-oparray-literal

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, Zend/zend_object_handlers.c |
| site, as the source corpus records it | Zend/zend_object_handlers.c:75 (with Zend/zend_execute_API.c:748-753) |
| upstream fix | `e000da960b5c` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_object_handlers.c:75`, from the museum tarball this corpus pins:

```c
   73  	ZVAL_STRINGL(&__get_name, ZEND_GET_FUNC_NAME, sizeof(ZEND_GET_FUNC_NAME)-1, 0);
   74  
   75  	call_args[0] = &member;
   76  
   77  	/* go call the __get handler */
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | commit-history-reconstruction |
| origin_stream | history-5x-core3 |
| history_status | fixed-by-rewrite |
| confidence | medium |
| crashes on a pristine build | True |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-141.php`. Nothing
  was removed and nothing was added: it is not a reduction of ours.
- It runs on the whole CLI interpreter, so the defect is reached through PHP's own request
  lifecycle rather than through a driver around the allocator.
- There is no labelled read probe in a PHP script, so attribution rests on `fault_sites`, taken
  from the host-ASan report below and recorded before any Capstone or CheriBSD run.

## Measured at the pin (host ASan, 2026-10-28)

| build | outcome |
|---|---|
| detector patches (cache off, REAL_SIZE unrounded) | ASAN:heap-use-after-free |
| Zend's allocator as shipped | ASAN:heap-use-after-free |

Access stack of the first report: `zend_pzval_lock_func` (Zend/zend_execute.c:59), `zend_fetch_var_address` (Zend/zend_execute.c:798).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
