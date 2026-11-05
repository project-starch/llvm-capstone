# CRASH-121 -- unserialize-r-lowercase-backref-zval_ptr_dtor-frees-var_hash-tracked-current-zval-uaf

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | pure-ext, ext/standard/var_unserializer.c |
| site, as the source corpus records it | ext/standard/var_unserializer.c:886-890 (defect: `zval_ptr_dtor(rval)` at :887, faulting deref at :890) == ext/standard/var_unserializer.re:292-296 |
| upstream fix | `1ac4d8f2c632` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`ext/standard/var_unserializer.c:886`, from the museum tarball this corpus pins:

```c
  884  	}
  885  
  886  	if (*rval != NULL) {
  887  		zval_ptr_dtor(rval);
  888  	}
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | commit-history-reconstruction |
| origin_stream | history-5x-stdext |
| history_status | historical-known |
| confidence | high |
| crashes on a pristine build | True |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-121.php`. Nothing
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

Access stack of the first report: `php_var_unserialize` (ext/standard/var_unserializer.c:890), `process_nested_data` (ext/standard/var_unserializer.c:168).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
