# CRASH-103 -- stristr-zval_ptr_dtor-on-own-passed-args-double-free-uaf

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | pure-ext, ext/standard/string.c |
| site, as the source corpus records it | ext/standard/string.c:1412-1413 (also 1385-1386 on the empty-needle path) |
| upstream fix | `90f4590d87fe` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`ext/standard/string.c:1412`, from the museum tarball this corpus pins:

```c
 1410  	}
 1411  
 1412  	zval_ptr_dtor(haystack);
 1413  	zval_ptr_dtor(needle);
 1414  	efree(haystack_orig);
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | commit-history-reconstruction |
| origin_stream | history-5x-stdext |
| history_status | historical-known |
| confidence | high |
| crashes on a pristine build | False |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-103.php`. Nothing
  was removed and nothing was added: it is not a reduction of ours.
- It runs on the whole CLI interpreter, so the defect is reached through PHP's own request
  lifecycle rather than through a driver around the allocator.
- There is no labelled read probe in a PHP script, so attribution rests on `fault_sites`, taken
  from the host-ASan report below and recorded before any Capstone or CheriBSD run.

## Measured at the pin (host ASan, 2026-10-28)

| build | outcome |
|---|---|
| detector patches (cache off, REAL_SIZE unrounded) | ASAN:heap-use-after-free |
| Zend's allocator as shipped | OK |

Access stack of the first report: `_zval_ptr_dtor` (Zend/zend_execute_API.c:389), `zend_ptr_stack_clear_multiple` (Zend/zend_execute.h:124).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
