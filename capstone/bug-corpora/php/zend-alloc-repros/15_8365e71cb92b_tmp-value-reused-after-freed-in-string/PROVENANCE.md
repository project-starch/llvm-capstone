# CRASH-057 -- tmp-value-reused-after-freed-in-string-offset-assign

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, Zend/zend_execute.c |
| site, as the source corpus records it | Zend/zend_execute.c:535 |
| upstream fix | `8365e71cb92b` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_execute.c:535`, from the museum tarball this corpus pins:

```c
  533  		
  534  /*		T(result->u.var).var.ptr_ptr = &EG(uninitialized_zval_ptr); */
  535  		T(result->u.var).var.ptr_ptr = &value;
  536  		SELECTIVE_PZVAL_LOCK(*T(result->u.var).var.ptr_ptr, result);
  537  		AI_USE_PTR(T(result->u.var).var);
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | commit-history-reconstruction |
| origin_stream | history-5.x |
| history_status | historical-known |
| confidence | high |
| crashes on a pristine build | True |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-057.php`. Nothing
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

Access stack of the first report: `_estrndup` (Zend/zend_alloc.c:403), `_zval_copy_ctor` (Zend/zend_variables.c:137).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
