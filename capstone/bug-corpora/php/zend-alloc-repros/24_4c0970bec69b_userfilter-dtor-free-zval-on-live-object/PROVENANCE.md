# CRASH-112 -- userfilter_dtor-free_zval-on-live-object-property-double-free

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | pure-ext, ext/standard/user_filters.c |
| site, as the source corpus records it | ext/standard/uuencode.c:0|ext/standard/user_filters.c:140 |
| upstream fix | `4c0970bec69b` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`ext/standard/user_filters.c:140`, from the museum tarball this corpus pins:

```c
  138  	if (SUCCESS == zend_hash_find(Z_OBJPROP_P(obj), "filter", sizeof("filter"), (void**)&tmp)) { 
  139  		zend_list_delete(Z_LVAL_PP(tmp));
  140  		FREE_ZVAL(*tmp);
  141  	} 
  142  
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
| build it needs | asan, requires writable /tmp |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-112.php`. Nothing
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

Access stack of the first report: `_zval_ptr_dtor` (Zend/zend_execute_API.c:389), `zend_hash_destroy` (Zend/zend_hash.c:519).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
