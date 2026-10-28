# CRASH-042 -- magic-getter-frees-container-then-in_get-guard-write-uaf

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, Zend/zend_object_handlers.c |
| site, as the source corpus records it | Zend/zend_object_handlers.c:294 |
| upstream fix | `41ad9b4d1fdd` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_object_handlers.c:294`, from the museum tarball this corpus pins:

```c
  292  			zobj->in_get = 1; /* prevent circular getting */
  293  			rv = zend_std_call_getter(object, member TSRMLS_CC);
  294  			zobj->in_get = 0;
  295  
  296  			if (rv) {
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | commit-history-reconstruction |
| origin_stream | history-5.x |
| history_status | historical-known |
| confidence | high |
| crashes on a pristine build | False |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-042.php`. Nothing
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

Access stack of the first report: `zend_std_read_property` (Zend/zend_object_handlers.c:294), `zend_fetch_property_address_read` (Zend/zend_execute.c:1157).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
