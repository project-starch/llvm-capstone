# CRASH-069 -- overloaded-call-trampoline-fbc-efreed-then-derefd-on-pending-exception-uaf

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, Zend/zend_execute.c |
| site, as the source corpus records it | Zend/zend_execute.c:2778 |
| upstream fix | `2507a544296b20623131e4ab34e0410594d02596` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_execute.c:2778`, from the museum tarball this corpus pins:

```c
 2776  
 2777  	if (EG(This)) {
 2778  		if (EG(exception) && EX(fbc) && EX(fbc)->common.fn_flags&ZEND_ACC_CTOR) {
 2779  			EG(This)->refcount--;
 2780  			zval_ptr_dtor(&EG(This));
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | fuzzing |
| origin_stream | fuzz |
| history_status | historical-known |
| confidence | high |
| crashes on a pristine build | True |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-069.php`. Nothing
  was removed and nothing was added: it is not a reduction of ours.
- It runs on the whole CLI interpreter, so the defect is reached through PHP's own request
  lifecycle rather than through a driver around the allocator.
- There is no labelled read probe in a PHP script, so attribution rests on `fault_sites`, taken
  from the host-ASan report below and recorded before any Capstone or CheriBSD run.

## Measured at the pin (host ASan, 2026-10-28)

| build | outcome |
|---|---|
| detector patches (cache off, REAL_SIZE unrounded) | ASAN:heap-use-after-free |
| Zend's allocator as shipped | EXIT:255 |

Access stack of the first report: `zend_do_fcall_by_name_handler` (Zend/zend_execute.c:2810).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
