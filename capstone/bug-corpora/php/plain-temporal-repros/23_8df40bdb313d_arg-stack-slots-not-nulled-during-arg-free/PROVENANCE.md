# CRASH-142 -- arg-stack-slots-not-nulled-during-arg-free-loop-reentrant-backtrace-reads-freed-arg-zvals-uaf

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, Zend/zend_execute.h |
| site, as the source corpus records it | Zend/zend_execute.h:117-127 (defective free loop at :124); observed read of the stale slot at Zend/zend_builtin_functions.c:1420 |
| upstream fix | `8df40bdb313d` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_execute.h:117`, from the museum tarball this corpus pins:

```c
  115  
  116  /* dedicated Zend executor functions - do not use! */
  117  static inline void zend_ptr_stack_clear_multiple(TSRMLS_D)
  118  {
  119  	void **p = EG(argument_stack).top_element-2;
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | commit-history-reconstruction |
| origin_stream | history-5x-core3 |
| history_status | historical-known |
| confidence | high |
| crashes on a pristine build | True |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-142.php`. Nothing
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

Access stack of the first report: `debug_backtrace_get_args` (Zend/zend_builtin_functions.c:1420), `zend_fetch_debug_backtrace` (Zend/zend_builtin_functions.c:1650).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
