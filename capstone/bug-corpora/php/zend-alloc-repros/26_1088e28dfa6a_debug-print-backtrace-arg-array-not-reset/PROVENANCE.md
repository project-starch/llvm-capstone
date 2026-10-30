# CRASH-140 -- debug-print-backtrace-arg-array-not-reset-per-frame-stale-freed-pointer-uaf-double-free

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, Zend/zend_builtin_functions.c |
| site, as the source corpus records it | Zend/zend_builtin_functions.c:1453 (`zval *arg_array = NULL;` hoisted out of the `while (ptr)` loop at :1486); UAF read at :1433 via :1558; double free at :1559 |
| upstream fix | `1088e28dfa6a` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_builtin_functions.c:1453`, from the museum tarball this corpus pins:

```c
 1451  	char *call_type;
 1452  	char *include_filename = NULL;
 1453  	zval *arg_array = NULL;
 1454  	void **cur_arg_pos = EG(argument_stack).top_element;
 1455  	void **args = cur_arg_pos;
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | commit-history-reconstruction |
| origin_stream | history-5x-core3 |
| history_status | historical-known |
| confidence | high |
| crashes on a pristine build | False |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-140.php`. Nothing
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

Access stack of the first report: `debug_print_backtrace_args` (Zend/zend_builtin_functions.c:1433), `zif_debug_print_backtrace` (Zend/zend_builtin_functions.c:1558).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
