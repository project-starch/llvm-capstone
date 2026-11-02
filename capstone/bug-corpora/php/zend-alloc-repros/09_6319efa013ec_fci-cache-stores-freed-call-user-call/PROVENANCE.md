# CRASH-038 -- fci-cache-stores-freed-call-user-call-trampoline-handler

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, Zend/zend_execute_API.c |
| site, as the source corpus records it | Zend/zend_execute_API.c:718-722 (unconditional `if (fci_cache) { fci_cache->function_handler = EX(function_state).function; ... }`); stale reuse at :725, faulting read at :733 |
| upstream fix | `6319efa013ec` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_execute_API.c:718`, from the museum tarball this corpus pins:

```c
  716  			}
  717  		}
  718  		if (fci_cache) {
  719  			fci_cache->function_handler = EX(function_state).function;
  720  			fci_cache->object_pp = fci->object_pp;
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

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-038.php`. Nothing
  was removed and nothing was added: it is not a reduction of ours.
- It runs on the whole CLI interpreter, so the defect is reached through PHP's own request
  lifecycle rather than through a driver around the allocator.
- There is no labelled read probe in a PHP script, so attribution rests on `fault_sites`, taken
  from the host-ASan report below and recorded before any Capstone or CheriBSD run.

## Measured at the pin (host ASan, 2026-10-28)

| build | outcome |
|---|---|
| detector patches (cache off, REAL_SIZE unrounded) | ASAN:heap-use-after-free |
| Zend's allocator as shipped | ASAN:heap-buffer-overflow |

Access stack of the first report: `zend_call_function` (Zend/zend_execute_API.c:733), `zif_array_map` (ext/standard/array.c:4085).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
