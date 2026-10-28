# CRASH-072 -- open-stream-derefs-freed-volatile-url-wrapper-hash-at-request-shutdown-uaf

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, main/streams/streams.c |
| site, as the source corpus records it | main/streams/streams.c:1403 (defect origin) / main/streams/streams.c:350 (faulting deref) |
| upstream fix | `d5bb03b11a964c74e1ae9a1c161f84ff83d3c3f8` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`main/streams/streams.c:1403`, from the museum tarball this corpus pins:

```c
 1401  		FG(stream_wrappers) = emalloc(sizeof(HashTable));
 1402  		zend_hash_init(FG(stream_wrappers), 0, NULL, NULL, 1);
 1403  		zend_hash_copy(FG(stream_wrappers), &url_stream_wrappers_hash, NULL, &tmpwrapper, sizeof(php_stream_wrapper));
 1404  	}
 1405  
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | fuzzing |
| origin_stream | fuzz |
| history_status | historical-known |
| confidence | medium |
| crashes on a pristine build | True |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-072.php`. Nothing
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

Access stack of the first report: `_php_stream_free` (main/streams/streams.c:350), `stream_resource_regular_dtor` (main/streams/streams.c:1343).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
