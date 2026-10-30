# CRASH-148 -- cast-to-object-adopts-globals-symbol-table-hashtable-as-owned-object-properties-invalid-free

## The upstream defect

| | |
|---|---|
| class | CWE-590 invalid-free (memory-corruption) |
| where | engine-core, Zend/zend_operators.c |
| site, as the source corpus records it | Zend/zend_operators.c:661 |
| upstream fix | `2a31dbbadfb9` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_operators.c:661`, from the museum tarball this corpus pins:

```c
  659  				TSRMLS_FETCH();
  660  
  661  				object_and_properties_init(op, zend_standard_class_def, op->value.ht);
  662  				return;
  663  				break;
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | commit-history-reconstruction |
| origin_stream | history-5x-core4 |
| history_status | fixed-by-rewrite |
| confidence | high |
| crashes on a pristine build | True |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-148.php`. Nothing
  was removed and nothing was added: it is not a reduction of ours.
- It runs on the whole CLI interpreter, so the defect is reached through PHP's own request
  lifecycle rather than through a driver around the allocator.
- There is no labelled read probe in a PHP script, so attribution rests on `fault_sites`, taken
  from the host-ASan report below and recorded before any Capstone or CheriBSD run.

## Measured at the pin (host ASan, 2026-10-28)

| build | outcome |
|---|---|
| detector patches (cache off, REAL_SIZE unrounded) | ASAN:attempting free on address |
| Zend's allocator as shipped | ASAN:attempting free on address |

Access stack of the first report: `_efree` (Zend/zend_alloc.c:287), `zend_objects_free_object_storage` (Zend/zend_objects.c:89).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
