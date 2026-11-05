# CRASH-161 -- zval-copy-ctor-array-hands-over-hashtable-ownership-only-after-zend-hash-copy-bailout-leaves-two-zvals-owning-one-hashtable-double-free

## The upstream defect

| | |
|---|---|
| class | CWE-415 double-free (memory-corruption) |
| where | engine-core, Zend/zend_variables.c |
| site, as the source corpus records it | Zend/zend_variables.c:151-152 (zend_hash_copy runs before `zvalue->value.ht = tmp_ht;`) |
| upstream fix | `ed4c0245c7ca` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_variables.c:151`, from the museum tarball this corpus pins:

```c
  149  				ALLOC_HASHTABLE_REL(tmp_ht);
  150  				zend_hash_init(tmp_ht, 0, NULL, ZVAL_PTR_DTOR, 0);
  151  				zend_hash_copy(tmp_ht, original_ht, (copy_ctor_func_t) zval_add_ref, (void *) &tmp, sizeof(zval *));
  152  				zvalue->value.ht = tmp_ht;
  153  			}
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | commit-history-reconstruction |
| origin_stream | history-5x-roundE-core5(5.4-5.6) |
| history_status | historical-known |
| confidence | high |
| crashes on a pristine build | False |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-161.php`. Nothing
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

Access stack of the first report: `zend_hash_destroy` (Zend/zend_hash.c:514), `_zval_dtor` (Zend/zend_variables.c:52).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
