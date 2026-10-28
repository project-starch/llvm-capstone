# CRASH-159 -- assign-op-and-incdec-property-helpers-hold-unaddrefd-object-across-magic-getter-then-write-property-uaf

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, Zend/zend_execute.c |
| site, as the source corpus records it | Zend/zend_execute.c:1699 (zend_binary_assign_op_obj_helper); identical construct at Zend/zend_execute.c:1223 and Zend/zend_execute.c:1281 |
| upstream fix | `abb09693ac4d` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_execute.c:1699`, from the museum tarball this corpus pins:

```c
 1697  			switch (opline->extended_value) {
 1698  				case ZEND_ASSIGN_OBJ:
 1699  					Z_OBJ_HT_P(object)->write_property(object, property, z TSRMLS_CC);
 1700  					break;
 1701  				case ZEND_ASSIGN_DIM:
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | commit-history-reconstruction |
| origin_stream | history-5x-roundE-core5(5.4-5.6) |
| history_status | historical-known |
| confidence | high |
| crashes on a pristine build | True |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-159.php`. Nothing
  was removed and nothing was added: it is not a reduction of ours.
- It runs on the whole CLI interpreter, so the defect is reached through PHP's own request
  lifecycle rather than through a driver around the allocator.
- There is no labelled read probe in a PHP script, so attribution rests on `fault_sites`, taken
  from the host-ASan report below and recorded before any Capstone or CheriBSD run.

## Measured at the pin (host ASan, 2026-10-28)

| build | outcome |
|---|---|
| detector patches (cache off, REAL_SIZE unrounded) | ASAN:heap-use-after-free |
| Zend's allocator as shipped | ASAN:SEGV |

Access stack of the first report: `zend_binary_assign_op_obj_helper` (Zend/zend_execute.c:1699), `zend_binary_assign_op_helper` (Zend/zend_execute.c:1732).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
