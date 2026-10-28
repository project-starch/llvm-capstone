# CRASH-010 -- foreach-key-eq-value-refcount-uaf

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, Zend/zend_execute.c |
| site, as the source corpus records it | Zend/zend_execute.c:142 |
| upstream fix | `fc96c7f7fa18b0fc9216bcc9f94e72a753da7689` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_execute.c:142`, from the museum tarball this corpus pins:

```c
  140  	if (node->op_type==IS_VAR) {
  141  		if (T(node->u.var).var.ptr_ptr) {
  142  			PZVAL_UNLOCK(*T(node->u.var).var.ptr_ptr);
  143  		} else {
  144  			/* string offset */
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | carried-from-4.0.2 |
| origin_stream | carried-from-4.0.2 |
| history_status | fixed-by-rewrite |
| confidence | high |
| crashes on a pristine build | False |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-010.php`. Nothing
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
