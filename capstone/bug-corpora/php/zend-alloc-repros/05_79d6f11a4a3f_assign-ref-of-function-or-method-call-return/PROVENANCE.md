# CRASH-024 -- assign-ref-of-function-or-method-call-return-drops-live-this-uaf

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, Zend/zend_compile.c |
| site, as the source corpus records it | Zend/zend_compile.c:487-508 (zend_do_assign_ref - missing `if (zend_is_function_or_method_call(rvar)) opline->extended_value = ZEND_RETURNS_FUNCTION;`) + Zend/zend_execute.c:2241-2248 (zend_assign_ref_handler - missing the corresponding fcall_returned_reference guard). NOTE: the claimed 'Zend/zend_compile.c:582' is wrong (that line is inside zend_do_for_before_statement) and 'Zend/zend_vm_def.h' does not exist at 5.0.0 - the VM was still hand-written zend_execute.c; zend_vm_def.h arrives in 5.1. |
| upstream fix | `79d6f11a4a3f` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_compile.c:487`, from the museum tarball this corpus pins:

```c
  485  
  486  
  487  void zend_do_assign_ref(znode *result, znode *lvar, znode *rvar TSRMLS_DC)
  488  {
  489  	int last_op_number = get_next_op_number(CG(active_op_array))-1;
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

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-024.php`. Nothing
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

Access stack of the first report: `zend_pzval_lock_func` (Zend/zend_execute.c:59), `zend_fetch_var_address` (Zend/zend_execute.c:798).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
