# CRASH-067 -- byref-return-parenthesized-undefined-var-assignref-rebind-stale-reference-uaf

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, Zend/zend_execute.c |
| site, as the source corpus records it | Zend/zend_execute.c:2243 |
| upstream fix | `8ce349b8e03c54fcbffb194fb0db61bd5f3ef49d` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_execute.c:2243`, from the museum tarball this corpus pins:

```c
 2241  int zend_assign_ref_handler(ZEND_OPCODE_HANDLER_ARGS)
 2242  {
 2243  	zval **value_ptr_ptr = get_zval_ptr_ptr(&opline->op2, EX(Ts), BP_VAR_W);
 2244  
 2245  	zend_assign_to_variable_reference(&opline->result, get_zval_ptr_ptr(&opline->op1, EX(Ts), BP_VAR_W), value_ptr_ptr, EX(Ts) TSRMLS_CC);
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | fuzzing |
| origin_stream | fuzz |
| history_status | historical-known |
| confidence | medium |
| crashes on a pristine build | False |
| build it needs | asan |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-067.php`. Nothing
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

Access stack of the first report: `zend_assign_to_variable` (Zend/zend_execute.c:676), `zend_assign_dim_handler` (Zend/zend_execute.c:2221).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
