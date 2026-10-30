# CRASH-065 -- zend_do_free-marks-unrelated-jmp_no_ctor-result-unused-freeing-live-temp

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | engine-core, Zend/zend_compile.c |
| site, as the source corpus records it | Zend/zend_compile.c:894 |
| upstream fix | `89e53d5ab231` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`Zend/zend_compile.c:894`, from the museum tarball this corpus pins:

```c
  892  				 * proceeding INIT_FCALL_BY_NAME as unused
  893  				 */
  894  				if (opline->opcode == ZEND_JMP_NO_CTOR) {
  895  					opline->op1.u.EA.type |= EXT_TYPE_UNUSED;
  896  					(opline-1)->result.u.EA.type |= EXT_TYPE_UNUSED;
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

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-065.php`. Nothing
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

Access stack of the first report: `zend_pzval_unlock_func` (Zend/zend_execute.c:65), `_get_zval_ptr` (Zend/zend_execute.c:100).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
