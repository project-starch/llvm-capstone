# CRASH-129 -- mb-ereg-replace-e-option-eval-failure-stale-dangling-zval-v-uaf-double-free

## The upstream defect

| | |
|---|---|
| class | CWE-416 use-after-free (memory-corruption) |
| where | pure-ext, ext/mbstring/php_mbregex.c |
| site, as the source corpus records it | ext/mbstring/php_mbregex.c:732 (unchecked zend_eval_string) — fault surfaces at ext/mbstring/php_mbregex.c:735 |
| upstream fix | `3e0680f2ee00` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`ext/mbstring/php_mbregex.c:732`, from the museum tarball this corpus pins:

```c
  730  				smart_str_appendc(&eval_buf, '\0');
  731  				/* do eval */
  732  				zend_eval_string(eval_buf.c, &v, description TSRMLS_CC);
  733  				/* result of eval */
  734  				convert_to_string(&v);
```

## How it was collected

Carried over from the external PHP 5.0.0 bug corpus, which records per case:

| | |
|---|---|
| discovery_method | commit-history-reconstruction |
| origin_stream | history-5x-tier2 |
| history_status | historical-known |
| confidence | high |
| crashes on a pristine build | False |
| build it needs | fullext |

## What is real and what is reduced

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-129.php`. Nothing
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

Access stack of the first report: `_php_mb_regex_ereg_replace_exec` (ext/mbstring/php_mbregex.c:735), `zif_mb_ereg_replace` (ext/mbstring/php_mbregex.c:780).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
