# CRASH-125 -- mb_detect_encoding-parse-encoding-list-fail-list-not-nulled-double-free

## The upstream defect

| | |
|---|---|
| class | CWE-415 double-free (memory-corruption) |
| where | pure-ext, ext/mbstring/mbstring.c |
| site, as the source corpus records it | ext/mbstring/mbstring.c:2200 (efree without list=NULL); second free at ext/mbstring/mbstring.c:2230 |
| upstream fix | `09370413757a` |

**The fix hunk is not quoted here.** php-src's history is not available on this host, so the fix is
cited by hash and the PIN's own defective source is quoted instead. Quoting the hunk is the one
thing this file owes a reader that it cannot yet pay; it can be added once a php-src mirror is on
this machine.

## The pin's own source

`ext/mbstring/mbstring.c:2200`, from the museum tarball this corpus pins:

```c
 2198  			if (!php_mb_parse_encoding_list(Z_STRVAL_PP(arg_list), Z_STRLEN_PP(arg_list), &list, &size, 0 TSRMLS_CC)) {
 2199  				if (list) {
 2200  					efree(list);
 2201  					size = 0;
 2202  				}
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

- `trigger.php` is that corpus's own script, copied byte for byte from `corpus/CRASH-125.php`. Nothing
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

Access stack of the first report: `_efree` (Zend/zend_alloc.c:263), `zif_mb_detect_encoding` (ext/mbstring/mbstring.c:2230).

The pair is the evidence for this case's corpus: a release the system allocator sees keeps its
reading with the allocator as shipped; one Zend's cache absorbs loses it.
