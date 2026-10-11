# row8_cpython_backup_closed

Upstream: https://github.com/python/cpython/issues/85981
Table 3 row: 8
Class: use-after-close
Essence: Backup setup reads a source database handle from a host connection wrapper after that wrapper has been closed and freed.
Observed: ERROR: AddressSanitizer: heap-use-after-free
