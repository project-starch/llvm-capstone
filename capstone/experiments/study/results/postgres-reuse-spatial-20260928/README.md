# PostgreSQL spatial PoisonCap-adapter reuse qualification

Three complete PostgreSQL 17.5 single-user `work.sql` processes passed the
native 16-byte-MAXALIGN oracle: all 22 normalized SQL result rows agree,
including the final count of 1,500. Each process started from a fresh copy
of the same pristine cluster in one CheriBSD guest. The opt-in observer at
the inner memory-context chunk boundary reports exactly 54,004 handouts,
51,149 releases and 44,974 observed start reuses per process; all three
32-bin histograms are identical, reconcile to the reuse count and have
`error=0`. The CheriBSD outer revocation default was disabled; this is the
PoisonCap adapter's spatial control (`PG_POISONCAP_MODE=0`).

The [raw archive](cheribsd-raw.tar.gz) preserves the exact runner, point,
platform and file hashes, fresh-cluster declaration, per-process command,
stdout/stderr and result records. The [build manifest](build-manifest.json)
records the binary and source inputs. `python3 validate.py` checks the
archive against those identities and the native row hash. The earlier
runner attempt that failed only to strip PostgreSQL's `backend>` prompt from
the final observer line is excluded; this archive comes from the corrected
three-pass rerun.

The histogram describes distances among starts **observed to be reused**,
indexed by successful handouts. It is not a fixed-follow-up retirement
fraction or a physical working-set measure. The protected PoisonCap arm has
not passed the full SQL workload, so this is not a four-arm figure. A
scratch-only queue candidate still stalled during the INSERT even when its
`cheri_revoke` call was disabled; that candidate was not included here.
