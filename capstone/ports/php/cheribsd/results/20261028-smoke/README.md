# 20261028-smoke

Does PHP 5.0.0, built purecap for CheriBSD, actually run? Yes.

    /tmp/php -n -v              -> PHP 5.0.0 (cli) banner, rc=1 (as native PHP 5.0.0 does)
    /tmp/php -n smoke.php       -> SMOKE 3 / 1,2,3, rc=0

The script sorts, which is the path `patches/0002` exists for: PHP's own `_zend_qsort_swap` moves
`Bucket *` through `int` and `char` stores, which strips a capability's tag. The same patch the
virtual port needs is applied here, and the sort comes back right.

This is a qualification, not a measurement: no corpus case ran. `inputs.json` records the binary's
hash, the ELF flags that were asserted rather than assumed, and the guest.
