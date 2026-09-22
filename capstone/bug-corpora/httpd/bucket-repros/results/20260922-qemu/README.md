# Eight cases, sixteen arms, under QEMU: spatial completes, sublet faults at the probe, 2026-09-22

Every `spatial` arm completed and every `sublet` arm faulted with cause 24
(revoked authority) at the address the boot itself published for
`apr_defect_read` -- eight of eight, case 4 included, whose reduced consumer
declares the connection handback as the lender's epoch (its PROVENANCE.md
says why that is the one step upstream does not take). The negative control
beside this record corrupted every fixture so no case ran, and all six
selected oracles reported FAIL, as they must; `arms_passed: 0` there is the
control firing, not a failure.

    matrix.tsv    one line per arm: verdict, cause, fault PC, published probe address
    inputs.json   image, loader, emulator and compiler hashes per arm

QEMU `ce93cb32…`, the Capstone clang `5003f54c…`, revocation node pool
65,536; eight images, one per case, built through `ports/apr/pools` with
`-DAPRP_BUCKETS=ON`. Which mechanism caught each case is in the corpus README:
cases 2, 3, 5 and 7 end at a pool destroy and are the pool port's release;
cases 0, 1, 4 and 6 end when a bucket allocator's blocks go back to APR --
at its destroy, or at the handback the reduced consumer declares -- and are
the same release reached through the bucket allocator's lend. The bucket
allocator's own freelist -- file and reissue of a small node -- is exercised
by no case here and is the port's fixture suite instead
(`ports/apr/pools/security-tests/qemu/run-buckets.py`).

This is the day's second run of the suite; the first, with case 4 still
reading a live bucket, is retained in the archive. Raw serial logs are archived outside the repository at
`~/artifacts/httpd/20260922-bucket-corpus/raw-campaign.tar.gz`.
