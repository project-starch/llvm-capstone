# Eight cases, sixteen arms, under QEMU: spatial completes, sublet faults at the probe, 2026-09-22

Every `spatial` arm completed and every `sublet` arm did what its oracle
says: seven faults with cause 24 (revoked authority) at the address the boot
itself published for `apr_defect_read`, and one completion -- case 4, whose
reduced sequence ends no lifetime and whose oracle requires the completion,
a recorded non-detection rather than a pass by accident. The negative
control beside this record corrupted every fixture so no case ran, and all
six selected oracles reported FAIL, as they must; `arms_passed: 0` there is
the control firing, not a failure.

    matrix.tsv    one line per arm: verdict, cause, fault PC, published probe address
    inputs.json   image, loader, emulator and compiler hashes per arm

QEMU `ce93cb32…`, the Capstone clang `5003f54c…`, revocation node pool
65,536; eight images, one per case, built through `ports/apr/pools` with
`-DAPRP_BUCKETS=ON`. Which mechanism caught each case is in the corpus README:
cases 2, 3, 5 and 7 end at a pool destroy and are the pool port's release;
cases 0, 1 and 6 end when a bucket allocator's blocks go back to APR, and are
the same release reached through the bucket allocator's lend. The bucket
allocator's own freelist -- file and reissue of a small node -- is exercised
by no case here and is the port's fixture suite instead
(`ports/apr/pools/security-tests/qemu/run-buckets.py`).

Raw serial logs are archived outside the repository at
`~/artifacts/httpd/20260922-bucket-corpus/raw-campaign.tar.gz`.
