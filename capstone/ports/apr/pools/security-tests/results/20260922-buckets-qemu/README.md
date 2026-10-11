# The bucket allocator's seven lifetime fixtures, both modes, under QEMU, 2026-09-22

14 of 14 arms as the table in `security-tests/qemu/run-buckets.py` requires.
Mode 0 completes where nothing is out of bounds and refuses the double free
by its records (`538`); mode 1 faults at the labelled site every time the
lifetime ended -- the read and the write through a filed small node (cause
24), a freed large node, the allocator destroyed, and the double free at the
allocator's own probe `aprb_free_probe`. The bounds fixture faults in both
modes with cause 5 at the read, the positive control that faults are seen at
all; the live control completes in both, having observed the LIFO reissue.

    matrix.tsv    one line per arm: verdict, cause, fault PC, published site
    inputs.json   image, loader, emulator and compiler hashes per arm

This is the second run of the suite. The first, the same day, passed 13 of
14 and lost fixture 6 in mode 1 to a 90-second guest timeout while a
CheriBSD build ran on the same host -- no marker, no fault, no report -- an
infrastructure loss, not a measurement; the run before that, on the previous
build, faulted at the capability arithmetic ahead of the probe, which is
what moved the probe in front of the arithmetic in the hooks patch. Raw
serial logs are archived outside the repository with the corpus's at
`~/artifacts/httpd/20260922-bucket-corpus/raw-campaign.tar.gz`.
