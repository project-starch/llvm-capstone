# ffmpeg/pool-repros on PoisonCap, and case 3's backing arm -- 2026-10-09

PoisonCap: the rebuilt published platform (ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap), the
pool port built with FFPOOL_POISONCAP (host/cheribsd/poisoncap/build.sh), one boot:

    python3 ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/run.py <build> <out> --stage pool \
      --disable-default-revocation --case poison-live --case poison-read --case poison-write \
      --case poison-reuse --case poison-reused-read --case pool-{0,2}-{36,37,38,39} --sdk ... --rootfs ... --image ...

Probe 39 (case 3) was added to run.py's list today; 36-38 were registered but had no committed bundle
(the advisory in their case.json), so this is also their first committed reading.

## Platform controls, same boot

    poison-live: exit 0  PASS
    poison-read: exit 162  PASS
    poison-write: exit 162  PASS
    poison-reuse: exit 0  PASS
    poison-reused-read: exit 162  PASS
    cheribsd-abi: exit 0  PASS
    cheribsd-bounds: exit 162  PASS

## Cases

    pool-0-36: exit 0  PASS
    pool-2-36: exit 162  PASS
    pool-0-37: exit 0  PASS
    pool-2-37: exit 162  PASS
    pool-0-38: exit 0  PASS
    pool-2-38: exit 162  PASS
    pool-0-39: exit 0  PASS
    pool-2-39: exit 162  PASS

## Case 3's backing arm (Capstone, mode 1: revocation of the pool's backing allocation)

    bash ports/ffmpeg/buffer-pool/security-tests/qemu/run.sh <out> --cases 39 --modes 1 --rounds 1

    {"mode": 1, "case": 39, "name": "vvc-nonref-output-releases-tabs-stale-read", "expected": "completed", "runner_exit": 0, "domain_sha256": "ac4473aa27353f9ef205f3595114ac0ed022b2b94151cb26ca4d48b4aa3dd27e", "passed": true}

The domain image is ac4473aa27353f9e..., the one the 2026-10-04 probe-39 pair ran (its SHA256SUMS);
probe 39's code is unchanged since (only probes 40-42 were added).
