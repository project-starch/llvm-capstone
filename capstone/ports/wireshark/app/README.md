# Offline tshark application

Build the dependencies with `deps/build-*.sh`, then use `host/cross-build.sh`
and `host/build-domain.sh`. All compile/link steps use the shared delegated
ABI-v2 SDK. `JOBS` limits parallel compilation. The retained whitelist and
source patches define the offline dissector scope.

See [the shared application instructions](../../common/application/README.md)
for the compiler requirement and VM setup. Select an existing VM with
`CAPSTONE_VM_STATE`; set `TS_WORK` to the build root and `TSAPP_STOCK` to a native
stock tshark of the pinned release. `host/run-qemu.sh stages` checks milestones;
`host/run-qemu.sh oracle dhcp dns_port http arp dhcp.flip dns_port.flip http.flip
arp.flip ntp dns-ooo` compares application stdout byte for byte with native stock
tshark, and the guest's stderr byte for byte with the native MINIMAL build's
(`TSAPP_MINIMAL`, default `$TS_WORK/native-min-pa/run/tshark`: the same whitelist,
patches and generated `dissectors.c`, so the same registration notices, which stock
does not print). The guest's `TSAPP-HEAP` line, and under `CAPSTONE_DELEGATE_STATS`
the runtime's `capstone-domain:` and `capstone-exec:` lines, are removed first. An
empty reference is an error, never a match. `TSAPP_MINIMAL=none` skips the stderr
verdict and says so on every verdict line. `ntp` must differ on stdout because it is
outside the whitelist. Use `TSAPP_USER=UID:GID` for an unprivileged existing guest
account.

tshark takes its `malloc` from the platform's allocator. On the virtual profile
(`CAPSTONE_APPLICATION_PROFILE=virtual`) that is musl's mallocng, which bounds every
object and ends its lifetime on `free`; `TSAPP_HEAP` then selects nothing. On the
physical profile `TSAPP_HEAP=level0|shrink` selects the first-fit heap without or
with per-object bounds, for both `host/build-domain.sh` and the runner. The Sublet
heap arm (`TSAPP_HEAP=sublet`, `runtime/sublet_heap.c`, and patch 0007, which shrank
wmem's blocks to fit its pool) was removed on 2026-10-11; its runs stay in `results/`.

`host/build-domain.sh` gates every image: no undefined weak symbol, and no constructor section
outside the arrays the runtime walks. **Open:** the `sysalloc-none` (`TSAPP_HEAP=level0`) and `sysalloc-bounds`
(`TSAPP_HEAP=shrink`) arms run no link control. An ABI-v2 SDK links its runtime archive whole, so there is no runtime object to
leave out, and no v2 equivalent has been defined for them.

Safety fixtures use `common/application/check-safety.py --port wireshark`, with
`--state`, `--images`, `--out`, `--arm` and fixture numbers. The predictions in
`host/safety-expect.txt` are unchanged. Fault attribution requires the recorded
SIGSEGV, matching PC diagnostic and the fixture's target address. It does not
accept an arbitrary crash. Run this check with exclusive use of the selected VM.

`TSAPP_DOMAIN_ENV=NAME=value,...` adds environment to the domain; the delegated
runtime prints its unserved-syscall report under `CAPSTONE_DELEGATE_STATS=1`.

The full build of PR #128 lives in `/tmp/capstone/delegation-ports/tshark-source`
(dependencies, cross build, native stock tshark); its compile commands embed that
path, so a later runtime is qualified by setting `TS_WORK` to it and running
`host/build-domain.sh`, which relinks the images in seconds
(`results/2026-09-30-qemu-tshark-current-stack`).

There is no private libc-test launcher and no argv/environment side files.
Historical result directories describe their original binaries and transport.
