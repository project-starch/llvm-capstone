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
arp.flip ntp dns-ooo` compares application stdout byte for byte. Native and guest
stderr are retained separately. `ntp` must differ because it is outside the
whitelist. Use `TSAPP_USER=UID:GID` for an unprivileged existing guest account.

Safety fixtures use `common/application/check-safety.py --port wireshark`, with
`--state`, `--images`, `--out`, `--arm` and fixture numbers. The predictions in
`host/safety-expect.txt` are unchanged. Fault attribution requires the recorded
SIGSEGV, matching PC diagnostic and the fixture's target address. It does not
accept an arbitrary crash. Run this check with exclusive use of the selected VM.

There is no private libc-test launcher and no argv/environment side files.
Historical result directories describe their original binaries and transport.
