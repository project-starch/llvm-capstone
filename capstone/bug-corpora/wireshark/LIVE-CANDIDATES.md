# Wireshark 4.6.8: 92 defects whose fix was never backported, live at our pin

**Read this if you are looking for cases to build.** The wmem corpus's 18 cases are 16
`live_in_pin: false` and 2 live; the plain-heap corpus's one case is not live either. That
looked like the supply running out. It was the **choice of population**.

| population | commits | spatial-worded | temporal-worded |
|---|---:|---:|---:|
| `v4.6.8..origin/release-4.6` — what the earlier triage searched | **83** | 11 | 1 |
| `v4.6.8..origin/master` | **4,401** | **206** | 16 |

The maintenance branch was chosen because *"master carries the same fixes under different
hashes plus work that never reached 4.6"*
([`wireshark-wmem-defect-triage.md`](../../docs/ref/wireshark-wmem-defect-triage.md)). That is
a sound argument against using **ancestry** as a liveness test. It is not an argument for the
narrower population, and it throws away exactly the class we care about most: **a defect whose
fix was never backported is still in 4.6.8**, and it is invisible to a search that only reads
the 4.6 branch. The Perl corpus hit the same thing from the other side — 4 of its 11 cases are
fixes *dated before* their pin that never reached the maintenance branch.

## What was done

Every one of the 221 spatial- or temporal-worded commits in `v4.6.8..origin/master` that
touches `wiretap/`, `epan/` or `wsutil/` C source was checked for liveness **by content**:
take the lines the fix ADDS to a C file, and ask whether the pinned file already contains them.

| verdict | commits | meaning |
|---|---:|---|
| **LIVE** | **92** | none of the fix's added lines are in the pinned file |
| `FIXED-AT-PIN` | 66 | most are — the fix was backported after all |
| `PARTIAL` | 37 | some are. **Unread, not absent** — worth a human |
| `NO-FILE` | 6 | the file does not exist at 4.6.8 |
| `NO-SIGNAL` | 20 | the diff has no line distinctive enough to search for |

That 66 is what makes the 92 meaningful: the check demonstrably separates a backported fix
from an unfixed one rather than calling everything live.

**Two were verified by hand against the pinned source**, because a list produced by a script is
a claim until one is read:

- `b2bc518e4d` (Bencode) — the pin has the unfixed
  `proto_tree_add_item(itree, hf_bencode_truncated_data, tvb, offset + used, -1, ENC_NA)` at
  `packet-bencode.c:264`, while line 234 already carries the `length ? -1 : 0` form. The fix's
  own comment says exactly that: *"the dictionary above gets this right already"*.
- `716a200295` (RTPS, **OOB write**) — at the pin, `sample_info_flags` and `sample_info_length`
  are `wmem_alloc(pinfo->pool, … * sample_info_max)` (`packet-rtps.c:15993-15994`), the guard at
  `:16005` is the unfixed `rtps_max_batch_samples_dissected > 0 && …`, and `:16014` writes
  `sample_info_flags[sample_info_count]` unbounded. With the preference at its default 0 the
  loop has no bound and `sample_info_max` is 1024, so a batch with more samples writes past both
  arrays — **an out-of-bounds write in wmem pool memory, live at 4.6.8.**

## Where to start

Rows marked **!** are the ones where upstream names the memory defect in the subject itself.
Sorted so those come first, then the `wmem packet scope` layer, which is this study's
nested-allocator cell.

`716a200295` is the single most valuable row: the inventory records the nested-spatial cell as
having **no live upstream defect**, and this is one.

## Caveats, so nobody over-reads the table

- **LIVE means the pinned file lacks the fix's added lines.** It does not yet mean the defect is
  *reachable* in a tshark run, and it does not mean a trigger exists. Both still have to be
  shown per case, as the existing cases do.
- The allocator column is **coarse**: `epan/dissectors/` almost always allocates from a wmem
  packet scope and `wiretap/` from `g_malloc`/`ws_buffer`, but the case's own `allocator_layer`
  must be read from the code, not from the directory.
- Wording still gates the 221. A shape-based pass over all 4,401 would admit more; this list is
  a floor, not a census.

| | sha | file at the pin | upstream's subject | allocator layer |
|---|---|---|---|---|
| **!** | `b2bc518e4d` | `epan/dissectors/packet-bencode.c` | Bencode: Don't run past the end of a truncated list | wmem packet scope |
| **!** | `0defc09e5f` | `epan/dissectors/packet-http.c` | HTTP: Prevent a use-after-free | wmem packet scope |
| **!** | `d829fcf75c` | `epan/dissectors/packet-ieee802154.c` | IEEE 802.15.4: Fix ccm_cbc_mac stack buffer underflow | wmem packet scope |
| **!** | `c8c396cf23` | `epan/dissectors/packet-obex.c` | OBEX: Fix an off by one when going up a path level | wmem packet scope |
| **!** | `716a200295` | `epan/dissectors/packet-rtps.c` | RTPS: Fix OOB write in DATA_BATCH sample info list | wmem packet scope |
| **!** | `373504f7c9` | `epan/dfilter/dfvm.c` | dfvm: Fix an error message to avoid an out-of-bounds read | epan/wsutil |
|  | `0a79d693cc` | `epan/dissectors/asn1/c1222/packet-c1222-template.c` | C12.22: Use tvb_captured_length_remaining | wmem packet scope |
|  | `778c180461` | `epan/dissectors/asn1/kerberos/packet-kerberos-template.c` | kerberos: Fix unwanted overflow in get_krb_pdu_len | wmem packet scope |
|  | `40c2c3bcda` | `epan/dissectors/dcerpc/idl2wrs.c` | idl2wrs: Fix a check for a single bit set in a bitmap | wmem packet scope |
|  | `2edee2bed5` | `epan/dissectors/packet-aeron.c` | epan: Use unsigned offset and lengths in tvb_skip_wsp, _uint8 | wmem packet scope |
|  | `c6d8f8950c` | `epan/dissectors/packet-afp.c` | Use the constant from wsutil/epochs.h for Un*x epoch to Y2K | wmem packet scope |
|  | `ac3e247292` | `epan/dissectors/packet-agentx.c` | AgentX: Take a subset TVB to reduce the chance of overflow | wmem packet scope |
|  | `e4d19b09c9` | `epan/dissectors/packet-alp.c` | alp: Coverity does not like post-decrement in while loops | wmem packet scope |
|  | `474b0b4a22` | `epan/dissectors/packet-ansi_a.c` | ANSI A: Fix overflow check | wmem packet scope |
|  | `e634ab8fc4` | `epan/dissectors/packet-bencode.c` | Bencode: Dissect 64 bit integers | wmem packet scope |
|  | `a1b93ac34a` | `epan/dissectors/packet-bencode.c` | Bencode: Fix signed overflow UB | wmem packet scope |
|  | `7446aa4375` | `epan/dissectors/packet-ber.c` | BER: prevent possible signed overlow undefined behavior | wmem packet scope |
|  | `5623d66c4d` | `epan/dissectors/packet-ber.c` | BER: Don't overflow converting octet len to bit length in BIT STRING | wmem packet scope |
|  | `2da9407eba` | `epan/dissectors/packet-bgp.c` | BGP: Check for underflow in MCAST NLRI | wmem packet scope |
|  | `1edd9ad6e2` | `epan/dissectors/packet-brdwlk.c` | brdwlk: Fix a field size (and signed overflow) | wmem packet scope |
|  | `6a5382d133` | `epan/dissectors/packet-btavctp.c` | BT AVCTP: Check for overflow when reassembling | wmem packet scope |
|  | `bae955afe4` | `epan/dissectors/packet-bthci_acl.c` | BT HCI ACL: Do not overflow in reassembly | wmem packet scope |
|  | `6b5ff490f4` | `epan/dissectors/packet-bzr.c` | BZR: Improve an infinite loop check | wmem packet scope |
|  | `b070d4dd24` | `epan/dissectors/packet-cola.c` | Use the new tvb_find_uint8_ calls in a few dissectors | wmem packet scope |
|  | `d62ebd14bf` | `epan/dissectors/packet-dbus.c` | epan: Use unsigned offsets for the tvb_get_ accessors | wmem packet scope |
|  | `9a07a60398` | `epan/dissectors/packet-dcerpc-spoolss.c` | SPOOLSS: overflow | wmem packet scope |
|  | `f553d3f796` | `epan/dissectors/packet-dcm.c` | DCM: check for a potential overflow on the last packet | wmem packet scope |
|  | `17689584d7` | `epan/dissectors/packet-dcm.c` | DICOM: Avoid overflow in Export Objects | wmem packet scope |
|  | `32e71e73cd` | `epan/dissectors/packet-dcm.c` | DICOM: Integer promotions: still weird | wmem packet scope |
|  | `ff39300ead` | `epan/dissectors/packet-dcom-dispatch.c` | dcom: Refactor post-decrementing while loops | wmem packet scope |
|  | `827fc3cf95` | `epan/dissectors/packet-dcom-oxid.c` | DCOM: More while loops post-decrementing the test | wmem packet scope |
|  | `5b864da7af` | `epan/dissectors/packet-dhcpv6.c` | DHCPv6: Change tests due to optlen being unsigned | wmem packet scope |
|  | `3356be2007` | `epan/dissectors/packet-dns.c` | DNS: Prevent UB in RFC 1876 Altitude calculation | wmem packet scope |
|  | `6130d850a0` | `epan/dissectors/packet-do-irp.c` | Replace tvb_new_subset calls with simpler versions without -1 | wmem packet scope |
|  | `b5b7bd0cd1` | `epan/dissectors/packet-gadu-gadu.c` | Gadu-Gadu: Prevent minor case of overflow | wmem packet scope |
|  | `63d3e0503f` | `epan/dissectors/packet-gtpv2.c` | GTPv2: Use a for loop | wmem packet scope |
|  | `aa963ace35` | `epan/dissectors/packet-ipmi-trace.c` | IPMI trace: Invalid millisecond value can lead to overflow | wmem packet scope |
|  | `8f265c37b5` | `epan/dissectors/packet-isis-lsp.c` | isis-lsp: fix Coverity 1604195 Overflowed constant | wmem packet scope |
|  | `55b2074d9e` | `epan/dissectors/packet-kafka.c` | Kafka: Fix some overflow UB | wmem packet scope |
|  | `0fb2e4acae` | `epan/dissectors/packet-knet.c` | kNet: Fix content length calculation when 4 bytes are used | wmem packet scope |
|  | `056adf306c` | `epan/dissectors/packet-lldp.c` | LLDP: Avoid overflow | wmem packet scope |
|  | `e16ec641b4` | `epan/dissectors/packet-lorawan.c` | LoRaWAN: Avoid overflow | wmem packet scope |
|  | `6829f67d17` | `epan/dissectors/packet-mip6.c` | Mobile IPv6: Update Mobile Node Identifier subtypes | wmem packet scope |
|  | `701b21aa2e` | `epan/dissectors/packet-mka.c` | MKA/MACsec: Handle ES and SC bit both unset | wmem packet scope |
|  | `4e917e2a7d` | `epan/dissectors/packet-mpeg-sect.c` | mpeg-sect: Fix overflow in packet_mpeg_sect_mjd_to_utc_time | wmem packet scope |
|  | `f840fef082` | `epan/dissectors/packet-netperfmeter.c` | NetPerfMeter: Use ENC_TIME_USECS for timestamp | wmem packet scope |
|  | `15461ffb6c` | `epan/dissectors/packet-nfs.c` | NFS: Use %u printf format for unsigned integers | wmem packet scope |
|  | `91fb2d9c3a` | `epan/dissectors/packet-nmf.c` | NMF: Prevent overflow in nmf_get_pdu_len | wmem packet scope |
|  | `eb0d6b4fe4` | `epan/dissectors/packet-opensafety.c` | opensafety: Avoid some technically UB signed overflow | wmem packet scope |
|  | `7db3ead8bb` | `epan/dissectors/packet-oran.c` | ORAN FH CUS: avoid potential overflow while working out usec timing delt | wmem packet scope |
|  | `7c21644082` | `epan/dissectors/packet-pdcp-lte.c` | PDCP-LTE,PDCP-NR: Fix possible overflow | wmem packet scope |
|  | `98789dcb85` | `epan/dissectors/packet-per.c` | PER: prevent UB | wmem packet scope |
|  | `9c1a18b281` | `epan/dissectors/packet-prp.c` | PRP: Fixing missing length check that breaks MACsec padding | wmem packet scope |
|  | `4f4ccae253` | `epan/dissectors/packet-pw-atm.c` | ATM PW: Convert tvb_new_subset_length_caplen | wmem packet scope |
|  | `8489dc2308` | `epan/dissectors/packet-rtps.c` | RTPS: Move prefix constant string inside static function | wmem packet scope |
|  | `f39c8d49df` | `epan/dissectors/packet-rtps.c` | RTPS: Make some fixed suffix and prefix strings const char * const | wmem packet scope |
|  | `976a4baffa` | `epan/dissectors/packet-rtps.c` | RTPS: Fix signed overflow | wmem packet scope |
|  | `88c378ef55` | `epan/dissectors/packet-slsk.c` | slsk: Fix signed overflow UB | wmem packet scope |
|  | `4f00d6092d` | `epan/dissectors/packet-smb2.c` | SMB2 Fix pipeline overfow using decode_smb2_name | wmem packet scope |
|  | `1d8acb21ab` | `epan/dissectors/packet-solaredge.c` | SolarEdge: Fix buffer overflow | wmem packet scope |
|  | `eecf92ebc8` | `epan/dissectors/packet-spdy.c` | SPDY: Fix potential reassembly overflow | wmem packet scope |
|  | `be847a28c5` | `epan/dissectors/packet-sua.c` | SUA: Fix trivial signed overflow | wmem packet scope |
|  | `c4635132c5` | `epan/dissectors/packet-tds.c` | TDS: Prevent signed overflow | wmem packet scope |
|  | `6fd3e40947` | `epan/dissectors/packet-tecmp.c` | TECMP: Fix integer overflow (OSS Fuzz issue 468513074) | wmem packet scope |
|  | `3cf60aa8f1` | `epan/dissectors/packet-thrift.c` | Thrift: Enforce varint maximum length | wmem packet scope |
|  | `5d7e27a07d` | `epan/dissectors/packet-tls-utils.c` | TLS-Utils: Check that ECH payload length is as long as the auth tag | wmem packet scope |
|  | `7a5fb37c35` | `epan/dissectors/packet-wccp.c` | WCCP: Use proto_item_set_end to avoid UB overflow | wmem packet scope |
|  | `30a0696193` | `epan/dissectors/packet-wccp.c` | WCCP: Fix possible underflow | wmem packet scope |
|  | `7bf3b7b8de` | `epan/dissectors/packet-winsrepl.c` | WINS Replication: Check for bytes existing | wmem packet scope |
|  | `c7ee0a061b` | `epan/dissectors/packet-wtp.c` | WTP: Use wmem_strbuf_t over sprintf | wmem packet scope |
|  | `9c615edd85` | `epan/dissectors/packet-x11.c` | X11: Make sure we don't overflow when stepping to the next message | wmem packet scope |
|  | `9a1d01e0a2` | `epan/dissectors/packet-x11.c` | X11: Check for overflow | wmem packet scope |
|  | `1e2ec8679e` | `epan/dissectors/packet-x11.c` | X11: Fix GenericExtension dissection | wmem packet scope |
|  | `34c9c6c530` | `epan/dissectors/packet-xtp.c` | XTP: Check reported length remaining when we use it | wmem packet scope |
|  | `030bf6ad01` | `epan/dissectors/packet-zbee-zcl-general.c` | ZigBee ZCL Touchlink: empty the commissioning map on a redissect. | wmem packet scope |
|  | `00fc37d700` | `epan/dissectors/packet-zbee-zcl.h` | Zigbee: Fix build with 32-bit time_t | wmem packet scope |
|  | `05940086cf` | `epan/asn1.c` | asn1: Use errno and expert items instead of DISSECTOR_ASSERT for get rea | epan/wsutil |
|  | `eed529e074` | `epan/column-utils.c` | column: Don't overflow when converting a large time to hms | epan/wsutil |
|  | `191277ff41` | `epan/dfilter/semcheck.c` | dfilter: fix NULL deref with FT_SCALAR in compatible_ftypes | epan/wsutil |
|  | `2cfe98b2d6` | `epan/packet_info.h` | epan: Add increment and decrement dissection depth by n functions | epan/wsutil |
|  | `6a2b72e43d` | `epan/proto.c` | epan: Avoid overflow calculating number of bytes required 7 bit encoding | epan/wsutil |
|  | `cc44c2cae4` | `epan/tvbuff.c` | epan: Use unsigned offsets in tvb_ensure_*_length_remaining | epan/wsutil |
|  | `dfce6e922f` | `epan/tvbuff.h` | epan: Use unsigned offsets with tvb_*_length_remaining | epan/wsutil |
|  | `3265de8a3e` | `wiretap/blf.c` | BLF: prevent overflow (cid1666424) | plain (wiretap) |
|  | `f207d25f4b` | `wiretap/libpcap.c` | wiretap: pcap[ng]: Don't let the reported length underflow w/ phdr | plain (wiretap) |
|  | `1dbe074013` | `wiretap/nettrace_3gpp_32_423.c` | nettrace_3gpp_32_423: Fix memory leaks and changeTime overflow bug | plain (wiretap) |
|  | `de719cc5ac` | `wiretap/peekclassic.c` | peekclassic: Fix possible signed overflow | plain (wiretap) |
|  | `76459b8134` | `wiretap/snoop.c` | snoop: Error before allocation if a pseudo-header size is too large | plain (wiretap) |
|  | `d660235e79` | `wsutil/nstime.c` | wsutil: Fix nstime_delta when the result and a are the same pointer | epan/wsutil |
|  | `af007c7894` | `wsutil/nstime.c` | nstime: Handle overflow in nstime_delta and nstime_sum | epan/wsutil |
|  | `9a6664beb7` | `wsutil/wmem/wmem_array.c` | wmem: Add another overflow check in wmem_array_grow | epan/wsutil |
|  | `70feff28d8` | `wsutil/wmem/wmem_array.c` | wmem: Check for overflow when growing wmem_array | epan/wsutil |
