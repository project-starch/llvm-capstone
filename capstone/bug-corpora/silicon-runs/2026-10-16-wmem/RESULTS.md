# wireshark's wmem corpus on silicon: results (2026-10-16, caplifive_sublet_d93a8d66b)

Pre-registered in README.md (lane 8bb5542ed567, before the boot). One boot, 07:43-08:26:
- firmware 6cbb3a4494fcdc73, Image bded55b35f656a8b, on the line of sets 2-4;
- two arms, 106 runs, every one ended by itself (each arm's rung returned 0).

**Every row and every control is as pre-registered,** by outcome, cause and site, on the pre-registered image
(sha256): 26 of 26 on each arm (23 cases, 3 wmem controls) and the four tool controls of each. Every fixed run
printed VERDICT FIXED and exited 0.

| arm | 14 temporal | 9 spatial | wmem controls (uaf-chunk / uaf-jumbo / bounds-chunk) |
|---|---|---|---|
| wmem-silicon-wm-stock (wmem as released) | 14 MISSED, the stale chunk reoccupied before the read | 9 MISSED | completes / CAUGHT 25 / completes |
| wmem-silicon-wm-sublet (patch 0001) | **14 CAUGHT 25**: 13 at the probe, 22 (a double free) in its declared wmem_block_sublet_free | **9 CAUGHT 28** at the probe | CAUGHT 25 / CAUGHT 25 / CAUGHT 28 |

**What it shows:**
- **Stock wmem defeats per-object malloc protection on silicon.** Its packet pool keeps its first 2 MiB block across
  the reset and hands the same chunks to the next packet. Every stale reference and every overrun stays inside that
  one malloc'd block: 23 MISSED.
  - Only the jumbo, the one object wmem returns to `g_free`, is caught (control 01). It needs level0's placement of
    a 2 MiB block on its granule, which gives the block a lifetime.
- **Patch 0001 catches all 23 on silicon,** temporal and spatial alike. Every block and block_fast object is a
  CDERIVE child of its block, bounded to the request and CREVOKEd at free, at realloc, and at the reset that ends
  its pool. The reset revokes a block's generation and derives a new one, at packet rate, without a node exhaustion
  (30) or a wedge.

## Per case

| case | silicon-wm-stock | silicon-wm-sublet |
|---|---|---|
| 00_3c8be14c82_rpcrdma_write_offsets_global | MISSED | CAUGHT 25 |
| 01_c14d731e45_cms_oid_global | MISSED | CAUGHT 25 |
| 02_99da8c2cdc_mdb_address_column | MISSED | CAUGHT 25 |
| 03_6eab9f83ab_cola_info_column | MISSED | CAUGHT 25 |
| 04_b48759e4a4_qnet6_col_set_str | MISSED | CAUGHT 25 |
| 05_5a109265a6_usbll_address_struct | MISSED | CAUGHT 25 |
| 06_a8b16d74e1_x509if_last_dn_static | MISSED | CAUGHT 25 |
| 07_fb504bc76c_mysql_auth_method | MISSED | CAUGHT 25 |
| 08_31ab1a0a17_sip_cseq_method | MISSED | CAUGHT 25 |
| 09_693dc40936_geonw_proto_data_tvb | MISSED | CAUGHT 25 |
| 10_6fd3af5e99_t38_reassembly_buffer | MISSED | CAUGHT 25 |
| 11_3a5f82dfb5_http_header_map | MISSED | CAUGHT 25 |
| 12_90bb3a5c9e_xml_root_name_recycled | MISSED | CAUGHT 25 |
| 13_0261fd7da6_http_range_cursor_past_chunk | MISSED | CAUGHT 28 |
| 14_1d8acb21ab_solaredge_payload_six_past | MISSED | CAUGHT 28 |
| 15_d24613c461_opcua_padding_below_chunk | MISSED | CAUGHT 28 |
| 16_e8ef9df09d_dcp_etsi_rs_parity_write | MISSED | CAUGHT 28 |
| 17_5a560f3f6a_dns_one_byte_write | MISSED | CAUGHT 28 |
| 18_716a200295_rtps_batch_sample_info_unguarded | MISSED | CAUGHT 28 |
| 19_4a4871a831_ntlmssp_blob_length_before_check | MISSED | CAUGHT 28 |
| 20_ed20250c13_proto_undecoded_bitmap_unbounded | MISSED | CAUGHT 28 |
| 21_69dac89280_tcp_flags_str_sixteen_bytes | MISSED | CAUGHT 28 |
| 22_c702b44a01_usbhid_output_usages_freed_twice | MISSED | CAUGHT 25 in wmem_block_sublet_free |
| 00_control_uaf_chunk | MISSED | CAUGHT 25 |
| 01_control_uaf_jumbo | CAUGHT 25 | CAUGHT 25 |
| 02_control_bounds_chunk | MISSED | CAUGHT 28 |

Rows 00-12 and 22 are the temporal cases and 13-21 the spatial ones (the corpus README). A CAUGHT with no site is at
the case's labelled probe (`wm_probe` / `wm_write_probe`).

**Judged by** `tools/run-silicon-cases.py board --prereg` on the run-scoped transcript, then compared row by row with
the pre-registered outcome, cause and site, keyed by image hash.
