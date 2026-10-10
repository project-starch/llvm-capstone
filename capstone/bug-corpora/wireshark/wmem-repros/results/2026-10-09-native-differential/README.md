# wmem-repros native fix differential -- 2026-10-09

Pre-registered at b64cd18b85d0 (not blind: the cases were developed against an exploratory build).
Runner: runners/run-native.sh -- hosted build, upstream wmem allocator (WM_CHUNKS=OFF).

    22 of 22 two-sided: buggy VERDICT DEFECT-REPRODUCED rc=0, fixed VERDICT FIXED rc=0

    00_3c8be14c82_rpcrdma_write_offsets_global buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    01_c14d731e45_cms_oid_global buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    02_99da8c2cdc_mdb_address_column buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    03_6eab9f83ab_cola_info_column buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    04_b48759e4a4_qnet6_col_set_str buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    05_5a109265a6_usbll_address_struct buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    06_a8b16d74e1_x509if_last_dn_static buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    07_fb504bc76c_mysql_auth_method buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    08_31ab1a0a17_sip_cseq_method buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    09_693dc40936_geonw_proto_data_tvb buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    10_6fd3af5e99_t38_reassembly_buffer buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    11_3a5f82dfb5_http_header_map buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    12_90bb3a5c9e_xml_root_name_recycled buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    13_0261fd7da6_http_range_cursor_past_chunk buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    14_1d8acb21ab_solaredge_payload_six_past buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    15_d24613c461_opcua_padding_below_chunk buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    16_e8ef9df09d_dcp_etsi_rs_parity_write buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    17_5a560f3f6a_dns_one_byte_write buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    18_716a200295_rtps_batch_sample_info_unguarded buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    19_4a4871a831_ntlmssp_blob_length_before_check buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    20_ed20250c13_proto_undecoded_bitmap_unbounded buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
    21_69dac89280_tcp_flags_str_sixteen_bytes buggy rc=0 VERDICT DEFECT-REPRODUCED | fixed rc=0 VERDICT FIXED
