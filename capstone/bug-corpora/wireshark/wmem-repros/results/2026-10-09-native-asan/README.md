# wireshark/wmem-repros under AddressSanitizer -- 2026-10-09

Predictions committed before the run: b64cd18b85d0. Runner: tools/run-native-asan.py, built by
runners/run-asan.sh (the port library itself carries the sanitizer).

Build: cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0; -fsanitize=address -fno-omit-frame-pointer -g -O0; wmem and the port built with the same flags
ASAN_OPTIONS: detect_leaks=0:abort_on_error=0:halt_on_error=1:color=never:quarantine_size_mb=1024

## Controls

    asan-control past 402653184: heap-buffer-overflow (required heap-buffer-overflow)
    asan-control uaf 402653184: heap-use-after-free (required heap-use-after-free)

## Cases

    00_3c8be14c82_rpcrdma_write_offsets_global                     fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    01_c14d731e45_cms_oid_global                                   fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    02_99da8c2cdc_mdb_address_column                               fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    03_6eab9f83ab_cola_info_column                                 fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    04_b48759e4a4_qnet6_col_set_str                                fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    05_5a109265a6_usbll_address_struct                             fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    06_a8b16d74e1_x509if_last_dn_static                            fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    07_fb504bc76c_mysql_auth_method                                fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    08_31ab1a0a17_sip_cseq_method                                  fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    09_693dc40936_geonw_proto_data_tvb                             fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    10_6fd3af5e99_t38_reassembly_buffer                            fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    11_3a5f82dfb5_http_header_map                                  fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    12_90bb3a5c9e_xml_root_name_recycled                           fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    13_0261fd7da6_http_range_cursor_past_chunk                     fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    14_1d8acb21ab_solaredge_payload_six_past                       fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    15_d24613c461_opcua_padding_below_chunk                        fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    16_e8ef9df09d_dcp_etsi_rs_parity_write                         fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    17_5a560f3f6a_dns_one_byte_write                               fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    18_716a200295_rtps_batch_sample_info_unguarded                 fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    19_4a4871a831_ntlmssp_blob_length_before_check                 fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    20_ed20250c13_proto_undecoded_bitmap_unbounded                 fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    21_69dac89280_tcp_flags_str_sixteen_bytes                      fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
