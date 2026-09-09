# SQLite C API boundary classification

Source: `sqlite3.h` (SQLite 3.53.3). Directions: `H->E` host to engine, `E->H` engine to host, `E<->H` bidirectional/handle, `CB` callback, `N/A` no boundary pointer.

| Function (base) | Group | Direction | Obligation | Primitive | Notes/variants |
|---|---|---|---|---|---|
| sqlite3_activate_cerod | configuration | H->E | activation string valid for the call | L | Only declared with SQLITE_ENABLE_CEROD |
| sqlite3_bind_* | bind | H->E | input value or buffer remains valid according to STATIC/TRANSIENT/destructor policy | L,R | blob/blob64, double, int/int64, null, text/text16/text64, value, zeroblob/zeroblob64; pointer is separate |
| sqlite3_bind_parameter_index | bind metadata | H->E | parameter-name string valid for the call | L | Immediate lookup |
| sqlite3_carray_bind[_v2] | carray | H->E | array and optional destructor context remain valid until release/reset/finalize | L,R | v2 separates destructor argument from array pointer |
| sqlite3_compileoption_used | compile options | H->E | option-name string valid for the call | L | Immediate lookup |
| sqlite3_complete[16] | SQL parser | H->E | SQL text valid for the call | L | UTF-8 and UTF-16 variants |
| sqlite3_declare_vtab | virtual table | H->E | schema SQL valid for the call | L | Called during virtual-table setup |
| sqlite3_deserialize | serialization | H->E | database image obeys FREEONCLOSE/RESIZE/READONLY ownership flags | L,R,H | Engine may retain and later free or resize the supplied image |
| sqlite3_drop_modules | virtual table | H->E | NULL-terminated module-name array and strings valid for the call | L | Registration effects persist; input list is immediate |
| sqlite3_exec | SQL execution | H->E | SQL text valid through the call; synchronous row callback arguments are callback-scoped | L,R | xCallback is invoked synchronously and not retained; secondary CB behavior |
| sqlite3_keyword_check | keyword | H->E | input bytes valid for the call | L | Length-delimited keyword lookup |
| sqlite3_log | logging | H->E | format string and referenced arguments valid for the call | L | Immediate formatting |
| sqlite3_overload_function | virtual table | H->E | function-name string valid for the call | L | Name is consumed during registration |
| sqlite3_result_* | UDF result | H->E | result buffers/values obey STATIC/TRANSIENT/destructor policy until SQLite consumes them | L,R | blob/blob64, double, error/error16/error_code/error_nomem/error_toobig, int/int64, null, text/text16/text16be/text16le/text64, value, zeroblob/zeroblob64, subtype; pointer is separate |
| sqlite3_set_errmsg | error | H->E | error-message string valid for the call | L | SQLite copies the message |
| sqlite3_str_append* | string builder | H->E | input format/string bytes valid for the call | L | append, appendall, appendchar, appendf, vappendf |
| sqlite3_stricmp / sqlite3_strnicmp / sqlite3_strglob / sqlite3_strlike | string comparison | H->E | input strings valid for the call | L | Immediate comparison/pattern matching |
| sqlite3_win32_set_directory* | platform configuration | H->E | directory string valid through configuration update | L | base, 8, and 16 variants; Windows only |
| sqlite3_bind_parameter_name | bind metadata | E->H | borrowed name valid while statement remains valid | R,H | Returned pointer is statement-owned |
| sqlite3_column_blob / sqlite3_column_text* / sqlite3_column_value | column data | E->H | borrowed row data invalid after step/reset/finalize or incompatible conversion | R | blob, text/text16, and unprotected sqlite3_value |
| sqlite3_column_name* / sqlite3_column_decltype* / sqlite3_column_database_name* / sqlite3_column_table_name* / sqlite3_column_origin_name* | column metadata | E->H | borrowed metadata invalid after statement finalization, automatic reprepare, or schema change | R,H | UTF-8 and UTF-16 name/decltype/origin variants |
| sqlite3_compileoption_get / sqlite3_keyword_name | static metadata | E->H | borrowed process-lifetime bytes must not be freed | R | keyword_name also returns an explicit byte length |
| sqlite3_context_db_handle / sqlite3_db_handle / sqlite3_db_mutex / sqlite3_database_file_object | handle lookup | E->H | borrowed parent-owned handle cannot outlive its callback, statement, connection, filename, or open-file owner | H,R | Returns sqlite3, mutex, or sqlite3_file handles without new ownership |
| sqlite3_db_filename / sqlite3_db_name / sqlite3_filename_database / sqlite3_filename_journal / sqlite3_filename_wal / sqlite3_uri_parameter / sqlite3_uri_key | name and URI metadata | E->H | borrowed string/filename metadata cannot outlive its connection or filename owner | R,H | Includes database names, VFS filename components, URI parameters, and keys |
| sqlite3_errmsg[16] / sqlite3_errstr | error | E->H | borrowed message invalidated by later API activity or library lifetime | R | errmsg UTF-8/UTF-16; errstr is static |
| sqlite3_get_auxdata / sqlite3_get_clientdata / sqlite3_user_data | stored context | E->H | borrowed context valid only until replacement, destruction, or callback/registration lifetime ends | R,H | Auxdata is expression-scoped; clientdata and user_data follow their registrations |
| sqlite3_mprintf / sqlite3_vmprintf / sqlite3_str_finish | allocated strings | E->H | allocated string ownership transfers to host and requires sqlite3_free | L | str_finish also consumes its builder |
| sqlite3_next_stmt | statement enumeration | E->H | borrowed statement handle invalidated by finalize or parent close | H,R | Returned handle is not newly owned |
| sqlite3_preupdate_old / sqlite3_preupdate_new | preupdate values | E->H | borrowed sqlite3_value valid only during current preupdate callback | R | Only valid from the registered preupdate hook |
| sqlite3_randomness | random output | E->H | host output buffer must be writable for the call; unreadable until initialized | U | Engine fills exactly N bytes |
| sqlite3_serialize | serialization | E->H | returned image is borrowed or host-owned according to NOCOPY and must follow sqlite3_free rules | L,R | Output size pointer is initialized by the call |
| sqlite3_sql / sqlite3_expanded_sql / sqlite3_normalized_sql | statement SQL | E->H | borrowed SQL follows statement lifetime; expanded SQL transfers allocated ownership | R,L,H | expanded_sql requires sqlite3_free; normalized_sql is compile-option dependent |
| sqlite3_status[64] / sqlite3_db_status[64] / sqlite3_stmt_scanstatus[_v2] | status output | E->H | output storage must be writable and is initialized by the call | U | Scanstatus is compile-option dependent and may return borrowed text through pOut |
| sqlite3_str_value | string builder | E->H | borrowed buffer invalidated by builder mutation, reset, finish, or free | R,H | Do not free returned pointer |
| sqlite3_table_column_metadata | schema metadata | E->H | borrowed output strings follow connection/schema lifetime; scalar outputs require writable storage | R,U,H | Inputs are immediate table/column names |
| sqlite3_value_blob / sqlite3_value_text* | value data | E->H | borrowed bytes invalid after callback ends or a converting value accessor invalidates them | R | text, text16, text16be, text16le variants |
| sqlite3_vfs_find | VFS registry | E->H | borrowed VFS pointer remains valid only while registration remains active | R,H | NULL name selects default VFS |
| sqlite3_vtab_collation | virtual table | E->H | borrowed collation name valid for current xBestIndex processing | R | sqlite3_index_info-scoped |
| sqlite3_vtab_in_first / sqlite3_vtab_in_next / sqlite3_vtab_rhs_value | virtual table values | E->H | borrowed sqlite3_value output valid only in the current virtual-table planning/filter scope | R | Output value pointers are engine-owned |
| sqlite3_aggregate_context | UDF aggregate | E<->H | engine-owned mutable allocation is initialized on demand and invalid after aggregate completion | H,U | First nonzero-size call allocates callback-scoped state |
| sqlite3_backup_* | backup handle | E<->H | backup child handle depends on both connections and is invalid after finish | H | init, step, finish, remaining, pagecount |
| sqlite3_bind_pointer / sqlite3_result_pointer / sqlite3_value_pointer | typed pointer | E<->H | typed pointer and destructor remain valid through binding/result lifetime and type-matched retrieval | L,R,H | Raw host pointer is threaded through SQL values |
| sqlite3_blob_* | incremental blob | E<->H | blob handle is parented by connection/transaction and buffers are valid for each read/write | H,U,L | open, reopen, close, bytes, read, write |
| sqlite3_close / sqlite3_close_v2 | connection lifecycle | E<->H | connection and descendants become invalid according to immediate or zombie-close semantics | H,R | close_v2 defers destruction until child resources finish |
| sqlite3_create_filename / sqlite3_free_filename | VFS filename | E<->H | constructed filename owns copied components and must be freed exactly once | L,H | Filename may parent URI/component borrows |
| sqlite3_file_control | VFS control | E<->H | opaque argument layout/lifetime is opcode-specific and may carry data both ways | L,R,U | Most lifetime-sensitive direction depends on op |
| sqlite3_get_table / sqlite3_free_table | query table | E<->H | allocated result table transfers to host and must be released exactly once | L | Result contains an engine-allocated pointer array and strings |
| sqlite3_load_extension | extension loading | E<->H | input names are call-scoped; optional engine-allocated error string transfers to host | L,U | Error output must be released with sqlite3_free |
| sqlite3_malloc* / sqlite3_realloc* / sqlite3_free / sqlite3_msize | allocator | E<->H | allocated objects have exclusive ownership and must follow SQLite allocator pairing | L,U | malloc/malloc64, realloc/realloc64, free, msize |
| sqlite3_mutex_* | mutex lifecycle | E<->H | dynamic mutexes require balanced use and one free; static mutexes are borrowed and must not be freed | L,H | alloc, free, enter, try, leave, held, notheld |
| sqlite3_open* | connection lifecycle | E<->H | output connection handle must be checked, then closed exactly once after all children | H,U | open, open16, open_v2 |
| sqlite3_prepare* | statement lifecycle | E<->H | SQL input is call-scoped; output statement is parented by connection and requires finalize | H,U | prepare/v2/v3 and UTF-16 variants |
| sqlite3_snapshot_* | snapshot lifecycle | E<->H | snapshot allocation is freed exactly once and remains compatible with its database/WAL lineage | L,H | get, open, free, cmp, recover; compile-option dependent |
| sqlite3_snprintf / sqlite3_vsnprintf | formatting buffer | E<->H | host buffer is writable for the call and becomes initialized output | U,L | Format and referenced inputs are call-scoped |
| sqlite3_step / sqlite3_reset / sqlite3_finalize / sqlite3_clear_bindings | statement lifecycle | E<->H | operation advances or ends row/binding borrows; finalize destroys statement exactly once | H,R | step/reset invalidate column borrows; clear_bindings releases retained inputs |
| sqlite3_str_new / sqlite3_str_free / sqlite3_str_reset / sqlite3_str_truncate | string builder | E<->H | builder handle is parented by optional connection and invalid after finish/free | H | Mutation invalidates sqlite3_str_value borrows |
| sqlite3_value_dup / sqlite3_value_free | value lifecycle | E<->H | duplicated protected value transfers to host and must be freed exactly once | L,H | dup result is independent of callback-scoped source |
| sqlite3_wal_checkpoint[_v2] | WAL control | E<->H | schema input is call-scoped and optional output counters require writable storage | L,U | v2 initializes log/checkpoint counts |
| sqlite3_auto_extension / sqlite3_cancel_auto_extension / sqlite3_reset_auto_extension | extension callback | CB | registered entry point remains callable until cancellation/reset | L,R,S | Process-global callback registry |
| sqlite3_autovacuum_pages | autovacuum callback | CB | callback context and destructor remain valid until replacement or connection close | L,R,H,S | Engine stores and later invokes callback |
| sqlite3_busy_handler | busy callback | CB | callback context remains valid until replacement or connection close | L,R,H,S | Engine stores callback; busy_timeout replaces it |
| sqlite3_collation_needed[16] | collation callback | CB | callback context remains valid until replacement or connection close | L,R,H,S | UTF-8/UTF-16 requested-name callback variants |
| sqlite3_commit_hook / sqlite3_rollback_hook / sqlite3_update_hook / sqlite3_preupdate_hook / sqlite3_wal_hook | database hooks | CB | callback context remains valid until replacement/removal or connection close | L,R,H,S | Transaction, row-change, preupdate, and WAL hook families |
| sqlite3_create_collation / sqlite3_create_collation16 / sqlite3_create_collation_v2 | collation callback | CB | comparison callback/context remain valid while registered; v2 destructor runs on release | L,R,H,S | UTF-8/UTF-16 and destructor variant |
| sqlite3_create_function / sqlite3_create_function16 / sqlite3_create_function_v2 / sqlite3_create_window_function | UDF callback | CB | callbacks, pApp, and optional destructor remain valid while registered | L,R,H,S | Scalar, aggregate, and window callback families |
| sqlite3_create_module[_v2] | virtual-table callback | CB | module method table/context remain valid while registered; v2 destructor owns release | L,R,H,S | Engine retains a callback table |
| sqlite3_progress_handler | progress callback | CB | callback context remains valid until replacement/removal or connection close | L,R,H,S | Engine stores and repeatedly invokes callback |
| sqlite3_rtree_geometry_callback / sqlite3_rtree_query_callback | R-tree callback | CB | callback/context and optional destructor remain valid while SQL function is registered | L,R,H,S | Conditional R-tree APIs |
| sqlite3_set_authorizer | authorizer callback | CB | callback context remains valid until replacement/removal or connection close | L,R,H,S | Invoked during statement preparation |
| sqlite3_set_auxdata / sqlite3_set_clientdata | stored-data destructor | CB | stored pointer and destructor remain valid until replacement, scoped release, or connection close | L,R,H,S | Auxdata is expression-scoped; clientdata is named connection-scoped storage |
| sqlite3_trace_v2 | trace callback | CB | callback context remains valid until replacement/removal or connection close | L,R,H,S | Replaces deprecated trace/profile APIs |
| sqlite3_unlock_notify | unlock callback | CB | callback arguments remain valid until invocation or cancellation by connection close | L,R,H,S | May invoke callback before registration call returns |
| sqlite3_vfs_register / sqlite3_vfs_unregister | VFS callback table | CB | VFS object and method table remain valid for all registrations and open files | L,R,H,S | Unregister does not invalidate already-open file objects |
| sqlite3_bind_parameter_count / sqlite3_column_count / sqlite3_data_count / sqlite3_column_double / sqlite3_column_int[64] / sqlite3_column_bytes[16] / sqlite3_column_type / sqlite3_uri_boolean / sqlite3_uri_int64 / sqlite3_value_double / sqlite3_value_int[64] / sqlite3_value_bytes[16] / sqlite3_value_type / sqlite3_value_numeric_type / sqlite3_value_nochange / sqlite3_value_frombind / sqlite3_value_encoding / sqlite3_value_subtype / sqlite3_vtab_on_conflict / sqlite3_vtab_nochange / sqlite3_vtab_distinct / sqlite3_vtab_in / sqlite3_preupdate_count / sqlite3_preupdate_depth / sqlite3_preupdate_blobwrite / sqlite3_str_errcode / sqlite3_str_length | scalar accessors | N/A |  |  | Value-only statement, column, URI, value, preupdate, string-builder, and virtual-table queries |
| sqlite3_busy_timeout / sqlite3_setlk_timeout / sqlite3_config / sqlite3_db_config / sqlite3_test_control / sqlite3_vtab_config / sqlite3_enable_load_extension / sqlite3_enable_shared_cache / sqlite3_limit / sqlite3_wal_autocheckpoint | configuration | N/A |  |  | Administrative/value-only settings; variadic opcodes that carry pointers require separate opcode-level analysis |
| sqlite3_changes[64] / sqlite3_total_changes[64] / sqlite3_last_insert_rowid / sqlite3_set_last_insert_rowid / sqlite3_db_cacheflush / sqlite3_db_release_memory / sqlite3_db_readonly / sqlite3_txn_state / sqlite3_get_autocommit / sqlite3_interrupt / sqlite3_is_interrupted | connection state | N/A |  |  | Value-only connection operations, counters, and state |
| sqlite3_errcode / sqlite3_extended_errcode / sqlite3_error_offset / sqlite3_extended_result_codes / sqlite3_system_errno | error status | N/A |  |  | Value-only error queries/configuration |
| sqlite3_initialize / sqlite3_shutdown / sqlite3_os_init / sqlite3_os_end | library lifecycle | N/A |  |  | No application pointer transfer |
| sqlite3_keyword_count / sqlite3_libversion / sqlite3_sourceid / sqlite3_libversion_number / sqlite3_threadsafe | library metadata | N/A |  |  | Version/source strings are immutable process-lifetime constants; remaining functions return scalar metadata |
| sqlite3_memory_used / sqlite3_memory_highwater / sqlite3_release_memory / sqlite3_soft_heap_limit64 / sqlite3_hard_heap_limit64 | memory administration | N/A |  |  | Value-only allocator administration |
| sqlite3_sleep | timing | N/A |  |  | Value-only delay |
| sqlite3_stmt_readonly / sqlite3_stmt_isexplain / sqlite3_stmt_explain / sqlite3_stmt_busy / sqlite3_stmt_status / sqlite3_stmt_scanstatus_reset | statement administration | N/A |  |  | Value-only state/query/reset operations |

## Summary

Counts per direction: H->E = 18, E->H = 21, E<->H = 19, CB = 15, N/A = 9. Total rows = 82.

Nine declarations marked `SQLITE_DEPRECATED` were skipped: `sqlite3_trace`, `sqlite3_profile`, `sqlite3_aggregate_count`, `sqlite3_expired`, `sqlite3_transfer_bindings`, `sqlite3_global_recover`, `sqlite3_thread_cleanup`, `sqlite3_memory_alarm`, and `sqlite3_soft_heap_limit`. SQLite 3.53.3 does not attach `SQLITE_EXPERIMENTAL` to any function declaration; compile-option-gated public APIs are retained and identified in Notes.

## Coverage

All SQLite APIs exercised or named by the 18 applicable Stage 1 reproducers are represented: `sqlite3_progress_handler`, `sqlite3_create_function`, `sqlite3_column_name`, `sqlite3_open`, `sqlite3_close`, `sqlite3_close_v2`, `sqlite3_exec`, `sqlite3_prepare_v2`, `sqlite3_step`, `sqlite3_reset`, `sqlite3_finalize`, `sqlite3_backup_init`, `sqlite3_backup_finish`, `sqlite3_set_authorizer`, `sqlite3_bind_text`, `sqlite3_column_type`, `sqlite3_column_text`, `sqlite3_user_data`, `sqlite3_result_int`, `sqlite3_free`, and `sqlite3_errstr`. Missing entries: none.
