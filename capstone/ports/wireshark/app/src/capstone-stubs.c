/* CAPSTONE tshark port, census build: couplings from whitelisted dissectors (or the core) to
 * dissectors the whitelist leaves out. Each stub is on a path the workload does not take; the
 * stock-vs-minimal output comparison is what checks that claim. */
#include "config.h"
#include <epan/packet.h>
#include "packet-dcerpc.h"
#include "packet-ppp.h"
#include "packet-tls-utils.h"
#include "packet-dtls.h"
#include "packet-http2.h"

void decode_dcerpc_reset_all(void) {}
const enum_val_t fcs_options[] = { {NULL, NULL, 0} };
tvbuff_t *decode_fcs(tvbuff_t *tvb, packet_info *pinfo _U_, proto_tree *fh_tree _U_, int fcs_decode _U_, int proto_offset _U_) { return tvb; }
void dtls_dissector_add(unsigned port _U_, dissector_handle_t handle _U_) {}
void ssl_dissector_add(unsigned port _U_, dissector_handle_t handle _U_) {}
void ssl_dissector_delete(unsigned port _U_, dissector_handle_t handle _U_) {}
ssl_master_key_map_t *tls_get_master_key_map(bool load_secrets _U_) { return NULL; }
uint32_t http2_get_stream_id(packet_info *pinfo _U_) { return 0; }
void dissect_http2_settings_ext(tvbuff_t *tvb _U_, packet_info *pinfo _U_, proto_tree *http2_tree _U_, unsigned offset _U_) {}

/* Tables that whitelisted dissectors LOOK UP but whose owners are left out. A missing table
 * crashes the lookup (dissector_try_* dereferences it); an EMPTY one answers "nothing
 * registered", which is what the stock build answers too for every key its owner does not
 * claim. Found by scanning the whitelist's find_dissector_table() calls against its
 * register_dissector_table() calls. */
void proto_register_capstone_stubs(void);
static int proto_capstone_stubs;
void proto_register_capstone_stubs(void)
{
	proto_capstone_stubs = proto_register_protocol("Minimal-build stub tables", "capstone-stubs", "capstone_stubs");
	register_dissector_table("osinl.incl", "OSI incl NLPID (empty)", proto_capstone_stubs, FT_UINT8, BASE_HEX);
	register_dissector_table("streaming_content_type", "HTTP2 streaming content (empty)", proto_capstone_stubs, FT_STRING, STRING_CASE_SENSITIVE);
	register_dissector_table("llc.hpteam_pid", "LLC HP OUI PID (empty)", proto_capstone_stubs, FT_UINT16, BASE_HEX);
}
