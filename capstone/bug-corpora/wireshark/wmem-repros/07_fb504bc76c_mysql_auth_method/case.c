#include "corpus.h"

/* The per-connection record, as far as this case needs it. */
struct mysql_conn_data {
  unsigned char *auth_method;
};

WM_CASE(7) {
/* Row 7 -- MySQL, #19045, fix fb504bc76c.
 * mysql_dissect_auth_switch_request reads the authentication plugin name
 * into pinfo->pool and stores the pointer in the connection record, which
 * lives in file scope. The pool is reset between packets; the AuthSwitch
 * response, a later packet, compares the stored name with strcmp. */
  struct mysql_conn_data *conn_data =
      wmem_alloc(wm_file_scope(), sizeof *conn_data); /* wmem_new0(wmem_file_scope()), packet-mysql.c:4043 */
  CHECK(conn_data, 1);
  unsigned char *name = wmem_alloc(wm_packet, 43); /* tvb_get_string_enc(pinfo->pool, ...), :3714 */
  CHECK(name, 2);
  memset(name, 0, 43);
  memcpy(name, "caching_sha2_password", 22);
  conn_data->auth_method = name;
  wm_next_packet(); /* epan_dissect_reset, epan.c:589 */
  /* The AuthSwitch response arrives and consults the connection record. */
  wm_held = conn_data->auth_method;
  wm_mark();
  (void)wm_probe(wm_held); /* strcmp in mysql_dissect_auth_switch_response, :3745 */
}
