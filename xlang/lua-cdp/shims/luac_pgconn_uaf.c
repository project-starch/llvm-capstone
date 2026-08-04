/* LuaDBI #35 — Lua statement userdata ⟷ libpq PGconn use-after-free.
 * Source: ../../luadbi-35/boundary.md. ASan: heap-use-after-free READ size 4,
 * 376 bytes inside a 1056-byte PGconn freed by PQfinish.
 *
 * Two allocations: the statement full userdata (dbd_postgresql_statement_create,
 * caches its own raw PGconn*) and the libpq PGconn.
 *   Free-site (connection.c:141): db:close() -> connection_close ->
 *     PQfinish(conn->postgresql) -> freePGconn frees the 1056-byte PGconn.
 *     connection_close nulls the CONNECTION's field but not the statement's copy.
 *   Stale-use (statement.c:145): stmt:execute() -> statement_execute reads the
 *     cached handle: PQstatus(statement->postgresql) -> reads the freed PGconn.
 * READ size 4 at OFFSET 376 -> interior address via cincoffset on the revoked
 * capability (assert-on-untagged FAULT route). Control: the read returns; MISS.
 */
#include "luac_shim.h"
#include <stdint.h>

#define PGCONN_BYTES 1056
#define PQSTATUS_OFF 376 /* the status field ASan names */

static volatile uint64_t sink;

int main(void) {
  unsigned char *pgconn = (unsigned char *)malloc(PGCONN_BYTES); /* PQconnect */
  if (!pgconn)
    abort();
  memset(pgconn, 0, PGCONN_BYTES);

  /* The statement userdata caches its own copy of the PGconn*. */
  unsigned char *stmt_postgresql = pgconn;

  free(pgconn); /* db:close -> PQfinish -> freePGconn -> REVOKE */

  /* stmt:execute -> PQstatus reads a status field at offset 376. */
  sink = *(volatile uint32_t *)(stmt_postgresql + PQSTATUS_OFF); /* statement.c:145 */

  mock_report("luac_pgconn_uaf", "use-after-free-survived");
  return 0;
}
