/* Platform setup shared by every client. Keep allocator use in ../*.c. */
#include "client.h"
#include "domain-runtime.h"
#ifdef PG_CLIENT_SUBLET
#include "pg_subpool.h"
#else
void pg_level0_init(void *, size_t);
#endif

static unsigned long arena_type;

_Noreturn void pg_subpool_refuse(const char *why) {
  fail(why);
  give_up(0xbadc1101);
}

void pg_domain_entry(unsigned *res, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    switch (shares++) {
    case 0:
      meta = (void *)res;
      break;
    case 1:
      payload = (void *)res;
      break;
    case 2:
#ifdef PG_CLIENT_SUBLET
      /* Consume the linear arena in the share handler, before storing it as
       * an ordinary C pointer could lose its linear authority. */
      arena_type = pg_subpool_arena(res, PG_REPLAY_ARENA_SIZE);
#else
      pg_level0_init(res, PG_REPLAY_ARENA_SIZE);
#endif
      break;
    default:
      break; /* the generic loader's input region is unused by these clients */
    }
    return;
  }
  domain_result = res;
  if (shares < 4 || !meta || !payload || arena_type != 0)
    give_up(0xbadc1102);
  meta->length = 0;
  pg_domain_payload((char *)payload, (unsigned long *)&meta->length,
                    PG_REPLAY_PAYLOAD_SIZE);

  MemoryContext root =
      AllocSetContextCreateInternal(NULL, "example root", 0, 2048, 8192);
  TopMemoryContext = CurrentMemoryContext = root;
  int result = client_run(root);
  MemoryContextDelete(root);
  TopMemoryContext = CurrentMemoryContext = NULL;
  pg_domain_text("PG_CLIENT " PG_CLIENT_NAME " RESULT ");
  pg_domain_uint(result);
  pg_domain_text("\n");
  if (result)
    give_up(0xbadc1103);
  *res = 0;
}
