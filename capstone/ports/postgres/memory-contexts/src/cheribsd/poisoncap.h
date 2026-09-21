#ifndef PG_POISONCAP_H
#define PG_POISONCAP_H
/* Call once, before creating any context. Mode 0 has the identical layout
 * and allocation policy; mode 1 additionally invalidates retired storage. */
void pg_poisoncap_init(unsigned mode);
void pg_poisoncap_report(void);
#endif
