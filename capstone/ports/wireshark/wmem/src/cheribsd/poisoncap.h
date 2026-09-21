#ifndef WM_POISONCAP_H
#define WM_POISONCAP_H
/* Call once, before any pool exists. Mode 0 bounds every object exactly and
 * invalidates nothing; mode 1 additionally invalidates retired storage. */
void wm_poisoncap_init(unsigned mode);
void wm_poisoncap_report(void);
#endif
