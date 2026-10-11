#ifndef WM_SHIM_WS_ASSERT_H
#define WM_SHIM_WS_ASSERT_H
#include "port.h"
/* Upstream's scope assertions stay active: an allocation outside a pool's
 * scope is a misuse this port must refuse, not optimize away. */
#define ws_assert(x) ((x) ? (void)0 : wm_fail(103))
#endif
