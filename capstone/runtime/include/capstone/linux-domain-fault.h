#ifndef CAPSTONE_LINUX_DOMAIN_FAULT_H
#define CAPSTONE_LINUX_DOMAIN_FAULT_H

#include "capstone/domain-fault.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Linux launcher policy, called after call_dom(), outside a signal handler.
 * For ordinary results this returns without invoking cleanup. A fault invokes
 * the optional cleanup callback once and
 * terminates the calling PROCESS with SIGSEGV (even if ignored or blocked).
 * Join application workers before calling; cleanup must return. Reporting is
 * caller-owned: this policy performs no stdout/stderr I/O. This does not
 * destroy monitor domains or reclaim their regions. A supervisor that wants to
 * stay alive can instead inspect CAPSTONE_DOMAIN_FAULT_RETVAL itself.
 */
void capstone_domain_exit_on_fault(unsigned long result,
                                   void (*cleanup)(void *), void *context);

#ifdef __cplusplus
}
#endif
#endif
