#ifndef CAPSTONE_DOMAIN_FAULT_H
#define CAPSTONE_DOMAIN_FAULT_H

/* Reserved result of a cooperating domain's fault trampoline. This value is
 * also used by the monitor's separate S-mode fault-return path. Keep it usable
 * from preprocessed assembly as well as both host and domain C. */
#define CAPSTONE_DOMAIN_FAULT_RETVAL 0x0FA017ED

#endif
