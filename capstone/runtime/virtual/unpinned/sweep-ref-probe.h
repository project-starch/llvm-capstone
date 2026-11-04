/* Private ABI for a native-only page-reference test probe. */
#ifndef SWEEP_REF_PROBE_H
#define SWEEP_REF_PROBE_H
#ifdef __KERNEL__
#include <linux/ioctl.h>
#else
#include <sys/ioctl.h>
#endif
struct sweep_ref_sample { unsigned long address, references, mappings, pfn; };
#define SWEEP_REF_SAMPLE _IOWR(0xce, 8, struct sweep_ref_sample)
#endif
