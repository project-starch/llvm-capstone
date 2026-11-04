/* Private ABI for the isolated qualification fixture, native RV64 only. */
#ifndef ONQ_NAMESPACE_PROBE_H
#define ONQ_NAMESPACE_PROBE_H
#ifdef __KERNEL__
#include <linux/ioctl.h>
#else
#include <sys/ioctl.h>
#endif
/* address: the page; offset: the granule (16-byte aligned) within it that is seeded or read raw. */
struct onq_namespace_sample { unsigned long address, tag, low, high, frame, offset; };
/* Use encoded requests: small integers collide with VFS ioctls (2 is FIGETBSZ). */
#define ONQ_NS_SEED   _IOWR(0xce, 1, struct onq_namespace_sample)
#define ONQ_NS_SAMPLE _IOWR(0xce, 2, struct onq_namespace_sample)
#define ONQ_NS_CLEAR  _IOWR(0xce, 3, struct onq_namespace_sample)
#endif
