#ifndef CAPSTONE_SQLITE_HOSTCALL_H
#define CAPSTONE_SQLITE_HOSTCALL_H

typedef unsigned long long sqlite_hostcall_u64_t;
typedef long long sqlite_hostcall_s64_t;

struct sqlite_hostcall_v0 {
  sqlite_hostcall_u64_t phase;
  sqlite_hostcall_u64_t opcode;
  sqlite_hostcall_u64_t offset;
  sqlite_hostcall_u64_t length;
  sqlite_hostcall_s64_t result;
  sqlite_hostcall_s64_t error;
};

#define SQLITE_HC_RET_DONE 0UL

/* THE REGION SIZE IS A BUILD PARAMETER, and it is the ONE constant both halves must
 * agree on. The host creates and maps a region of this size; the domain bounds every
 * write by it. A mismatch is silent and destructive, so the host publishes the value it
 * actually used in `result` and the domain refuses to run if it disagrees -- see
 * SQLITE_HC_ERR_REGION_MISMATCH. Set it once, in the run script, for BOTH builds.
 *
 * The default stays 4096 deliberately: every board and QEMU result recorded so far was
 * measured against a 4 KiB build, and changing it here would silently perturb the image
 * those results describe. Raising it belongs to the build that consumes it.
 *
 * MEASURED CEILING (2026-08-21, by changing this one #define and running the real
 * workload): 4 KiB works, 1 MiB works with all five success markers, 64 MiB FAILS with
 * "map_region failed" and no markers at all. The 64 MiB arm is what makes the 1 MiB pass
 * meaningful -- without a failing arm, a pass is equally consistent with the constant
 * never reaching the build. Note it fails at MAP time, not at create time. */
#ifndef SQLITE_HC_REGION_SIZE
#define SQLITE_HC_REGION_SIZE 4096UL
#endif

#define SQLITE_HC_ANNOTATION_PERM_INOUT 0x1UL
#define SQLITE_HC_ANNOTATION_REV_SHARED 0x2UL

/* ------------------------------------------------------------ SQLLogicTest transport */
/* One call_dom per .test file: the host writes the whole file into the payload region and
 * the domain reads it in place. There is no streaming protocol and there cannot be one --
 * the host never dispatches on `opcode` mid-run and reads the payload only after the
 * domain returns (sqlite_host.c), and re-entering a domain re-runs BUILD_GP_CAPTABLE
 * (start-gp-captable-generic.S:30), which would destroy an in-memory database anyway.
 * SLT files are self-contained, so one file per entry is exactly the right granularity.
 *
 * "SLT\0" in ASCII, magic-guarded like the staged-probe selector it sits beside, so a
 * zeroed region is indistinguishable from today's behaviour and every existing build is
 * unaffected. It must not collide with the 0x5A6E00nn staged selector, and does not. */
#define SQLITE_HC_OP_SLT      0x534C5400UL
#define SQLITE_HC_OP_SLT_MASK 0xFFFFFF00UL

/* THE INPUT SITS IN THE TOP HALF, THE OUTPUT GROWS FROM ZERO. They share one region, so
 * they must not collide: the domain's output limit is lowered to this offset for SLT
 * builds, which is a compile-time constant swap and therefore costs the ordinary build
 * nothing. `offset` carries the input length. */
#define SQLITE_HC_SLT_INPUT_OFF (SQLITE_HC_REGION_SIZE / 2UL)
#define SQLITE_HC_SLT_MAX_INPUT (SQLITE_HC_REGION_SIZE - SQLITE_HC_SLT_INPUT_OFF)

/* Distinct return markers, so a failure to START is never confused with a clean run that
 * found nothing. Every one of these means NO records were evaluated. */
#define SQLITE_HC_ERR_REGION_MISMATCH 0x5117BAD0UL  /* host and domain disagree on size */
#define SQLITE_HC_ERR_BAD_INPUT       0x5117BAD1UL  /* absent or oversized input         */
#define SQLITE_HC_ERR_CONFIG_HEAP     0x5117BAD2UL  /* SQLITE_CONFIG_HEAP refused        */
#define SQLITE_HC_ERR_INITIALIZE      0x5117BAD3UL  /* sqlite3_initialize refused        */
#define SQLITE_HC_SLT_RAN             0x5117600DUL  /* the runner ran; read the payload  */

#endif
