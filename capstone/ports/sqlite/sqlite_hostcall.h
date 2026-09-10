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
 * never reaching the build. Note it fails at MAP time, not at create time.
 *
 * NARROWED 2026-09-10 (bench lane, measured with `parsenumber` so that a failure is the
 * REGION and not the workload; recorded here by the board lane because their branch is
 * not yet pushable -- take THIS block wholesale on merge rather than resolving it):
 *
 *     two 1 MiB regions   clean end to end
 *     two 4 MiB regions   clean end to end
 *     two 8 MiB regions   map_region FAILED, after SQ: D/mapped, before entry
 *
 * So the bracket is 4 MiB works / 8 MiB fails, closing a 63 MiB gap that had stood since
 * August. WHAT THIS STILL DOES NOT ESTABLISH, and the old note did not say it either: the
 * host creates **TWO** regions of this size, so the passing arm is two 4 MiB regions and
 * the failing arm two 8 MiB ones. **Whether the limit is per-region or on the total is
 * NOT established by these arms** -- that needs a run with two DIFFERENT sizes, which
 * nobody has done. Do not quote "4 MiB per region" from this. */
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

/* ------------------------------------------------------------ feature-set probe transport */
/* "FEA\0", the same magic-guarded shape as the SLT opcode beside it and non-colliding with it
 * or with the 0x5A6E00nn staged selector. It asks the domain to call the six SQLite APIs that
 * SQLITE_FEATURE_SET=restored puts back and report how many were compiled in.
 *
 * WHY A PROBE AT ALL. The SLT corpus cannot tell the two feature sets apart -- none of its files
 * uses a restored feature, so both builds produce identical tallies, which is equally consistent
 * with "the switch is inert". check-feature-set.sh settles that natively by looking at exported
 * symbols; this settles it INSIDE THE DOMAIN, on silicon, where the symbol table is not
 * observable. */
#define SQLITE_HC_OP_FEATURE      0x46454100UL
#define SQLITE_HC_OP_FEATURE_MASK 0xFFFFFF00UL

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
