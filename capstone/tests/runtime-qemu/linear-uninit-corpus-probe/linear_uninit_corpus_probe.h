#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_LINEAR_UNINIT_CORPUS_PROBE_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_LINEAR_UNINIT_CORPUS_PROBE_H

/* Constants shared by the controller (built with Buildroot gcc, no capability
 * builtins) and the domain payloads (built with the Capstone clang). Keep this
 * header free of builtins and of anything capability-typed.
 *
 * Corpus rows closed here:
 *   row14 cpython_uninit_connection (UNINIT, use-before-init)
 *   row11 go_double_finalize        (LINEAR, double-free)
 *
 * The region size, the DPI codes and the grant annotations are the same ones the
 * held-cap probe uses, because these probes reuse its receive protocol verbatim
 * (../intra-domain-mrev-revoke-probe/probe_domain.h). They are restated rather
 * than included so the controller does not have to pull in a header whose
 * comment block is about a different probe.
 */

#define CORPUS_REGION_SIZE 4096UL
#define CORPUS_OFFSET 8u
#define CORPUS_SENTINEL 0x5Eu

/* Where the arena is split to carve the "statement handle" sub-capability. */
#define CORPUS_SPLIT_MID 2048UL

/* shared_region_annotated(): PERM_INOUT (RW) + REV_TRANSFERRED.
 * RW because cstighten silently delinearises a read-only linear capability, and
 * csmrev then asserts on the non-linear input. TRANSFERRED because the domain
 * must own the arena outright: the whole lifecycle is intra-domain. */
#define CORPUS_ANNOTATION_PERM_INOUT 0x1u
#define CORPUS_ANNOTATION_REV_TRANSFERRED 0x3u

/* Return codes. A domain that reaches its *res store did NOT fault; the fault
 * probes therefore encode "the trap never happened" and the run script treats
 * any retval from them as a failure. */

/* row14 -- UNINIT */
#define CORPUS_RET_UNINIT_NOTRAP 0x14100000u    /* pre-init load returned      */
#define CORPUS_RET_UNINIT_NEG_NOTRAP 0x14110000u /* pre-init load at imm<0 returned */
#define CORPUS_RET_INIT_OK 0x1412005eu          /* post-init use works         */

/* row11 -- LINEAR */
#define CORPUS_RET_DROP_NOTRAP 0x11100000u      /* use after drop returned     */
#define CORPUS_RET_DOUBLE_DROP_NOTRAP 0x11110000u /* second drop returned      */
#define CORPUS_RET_NO_DROP_OK 0x11120033u       /* control: no drop, use works */
#define CORPUS_RET_DROP_SIBLING_OK 0x11130044u  /* arena survives the drop     */

#endif
