#ifndef CAPSTONE_CONTEXT_H
#define CAPSTONE_CONTEXT_H

/* Execution contexts minted by the domain runtime
 * (docs/plans/delegation-threads.md).
 *
 * A context is one sealed continuation over a thread area the runtime takes
 * from its context arena: a seal region, a start block, a TLS block and a
 * stack. The parent handle stays outside the area, in the caller's
 * struct capstone_context; revoking it ends every capability derived from the
 * area, the seal included. The constants below are shared with the entry glue
 * (musl-capstone runtime/start-musl.S). */

/* The word a context writes into its result slot once it has exited. */
#define CAPSTONE_CONTEXT_EXITED 0x4558495445444558

/* mstatus word of a minted seal: (3 << 38) | (2 << 34), the privilege and
 * XLEN fields create_domain writes for every domain it builds. A literal, so
 * that C and the assembler read the same 64-bit value. */
#define CAPSTONE_CONTEXT_MSTATUS 0xC800000000

/* Start block: the recovery-block slots of the entry glue, then the context's
 * own. Offsets in bytes. */
#define CAPSTONE_CONTEXT_SLOT_RETURN 0
#define CAPSTONE_CONTEXT_SLOT_RESULT 16
#define CAPSTONE_CONTEXT_SLOT_GP 32
#define CAPSTONE_CONTEXT_SLOT_SP 48
#define CAPSTONE_CONTEXT_SLOT_REQUEST 64
#define CAPSTONE_CONTEXT_SLOT_SUSPENDED_SP 80
#define CAPSTONE_CONTEXT_SLOT_TP 96
#define CAPSTONE_CONTEXT_SLOT_START 112
#define CAPSTONE_CONTEXT_SLOT_ARG 128
#define CAPSTONE_CONTEXT_WORD_DONE 160
#define CAPSTONE_CONTEXT_WORD_VALUE 168
#define CAPSTONE_CONTEXT_START_BYTES 256

/* A seal region must hold at least 33 capabilities (QEMU CAP_SEALED_SIZE_MIN);
 * the synchronous context uses the first 88 bytes. */
#define CAPSTONE_CONTEXT_SEAL_BYTES 1024

#ifndef __ASSEMBLER__
#include <stddef.h>
#include <capstone/capability-slot.h>

struct capstone_context {
  capstone_cap_slot handle;        /* revocation handle over the whole area */
  capstone_cap_slot seal;          /* the minted seal until it is handed off */
  volatile unsigned long *done;    /* completion word, valid until revoke */
  volatile unsigned long *value;   /* the start function's return value */
  unsigned long *start;            /* start block alias */
  void *tp;
  unsigned long area_base, area_bytes;
  unsigned long stack_base, stack_top;
};

/* Take area_bytes from the context arena and mint a context whose first
 * entry calls start(arg) on its own stack and TLS block. 0 on success;
 * -1 when the arena cannot supply the area or the area is too small. */
int capstone_context_mint(struct capstone_context *c, size_t area_bytes,
                          unsigned long (*start)(void *), void *arg);

/* Revoke the parent handle: every capability derived from the area dies, the
 * seal included, wherever it is held. The area stays in c->handle for reuse. */
void capstone_context_revoke(struct capstone_context *c);

/* Mint again into the area a previous revoke returned. */
int capstone_context_remint(struct capstone_context *c,
                            unsigned long (*start)(void *), void *arg);

/* Leave the current context for good (see start-musl.S). */
void __capstone_context_exit(unsigned long value) __attribute__((noreturn));

/* Enter a minted context with a nested, unsupervised CALL from this context
 * (probe use). Returns when it yields or exits; the seal comes back. */
void __capstone_context_call(capstone_cap_slot *seal, unsigned long request,
                             unsigned long *result);
#endif

#endif
