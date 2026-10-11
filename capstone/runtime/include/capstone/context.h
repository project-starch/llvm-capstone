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
 * own. Offsets in bytes. The entry glue calls the function in SLOT_START with
 * the capability in SLOT_ARG: the runtime's __capstone_context_run with the
 * start block itself, which installs the transport named in WORD_TRANSPORT
 * and then calls the application's function (SLOT_USER_START, SLOT_USER_ARG). */
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
#define CAPSTONE_CONTEXT_SLOT_USER_START 176
#define CAPSTONE_CONTEXT_SLOT_USER_ARG 192
#define CAPSTONE_CONTEXT_WORD_TRANSPORT 208
/* 224 to 255 are free; context-probe's entry audit parks two registers there. */
#define CAPSTONE_CONTEXT_START_BYTES 256

/* A seal region must hold at least 33 capabilities (QEMU CAP_SEALED_SIZE_MIN);
 * the synchronous context uses the first 88 bytes. */
#define CAPSTONE_CONTEXT_SEAL_BYTES 1024

#ifndef __ASSEMBLER__
#include <stddef.h>
#include <capstone/capability-slot.h>

struct capstone_context_event;

struct capstone_context {
  capstone_cap_slot handle;        /* revocation handle over the whole area */
  capstone_cap_slot seal;          /* the minted seal until it is handed off */
  unsigned long id;                /* (generation << 32) | slot, once adopted */
  volatile unsigned long *done;    /* completion word, valid until revoke */
  volatile unsigned long *value;   /* the start function's return value */
  unsigned long *start;            /* start block alias */
  void *tp;
  unsigned long area_base, area_bytes;
  unsigned long stack_base, stack_top;
  /* capstone_context_mint_split only: handles over the start block, the TLS
     block and the stack, each junior to `handle` (probe use, A4). */
  capstone_cap_slot child[3];
};

/* Take area_bytes from the context arena and mint a context whose first
 * entry calls start(arg) on its own stack and TLS block. 0 on success;
 * -1 when the arena cannot supply the area or the area is too small. */
int capstone_context_mint(struct capstone_context *c, size_t area_bytes,
                          unsigned long (*start)(void *), void *arg);

/* As capstone_context_mint, and keep a handle over each of the start block,
 * the TLS block and the stack, so that they can be revoked while the seal
 * stays valid (capstone_context_revoke_children). */
int capstone_context_mint_split(struct capstone_context *c, size_t area_bytes,
                                unsigned long (*start)(void *), void *arg);
void capstone_context_revoke_children(struct capstone_context *c);

/* As capstone_context_mint, with the seal's mstatus/privilege word and mie
 * word given (probe use, A12: what a supervised first entry does with them). */
int capstone_context_mint_words(struct capstone_context *c, size_t area_bytes,
                                unsigned long (*start)(void *), void *arg,
                                unsigned long mstatus, unsigned long mie);

/* As capstone_context_mint, with the seal's first pc given (probe use, A2: an
 * instrumented entry that continues at the runtime's own). */
int capstone_context_mint_entry(struct capstone_context *c, size_t area_bytes,
                                unsigned long (*start)(void *), void *arg, void *entry);

/* Revoke the parent handle: every capability derived from the area dies, the
 * seal included, wherever it is held. The area stays in c->handle for reuse. */
void capstone_context_revoke(struct capstone_context *c);

/* Mint again into the area a previous revoke returned. */
int capstone_context_remint(struct capstone_context *c,
                            unsigned long (*start)(void *), void *arg);

/* Offer the minted seal through the current call's descriptor and ask the
 * launcher to register it (CAPSTONE_CONTEXT_REGISTER) or to run it on a
 * launcher thread of its own (CAPSTONE_CONTEXT_THREAD). A THREAD context gets
 * a transport of its own, reserved and written into its start block before
 * the request, so its first entry can already make delegated calls; a
 * REGISTER context has none. The seal leaves c->seal either way. Returns the
 * context id, or -errno: EAGAIN (every transport in use), ESTALE, ENOENT,
 * ENOSPC (no slot), EINVAL (no descriptor in this entry). On an error the
 * caller revokes the area. */
long capstone_context_create(struct capstone_context *c, unsigned mode);

/* Step a registered context once from this context's launcher thread. */
long capstone_context_step(unsigned long id, struct capstone_context_event *event);

/* Remove a context's registration. */
long capstone_context_forget(unsigned long id);

/* Leave the current context for good (see start-musl.S). */
void __capstone_context_exit(unsigned long value) __attribute__((noreturn));

/* Enter a minted context with a nested, unsupervised CALL from this context
 * (probe use). Returns when it yields or exits; the seal comes back. */
void __capstone_context_call(capstone_cap_slot *seal, unsigned long request,
                             unsigned long *result);
#endif

#endif
