/* Minimal reproducer attempt: zend_hash_init_ex alone, without the rest of the engine.
 *
 * Bisected from rung B: stages 1-2 of zend_startup pass, and stage 3 fails at the FIRST
 * zend_hash_init_ex. The call consumes ~2.65 MB of stack and faults before calloc is ever
 * entered (proven with an unconditional fault-channel probe inside calloc). Frames involved
 * are tiny: _zend_hash_init_ex 160 B, _zend_hash_init 192 B, domain_main 144 B, and
 * _zend_hash_init is straight-line code plus one while loop and one calloc. A NULL
 * destructor behaves identically, so it is not the cast function pointer either.
 *
 * If this reproduces with only zend_hash.o linked, the repro is ~200 lines and belongs in
 * capstone/tests/compiler-repros/. If it does NOT, the trigger is image- or layout-
 * dependent (gp-captable / capability-global init), which is itself the finding.
 */
#include <zend.h>

void *malloc(unsigned long);

void domain_main(unsigned *res, unsigned func)
{
    (void)func;
    *res = 0x1u;                                  /* entered */

    HashTable *ht = (HashTable *) malloc(sizeof(HashTable));
    if (!ht) { *res = 0xE1u; return; }
    *res = 0x2u;                                  /* allocation survived */

    if (zend_hash_init_ex(ht, 100, NULL, NULL, 1, 0) != SUCCESS) { *res = 0xE2u; return; }
    *res = 0x3u;                                  /* the call returned */

    if (ht->nTableSize == 128) { *res = 0x7Eu; }  /* and did the right thing */
}
