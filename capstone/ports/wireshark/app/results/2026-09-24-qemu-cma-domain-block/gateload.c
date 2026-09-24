/* capstone-test.c's create-and-call, except that a creation the module refused (dom_id -1)
   is reported and NOT called: calling domain -1 makes the monitor fault on its own table
   (ISSUES-ARCHIVE Q-01's shape), which ends the boot and every test queued after it. */
#include <stdio.h>
#include <stdlib.h>
#include "lib/libcapstone.h"
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gateload <dom>\n"); return 2; }
    int rc = capstone_init();
    if (rc) { fprintf(stderr, "capstone_init failed\n"); return rc; }
    dom_id_t dom_id = create_dom(argv[1], NULL);
    printf("Created domain ID = %lu\n", (unsigned long)dom_id);
    if (dom_id == (dom_id_t)-1) { printf("GATE REFUSED %s\n", argv[1]); capstone_cleanup(); return 3; }
    unsigned long r = call_dom(dom_id);
    printf("Called dom (1-th time) retval = %lu\n", r);
    capstone_cleanup();
    return 0;
}
