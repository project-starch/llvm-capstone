#ifndef CAPSTONE_APPLICATION_SERVICE_H
#define CAPSTONE_APPLICATION_SERVICE_H

#include <stdint.h>

/* Application ABI v1 extensions. Existing HostCall v0 opcodes are unchanged.
 * Requests put this header before data at offset 16; metadata.length bounds
 * the data. fd is restricted to the three inherited standard descriptors. */
#define CAPSTONE_APP_READ 0x100u
#define CAPSTONE_APP_WRITE 0x101u
#define CAPSTONE_APP_CLOSE 0x102u
#define CAPSTONE_APP_STAT 0x103u
#define CAPSTONE_APP_FCNTL 0x104u
#define CAPSTONE_APP_ISATTY 0x105u

struct capstone_app_fd_request { uint64_t fd, value; };
struct capstone_app_stat { uint64_t size, mode; };

struct hostcall_v0;
int capstone_application_service(unsigned *stdio_mask,
    const struct hostcall_v0 *request, struct hostcall_v0 *response, char *payload);

#endif
