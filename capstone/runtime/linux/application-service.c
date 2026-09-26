#include "capstone/application-service.h"
#include "capstone/hostcall.h"
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int capstone_application_service(unsigned *mask, const struct hostcall_v0 *req,
                                 struct hostcall_v0 *res, char *payload) {
  if (req->opcode < CAPSTONE_APP_READ || req->opcode > CAPSTONE_APP_ISATTY)
    return -1;
  long result = -1;
  int error = EINVAL;
  struct capstone_app_fd_request arg;
  /* Snapshot before inspecting fields, including the request's payload. */
  memcpy(&arg, payload, sizeof arg);
  if (req->offset != sizeof arg ||
      req->length > HC_V0_REGION_SIZE - sizeof arg)
    goto done;
  if (arg.fd > 2 || !(*mask & (1u << arg.fd))) {
    error = EBADF;
    goto done;
  }
  int fd = (int)arg.fd;
  /* These mappings are PFNMAP on the guest: use ordinary Linux memory for
     I/O, preserving the existing file service's 9p bounce-buffer contract. */
  char bounce[HC_V0_REGION_SIZE];
  switch (req->opcode) {
  case CAPSTONE_APP_READ:
    result = read(fd, bounce, (size_t)req->length);
    if (result > 0)
      memcpy(payload + sizeof arg, bounce, (size_t)result);
    break;
  case CAPSTONE_APP_WRITE:
    memcpy(bounce, payload + sizeof arg, (size_t)req->length);
    result = write(fd, bounce, (size_t)req->length);
    break;
  case CAPSTONE_APP_CLOSE: {
    /* Close the inherited object (including its pipe endpoint), but reserve
       the number so later loader/service opens cannot steal fd 0, 1 or 2. */
    int nullfd = open("/dev/null", O_RDWR | O_CLOEXEC);
    if (nullfd >= 0) {
      result = dup2(nullfd, fd);
      int saved = errno;
      close(nullfd);
      errno = saved;
      if (result >= 0) {
        *mask &= ~(1u << fd);
        result = 0;
      }
    }
    break;
  }
  case CAPSTONE_APP_STAT: {
    struct stat st;
    result = fstat(fd, &st);
    if (!result) {
      struct capstone_app_stat out = {(uint64_t)st.st_size, (uint64_t)st.st_mode};
      memcpy(payload + sizeof arg, &out, sizeof out);
    }
    break;
  }
  case CAPSTONE_APP_FCNTL:
    if (arg.value != F_GETFL && arg.value != F_GETFD) {
      errno = ENOTSUP;
      break;
    }
    result = fcntl(fd, (int)arg.value);
    break;
  case CAPSTONE_APP_ISATTY:
    result = isatty(fd) ? 1 : 0;
    break;
  }
  error = result < 0 ? errno : 0;
done:
  res->result = result;
  res->error = error ? -error : 0;
  return 0;
}
