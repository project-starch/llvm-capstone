#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "../../wire.h"
struct old_cv_map { unsigned long long words[9]; };
int main(void)
{
    struct old_cv_map old = {0};
    struct cv_remap remap = {0};
    int fd = open("/dev/capstone-vm", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    errno = 0;
    if (ioctl(fd, _IOWR('V', 16, struct old_cv_map), &old) != -1 || errno != ENOTTY)
        return 2;
    errno = 0;
    if (ioctl(fd, CV_REMAP_BEGIN, &remap) != -1 || errno != EOPNOTSUPP)
        return 3;
    errno = 0;
    if (ioctl(fd, CV_REMAP_END, &remap) != -1 || errno != EOPNOTSUPP)
        return 4;
    if (close(fd)) return 5;
    puts("COPYABLE_ABI_PASS old-grant-refused legacy-remap-refused");
    return 0;
}
