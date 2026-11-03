/* Keep uid 0 for the experimental loader, but remove DAC bypass across exec. */
#include <linux/capability.h>
#include <linux/securebits.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    struct __user_cap_header_struct header = {_LINUX_CAPABILITY_VERSION_3, 0};
    struct __user_cap_data_struct data[2] = {{0}};
    if (argc == 3 && !strcmp(argv[2], "many-fds")) {
        alarm(20); /* Linux preserves this timer across exec. */
        execl(argv[1], argv[1], argv[2], (char *)0);
        return 126;
    }
    if (argc != 2 || prctl(PR_SET_SECUREBITS, SECBIT_NOROOT | SECBIT_NOROOT_LOCKED) ||
        syscall(SYS_capset, &header, data)) return 125;
    execl(argv[1], argv[1], "dac", (char *)0);
    return 126;
}
