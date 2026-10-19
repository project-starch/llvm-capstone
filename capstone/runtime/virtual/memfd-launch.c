/* memfd-launch MODE LAUNCHER IMAGE: run LAUNCHER on a memfd copy of IMAGE
 * ("child" argument), keeping the memfd's writable descriptor open meanwhile.
 *   open    no seals: anyone with that descriptor can still write the image
 *   write   F_SEAL_WRITE only: the image can still grow or shrink
 *   sealed  F_SEAL_WRITE, F_SEAL_GROW and F_SEAL_SHRINK: the image is fixed
 * The launcher must refuse the first two (ETXTBSY) and run the third.
 *   mapped  run LAUNCHER on IMAGE itself while this process holds a shared
 *           writable mapping of it (the descriptor closed): a writer the
 *           module's denial must count, so the launcher must refuse (ETXTBSY).
 * memfd-launch image-twice IMAGE: open the device, deny writers to IMAGE
 * (CV_IMAGE), ask again (EBUSY), open IMAGE for writing while denied (ETXTBSY)
 * and after the device is closed (allowed); prints one IMAGE_TWICE line and
 * exits 0 only when every answer is the expected one.
 * A native helper for run-exec.py; the launch modes exit with the launcher's
 * status. */
#define _GNU_SOURCE
#include "wire.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

static int launch(const char *launcher, const char *image)
{
    pid_t pid = fork();
    if (pid == 0) {
        execl(launcher, launcher, image, "child", (char *)NULL);
        _exit(127);
    }
    int status;
    if (pid < 0 || waitpid(pid, &status, 0) != pid) return 2;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

static const char *name(int rc, int error)
{
    return rc >= 0 ? "ok" : error == EBUSY ? "EBUSY" : error == ETXTBSY ? "ETXTBSY" : strerror(error);
}

int main(int argc, char **argv)
{
    if (argc == 3 && !strcmp(argv[1], "image-twice")) {
        int device = open("/dev/capstone-vm", O_RDWR | O_CLOEXEC), fd = open(argv[2], O_RDONLY);
        if (device < 0 || fd < 0) { perror("memfd-launch: open"); return 2; }
        int first = ioctl(device, CV_IMAGE, &fd), first_error = errno;
        int second = ioctl(device, CV_IMAGE, &fd), second_error = errno;
        int held = open(argv[2], O_WRONLY), held_error = errno;
        if (held >= 0) close(held);
        close(device);
        int released = open(argv[2], O_WRONLY), released_error = errno;
        if (released >= 0) close(released);
        printf("IMAGE_TWICE first=%s second=%s held=%s released=%s\n", name(first, first_error),
               name(second, second_error), name(held, held_error), name(released, released_error));
        return first == 0 && second < 0 && second_error == EBUSY &&
               held < 0 && held_error == ETXTBSY && released >= 0 ? 0 : 3;
    }
    if (argc != 4) {
        fprintf(stderr, "usage: memfd-launch open|write|sealed|mapped LAUNCHER IMAGE | image-twice IMAGE\n");
        return 2;
    }
    if (!strcmp(argv[1], "mapped")) {
        int fd = open(argv[3], O_RDWR);
        if (fd < 0) { perror("memfd-launch: open"); return 2; }
        /* The mapping keeps the file open for writing after the descriptor
         * is gone: the writer the launch must see. */
        if (mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) == MAP_FAILED) {
            perror("memfd-launch: mmap"); return 2;
        }
        close(fd);
        return launch(argv[2], argv[3]);
    }
    int seals = !strcmp(argv[1], "open") ? 0 : !strcmp(argv[1], "write") ? F_SEAL_WRITE :
                !strcmp(argv[1], "sealed") ? F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK : -1;
    int src = open(argv[3], O_RDONLY), fd = memfd_create("image", MFD_ALLOW_SEALING);
    if (seals < 0 || src < 0 || fd < 0) { perror("memfd-launch"); return 2; }
    char buf[65536];
    ssize_t n;
    while ((n = read(src, buf, sizeof buf)) > 0)
        if (write(fd, buf, n) != n) { perror("memfd-launch: write"); return 2; }
    if (n < 0 || (seals && fcntl(fd, F_ADD_SEALS, seals))) { perror("memfd-launch: seal"); return 2; }
    char path[64];
    snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
    return launch(argv[2], path);
}
