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
 * memfd-launch text-source IMAGE OTHER: the module serves image text only from the
 * image file whose writers CV_IMAGE denied. Three CV_ADD requests for an image grant
 * whose first page is a private read+execute mapping of a file, each on a fresh device
 * handle: with IMAGE registered and the text mapped from IMAGE (admitted), with IMAGE
 * registered and the text mapped from OTHER, a different file (refused, EACCES), and
 * with no image registered (refused, EACCES). Prints one TEXT_SOURCE line; exits 0 only
 * when every answer is the expected one.
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
    return rc >= 0 ? "ok" : error == EBUSY ? "EBUSY" : error == ETXTBSY ? "ETXTBSY" :
           error == EACCES ? "EACCES" : strerror(error);
}

/* One CV_ADD for a 64 KiB image grant (reg 3): its first page a private read+execute mapping of
 * TEXT, the other fifteen private anonymous read+write. IMAGE, when not NULL, is registered with
 * CV_IMAGE first. Returns the ioctl's result; *error is its errno. */
static int add_image_grant(const char *image, const char *text, int *error)
{
    const unsigned long size = 0x10000, page = 0x1000;
    int device = open("/dev/capstone-vm", O_RDWR | O_CLOEXEC), tfd = open(text, O_RDONLY);
    if (device < 0 || tfd < 0) { perror("memfd-launch: open"); *error = errno; return -2; }
    int fd = -1;
    if (image) {
        fd = open(image, O_RDONLY);
        if (fd < 0 || ioctl(device, CV_IMAGE, &fd)) { perror("memfd-launch: CV_IMAGE"); *error = errno; return -2; }
    }
    char *raw = mmap(NULL, 2 * size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) { perror("memfd-launch: mmap"); *error = errno; return -2; }
    char *base = (char *)(((unsigned long)raw + size - 1) & ~(size - 1));
    if (mmap(base, page, PROT_READ | PROT_EXEC, MAP_PRIVATE | MAP_FIXED, tfd, 0) == MAP_FAILED ||
        mmap(base + page, size - page, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED) {
        perror("memfd-launch: mmap"); *error = errno; return -2;
    }
    madvise(base, size, MADV_NOHUGEPAGE);   /* the module requires it where huge pages exist */
    struct cv_map r = { .address = (unsigned long)base, .bytes = size, .permissions = 7, .reg = 3,
                        .cursor = (unsigned long)base, .rx_bytes = page };
    int rc = ioctl(device, CV_ADD, &r);
    *error = errno;
    close(device);
    close(tfd);
    if (fd >= 0) close(fd);
    munmap(raw, 2 * size);
    return rc;
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
    if (argc == 4 && !strcmp(argv[1], "text-source")) {
        int e1, e2, e3;
        int same = add_image_grant(argv[2], argv[2], &e1);
        int other = add_image_grant(argv[2], argv[3], &e2);
        int none = add_image_grant(NULL, argv[2], &e3);
        printf("TEXT_SOURCE image=%s other=%s noimage=%s\n", name(same, e1), name(other, e2), name(none, e3));
        return same == 0 && other < 0 && e2 == EACCES && none < 0 && e3 == EACCES ? 0 : 3;
    }
    if (argc != 4) {
        fprintf(stderr, "usage: memfd-launch open|write|sealed|mapped LAUNCHER IMAGE | image-twice IMAGE | text-source IMAGE OTHER\n");
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
