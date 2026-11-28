/* qpdec: qpdiff plus SPEC's own ldecod (the JM reference decoder, a plain build) on both streams, and both streams
 * dumped whole in hex. Original header follows.
 * qpdiff: qpinit at --frames 1 --qp 26, then the first byte where the two streams differ, with 96 bytes of each
 * around it in hex and ASCII, and the count of differing bytes. Original header follows.
 * qpinit: runtime's x264 question. Encode the first 10 BuckBunny frames with B5TL1's x264 (.dom) and Plain-matched's,
 * once with x264's default rate control (SPEC's own mode) and once at a fixed --qp 26. Prints each output's size and an
 * FNV-1a 64 hash. Everything runs from /w, since the .dom loader confines a program to its cwd. */
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sched.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/reboot.h>
static void hash(const char *f) {
    FILE *p = fopen(f, "rb"); uint64_t h = 1469598103934665603ULL; long n = 0; int c;
    if (!p) { printf("QP %s missing\n", f); return; }
    while ((c = fgetc(p)) != EOF) { h ^= (unsigned char)c; h *= 1099511628211ULL; n++; }
    fclose(p); printf("QP %s bytes %ld fnv %016llx\n", f, n, (unsigned long long)h);
}
static void run(char *const argv[]) {
    printf("QP run %s %s %s\n", argv[0], argv[1], argv[2]); fflush(stdout);
    pid_t pid = fork();
    if (!pid) { execv(argv[0], argv); perror("QP exec"); _exit(127); }
    int st = 0; while (waitpid(pid, &st, WNOHANG) == 0) sched_yield();
    printf("QP end %s status 0x%x\n", argv[0], st); fflush(stdout);
}
static void dump(const char *f, long at) {
    FILE *p = fopen(f, "rb"); unsigned char b[96]; long from = at > 48 ? at - 48 : 0;
    fseek(p, from, SEEK_SET); size_t n = fread(b, 1, 96, p); fclose(p);
    printf("QPD %s @%ld hex ", f, from); for (size_t i = 0; i < n; i++) printf("%02x", b[i]);
    printf("\nQPD %s @%ld asc ", f, from); for (size_t i = 0; i < n; i++) putchar(b[i] >= 32 && b[i] < 127 ? b[i] : '.');
    printf("\n");
}
static void cmp(const char *x, const char *y) {
    FILE *a = fopen(x, "rb"), *b = fopen(y, "rb"); long i = 0, first = -1, ndiff = 0; int c, d;
    if (!a || !b) { printf("QPD missing\n"); return; }
    for (;; i++) { c = fgetc(a); d = fgetc(b); if (c == EOF && d == EOF) break; if (c != d) { ndiff++; if (first < 0) first = i; } if (c == EOF || d == EOF) { printf("QPD length differs at %ld\n", i); break; } }
    fclose(a); fclose(b); printf("QPD %s vs %s first diff %ld, %ld bytes differ in the overlap\n", x, y, first, ndiff);
    if (first >= 0) { dump(x, first); dump(y, first); }
}
static void hexall(const char *f) {
    FILE *p = fopen(f, "rb"); int c; long n = 0; if (!p) return;
    printf("QPHEX %s ", f); while ((c = fgetc(p)) != EOF) { printf("%02x", c); n++; } printf("\nQPHEX-END %s %ld\n", f, n); fclose(p);
}
int main(void) {
    if (mkdir("/w", 0755) || chdir("/w") || rename("/BB10.yuv", "/w/BB10.yuv")) { perror("QP setup"); return 1; }
    char *c[] = {"/x264.dom", "--qp", "26", "--frames", "1", "-o", "cap-qp1.264", "BB10.yuv", "1280x720", 0};
    char *d[] = {"/x264.plain", "--qp", "26", "--frames", "1", "-o", "plain-qp1.264", "BB10.yuv", "1280x720", 0};
    run(c); run(d);
    hash("cap-qp1.264"); hash("plain-qp1.264"); cmp("cap-qp1.264", "plain-qp1.264");
    hexall("cap-qp1.264"); hexall("plain-qp1.264");
    char *e[] = {"/ldecod", "-i", "cap-qp1.264", "-o", "cap-dec.yuv", 0};
    char *f[] = {"/ldecod", "-i", "plain-qp1.264", "-o", "plain-dec.yuv", 0};
    run(e); run(f); hash("cap-dec.yuv"); hash("plain-dec.yuv");
    printf("QP done\n"); fflush(stdout); sync(); reboot(RB_POWER_OFF); return 0;
}
