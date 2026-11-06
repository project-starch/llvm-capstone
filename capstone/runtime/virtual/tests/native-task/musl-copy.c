/* SPDX-License-Identifier: MIT */
/* Real musl read/write workload. Setup, output validation and diagnostics are
 * outside the profile window; every case copies the same resident 1 MiB. */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    const size_t total = 1024 * 1024;
    if (argc != 2) return 2;
    size_t chunk = strtoul(argv[1], 0, 10);
    if (chunk != 64 && chunk != 4096 && chunk != 65536) return 2;
    unsigned char *source = malloc(total), *result = malloc(total), *buffer = malloc(chunk);
    if (!source || !result || !buffer) return 3;
    for (size_t i = 0; i < total; ++i) source[i] = (i * 17 + i / 251) % 251;
    memset(result, 0, total);
    memset(buffer, 0, chunk);
    int input = open("copy-input", O_CREAT | O_TRUNC | O_RDWR, 0600);
    int output = open("copy-output", O_CREAT | O_TRUNC | O_RDWR, 0600);
    if (input < 0 || output < 0) return 4;
    /* Both files are populated first so the measured loop uses resident pages. */
    if (write(input, source, total) != (ssize_t)total ||
        write(output, result, total) != (ssize_t)total ||
        lseek(input, 0, SEEK_SET) || lseek(output, 0, SEEK_SET)) return 5;
    unsigned failures = 0;
    __asm__ volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory");
    for (size_t offset = 0; offset < total; offset += chunk) {
        failures += read(input, buffer, chunk) != (ssize_t)chunk;
        failures += write(output, buffer, chunk) != (ssize_t)chunk;
    }
    __asm__ volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory");
    failures += lseek(input, 0, SEEK_CUR) != (off_t)total;
    failures += lseek(output, 0, SEEK_END) != (off_t)total;
    failures += lseek(output, 0, SEEK_SET) != 0;
    failures += read(output, result, total) != (ssize_t)total;
    failures += memcmp(source, result, total) != 0;
    failures += close(input) != 0;
    failures += close(output) != 0;
    printf("MUSL_COPY bytes=%zu chunk=%zu calls=%zu failures=%u\n",
           total, chunk, 2 * (total / chunk), failures);
    free(buffer); free(result); free(source);
    return failures != 0;
}
