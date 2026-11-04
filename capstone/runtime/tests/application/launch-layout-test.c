/* Exercise the actual application preparation code, including its count guards. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef CAPSTONE_APPLICATION_SOURCE
#define CAPSTONE_APPLICATION_SOURCE "../../domain/application.c"
#endif
#define main launch_probe_entry
#include CAPSTONE_APPLICATION_SOURCE
#undef main

static unsigned expected_argc, expected_envc;
int launch_probe_entry(int argc, char **argv)
{
    assert((unsigned)argc == expected_argc && argv[argc] == NULL);
    for (unsigned i = 0; i < expected_argc; ++i)
        assert(!strcmp(argv[i], i ? "argument" : "program"));
    char **env = __capstone_domain_environ();
    for (unsigned i = 0; i < expected_envc; ++i)
        assert(env[i] && !strcmp(env[i], "E=value"));
    assert(env[expected_envc] == NULL);
    return 0;
}

int main(int argc, char **argv)
{
    assert(argc == 2);
    const char *mode = argv[1];
    static char block[CAPSTONE_LAUNCH_BYTES];
    static char *args[CAPSTONE_LAUNCH_STRINGS + 1];
    static char *env[CAPSTONE_LAUNCH_STRINGS + 1];
    struct capstone_launch_task task = {0};
    expected_argc = !strcmp(mode, "max-args") ? CAPSTONE_LAUNCH_STRINGS :
                    !strcmp(mode, "mixed") ? 511 : 1;
    expected_envc = !strcmp(mode, "max-args") ? 0 :
                    !strcmp(mode, "mixed") ? 513 :
                    !strcmp(mode, "max-env") ? CAPSTONE_LAUNCH_STRINGS - 1 : 1;
    for (unsigned i = 0; i < expected_argc; ++i) args[i] = i ? "argument" : "program";
    for (unsigned i = 0; i < expected_envc; ++i) env[i] = "E=value";
    assert(__capstone_launch_task() == NULL && __capstone_domain_environ()[0] == NULL);
    assert(!capstone_launch_pack(block, sizeof block, (int)expected_argc, args, env, "/", 7, &task));
    struct capstone_launch_header *h = (void *)block;
    size_t bytes = h->bytes;
    int bad = 1;
    if (!strcmp(mode, "bad-argc")) h->argc = UINT32_MAX;
    else if (!strcmp(mode, "bad-envc")) h->envc = CAPSTONE_LAUNCH_STRINGS;
    else if (!strcmp(mode, "bad-offset")) {
        uint32_t off = h->bytes;
        memcpy(block + sizeof *h + sizeof off, &off, sizeof off);
    } else if (!strcmp(mode, "truncated")) --bytes;
    else bad = 0;
    int rc = __capstone_application_prepare(block, bytes);
    if (bad) {
        assert(rc == EINVAL && __capstone_launch_task() == NULL);
        assert(__capstone_domain_environ()[0] == NULL);
    } else {
        assert(rc == 0 && capstone_main() == 0);
        assert(__capstone_application_prepare(block, bytes) == EINVAL);
    }
    printf("launch-layout %s PASS\n", mode);
    return 0;
}
