/* execve between protected programs.
 *
 * Started without EXEC_SMOKE_PARENT, the program runs three refused calls and
 * continues after each one:
 *   - execve of a plain ELF (/bin/busybox) fails with ENOEXEC, because a
 *     protected program may exec only another .dom image;
 *   - execve with an untagged argv fails with EFAULT;
 *   - execve with an argv reference that covers only its first cell fails
 *     with EFAULT, although the next cell in memory holds the terminator.
 * It then execs its own image with the arguments "alpha" and "beta" and the
 * one environment variable EXEC_SMOKE_PARENT, which carries the number of
 * live lifetime tables before the exec. The new image prints its arguments and
 * its environment and checks that the kernel released the old image's table.
 * It then runs its image once more through fexecve. musl tries execveat, which
 * the kernel refuses with ENOSYS, and falls back to execve of /proc/self/fd/N.
 * That image exits 0. Each check that fails prints its line and exits nonzero. */
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <capstone/capability.h>

#define PARENT_VARIABLE "EXEC_SMOKE_PARENT"
#define PLAIN_ELF "/bin/busybox"
#define COUNTERS "/sys/module/sublet/parameters/"
#define NO_COUNTER ((unsigned long)-1)

extern char **environ;

/* The environment of the three refused calls. An image started by one of them
 * reports the leak instead of running the parent again. */
static char leak_variable[] = PARENT_VARIABLE "=leak";
static char *leak_environment[] = { leak_variable, 0 };

static int run_parent(const char *self);
static int run_child(int argc, char **argv, const char *parent);
static int refuse_plain_elf(void);
static int refuse_untagged_argv(const char *self);
static int refuse_short_argv(const char *self);
static int exec_child(const char *self);
static int exec_by_descriptor(const char *self);
static unsigned long read_counter(const char *name);
static void *narrow(void *object, unsigned long size);
static unsigned long query(void *reference, int selector);
static void say(const char *format, ...);
static void append(char *line, unsigned long size, const char *text);
static void append_number(char *line, unsigned long size, unsigned long value, int negative);

int main(int argc, char **argv)
{
    const char *parent = getenv(PARENT_VARIABLE);

    if (parent)
        return run_child(argc, argv, parent);
    return run_parent(argv[0]);
}

static int run_parent(const char *self)
{
    if (refuse_plain_elf())
        return 1;
    if (refuse_untagged_argv(self))
        return 2;
    if (refuse_short_argv(self))
        return 3;
    return exec_child(self);
}

/* The new image: the arguments and the environment arrive as written, and the
 * number of live lifetime tables is the parent's, so the old table is gone. */
static int run_child(int argc, char **argv, const char *parent)
{
    if (!strcmp(parent, "leak")) {
        say("exec-smoke: FAIL, a refused execve started the new image\n");
        return 20;
    }
    if (!strcmp(parent, "fexecve")) {
        say("exec-smoke: fexecve ok, argc %d\n", argc);
        return argc != 1;
    }

    say("exec-smoke child: argc %d, argv[1] %s, argv[2] %s, environ[0] %s\n",
        argc, argc > 1 ? argv[1] : "-", argc > 2 ? argv[2] : "-", environ[0]);
    if (argc != 3 || strcmp(argv[1], "alpha") || strcmp(argv[2], "beta"))
        return 21;
    if (!environ[0] || strncmp(environ[0], PARENT_VARIABLE "=", sizeof PARENT_VARIABLE) ||
        environ[1])
        return 22;

    unsigned long before = strtoul(parent, 0, 10);
    unsigned long after = read_counter("live_contexts");
    unsigned long pages = read_counter("table_pages");
    say("exec-smoke child: live lifetime tables %lu before execve, %lu after, %lu table pages\n",
        before, after, pages);
    if (after == NO_COUNTER || after != before)
        return 23;

    say("exec-smoke: execve ok\n");
    return exec_by_descriptor(argv[0]);
}

static int refuse_plain_elf(void)
{
    char *args[] = { "busybox", "false", 0 };

    errno = 0;
    int result = execve(PLAIN_ELF, args, leak_environment);
    int error = errno;
    say("exec-smoke: plain ELF %s refused, execve %d, errno %d (%s)\n",
        PLAIN_ELF, result, error, strerror(error));
    return result != -1 || error != ENOEXEC;
}

/* The argv array holds a well-formed list, but the argument names it by its
 * address only. The kernel finds no reference behind it. */
static int refuse_untagged_argv(const char *self)
{
    char *args[] = { (char *)self, "untagged", 0 };
    volatile unsigned long address = (unsigned long)args;
    char **untagged = (char **)address;

    if (query(untagged, 1) != CAPSTONE_CAP_EMPTY) {
        say("exec-smoke: setup, the untagged argv kept its tag\n");
        return 1;
    }

    errno = 0;
    int result = execve(self, untagged, leak_environment);
    int error = errno;
    say("exec-smoke: untagged argv refused, execve %d, errno %d (%s)\n",
        result, error, strerror(error));
    return result != -1 || error != EFAULT;
}

/* The argv array holds a well-formed list of one argument and its terminator,
 * but the reference covers only the first cell. The kernel must not read the
 * terminator through it. */
static int refuse_short_argv(const char *self)
{
    char **cells = malloc(2 * sizeof *cells);

    if (!cells || query(cells, 1) != CAPSTONE_CAP_NONLINEAR) {
        say("exec-smoke: setup, no non-linear argv object\n");
        return 1;
    }
    cells[0] = (char *)self;
    cells[1] = 0;
    char **first = narrow(cells, sizeof *cells);
    if (query(first, 1) != CAPSTONE_CAP_NONLINEAR ||
        query(first, 4) - query(first, 3) != sizeof *cells) {
        say("exec-smoke: setup, the narrowed argv does not cover one cell\n");
        return 1;
    }

    errno = 0;
    int result = execve(self, first, leak_environment);
    int error = errno;
    say("exec-smoke: argv reference of one cell without its terminator refused, "
        "execve %d, errno %d (%s)\n", result, error, strerror(error));
    return result != -1 || error != EFAULT;
}

/* Returns only when execve fails. */
static int exec_child(const char *self)
{
    unsigned long contexts = read_counter("live_contexts");
    unsigned long pages = read_counter("table_pages");
    if (contexts == NO_COUNTER) {
        say("exec-smoke: setup, no %slive_contexts\n", COUNTERS);
        return 10;
    }

    char variable[64] = PARENT_VARIABLE "=";
    append_number(variable, sizeof variable, contexts, 0);
    char *args[] = { (char *)self, "alpha", "beta", 0 };
    char *environment[] = { variable, 0 };
    say("exec-smoke: execve %s with %lu live lifetime tables, %lu table pages\n",
        self, contexts, pages);

    execve(self, args, environment);
    int error = errno;
    say("exec-smoke: FAIL, execve of %s returned, errno %d (%s)\n", self, error,
        strerror(error));
    return 11;
}

/* Returns only when fexecve fails. */
static int exec_by_descriptor(const char *self)
{
    char variable[] = PARENT_VARIABLE "=fexecve";
    char *args[] = { (char *)self, 0 };
    char *environment[] = { variable, 0 };
    int fd = open(self, O_RDONLY);

    if (fd < 0) {
        say("exec-smoke: FAIL, open %s, errno %d\n", self, errno);
        return 24;
    }
    fexecve(fd, args, environment);
    int error = errno;
    say("exec-smoke: FAIL, fexecve of %s returned, errno %d (%s)\n", self, error,
        strerror(error));
    return 25;
}

/* A counter of the kernel's lifetime tables, from the module parameters. */
static unsigned long read_counter(const char *name)
{
    char path[96] = COUNTERS, text[32];

    append(path, sizeof path, name);
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return NO_COUNTER;
    long bytes = read(fd, text, sizeof text - 1);
    close(fd);
    if (bytes <= 0)
        return NO_COUNTER;
    text[bytes] = 0;
    return strtoul(text, 0, 10);
}

/* A child of object that starts at object's cursor and covers size bytes. */
static void *narrow(void *object, unsigned long size)
{
    unsigned long base = query(object, 3);

    return capstone_cap_derive(object, (unsigned long)object - base, size);
}

/* LCC: selector 1 gives the type, 3 the base and 4 the end. */
static unsigned long query(void *reference, int selector)
{
    unsigned long value;

    switch (selector) {
    case 1:
        __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x1" : "=r"(value) : "r"(reference));
        return value;
    case 3:
        __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x3" : "=r"(value) : "r"(reference));
        return value;
    case 4:
        __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x4" : "=r"(value) : "r"(reference));
        return value;
    default:
        abort();
    }
}

/* One unbuffered write per line, so no output stays in a buffer across execve.
 * The format knows %s, %d and %lu: the printf family needs the soft-float
 * routines, which the trap images do not link. */
static void say(const char *format, ...)
{
    char line[256] = "";
    char piece[2] = "";
    va_list args;

    va_start(args, format);
    for (const char *at = format; *at; ++at) {
        if (*at != '%') {
            piece[0] = *at;
            append(line, sizeof line, piece);
        } else if (at[1] == 's') {
            append(line, sizeof line, va_arg(args, const char *));
            at += 1;
        } else if (at[1] == 'd') {
            int value = va_arg(args, int);
            append_number(line, sizeof line, value < 0 ? -(long)value : value, value < 0);
            at += 1;
        } else if (at[1] == 'l' && at[2] == 'u') {
            append_number(line, sizeof line, va_arg(args, unsigned long), 0);
            at += 2;
        } else {
            abort();
        }
    }
    va_end(args);
    write(1, line, strlen(line));
}

/* Append text to the string in line, cut to size bytes with its terminator. */
static void append(char *line, unsigned long size, const char *text)
{
    unsigned long length = strlen(line);

    if (!text)
        text = "(null)";
    while (*text && length + 1 < size)
        line[length++] = *text++;
    line[length] = 0;
}

static void append_number(char *line, unsigned long size, unsigned long value, int negative)
{
    char digits[24];
    int at = sizeof digits;

    digits[--at] = 0;
    do {
        digits[--at] = '0' + value % 10;
        value /= 10;
    } while (value);
    if (negative)
        digits[--at] = '-';
    append(line, size, digits + at);
}
