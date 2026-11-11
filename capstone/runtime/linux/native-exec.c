/* capstone-vexec for the native task route.
 *
 *   capstone-vexec IMAGE [ARGUMENT...]
 *   capstone-vexec --platform
 *
 * The kernel runs a Capstone image as an ordinary Linux task (binfmt), so this
 * program holds no device, no context and no trampoline. It exists for the
 * runners: they start every image as `./capstone-vexec IMAGE ...` and read one
 * line when the image faults,
 *
 *   capstone-exec: domain fault cause=C pc=0x.. address=0x.. ra=0x.. fp=0x..
 *                  entry=0x.. code=0x..-0x.. image=IMAGE
 *
 * The launcher of the module route printed that line from its own step record.
 * Here the kernel logs the record (`capstone_task: fault ...`, module/binfmt-dom.h)
 * and this program prints the line from the record that names its child.
 *
 * The image runs as a child, and this program ends the way the child ended: the
 * same exit status, or death by the same signal. A caught signal goes on to the
 * child. SIGKILL of this program kills the child too (PR_SET_PDEATHSIG).
 * SIGSTOP stops only this program: a runner that stops the image itself must
 * signal the process group.
 *
 * As in the module route, the fault line goes to stderr when stderr is a
 * terminal or CAPSTONE_EXEC_DIAGNOSTICS is set, and to the file that
 * CAPSTONE_FAULT_RECORD names when that is set.
 *
 * --platform prints the adapter's parameters, one per line, and exits 0 only
 * when the kernel runs native tasks. Runners call it where they loaded the
 * module before. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define PARAMETERS "/sys/module/capstone_vm/parameters/"
#define RECORD_MARK "capstone_task: fault "

struct fault {
    uint64_t cause, pc, address, ra, fp, entry, code_base, code_end;
};

static int report_platform(void);
static int run_image(char **argv);
static int native_tasks_enabled(void);
static int read_parameter(const char *name, char *value, size_t size);
static int open_kernel_log(void);
static void forward_signals(void);
static void forward(int signal_number);
static int find_fault(int log, pid_t child, struct fault *fault);
static int parse_fault(const char *record, pid_t child, struct fault *fault);
static int word(const char *record, const char *key, int base, uint64_t *value);
static void publish_fault(const struct fault *fault, const char *image);
static void write_fault_line(int fd, const struct fault *fault, const char *image);
static void die_as_child(int status);

static volatile pid_t image_pid;

int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "--platform")) {
        return report_platform();
    }

    if (argc < 2 || argv[1][0] == '-') {
        fprintf(stderr, "usage: capstone-vexec IMAGE [ARGUMENT...] | capstone-vexec --platform\n");
        return 2;
    }

    return run_image(argv + 1);
}

static int report_platform(void)
{
    static const char *const names[] = {
        "native_task", "dom_binfmt", "native_dispatch", "checked_copy",
        "native_slot_transport", "unpinned", "supervisor_slots",
    };
    char value[64];

    for (size_t i = 0; i < sizeof names / sizeof *names; ++i) {
        if (read_parameter(names[i], value, sizeof value)) {
            printf("CAPSTONE_PLATFORM %s=absent\n", names[i]);
        } else {
            printf("CAPSTONE_PLATFORM %s=%s\n", names[i], value);
        }
    }

    if (!native_tasks_enabled()) {
        fprintf(stderr, "capstone-vexec: this kernel does not run native Capstone tasks "
                        "(capstone_vm.native_task=1 capstone_vm.dom_binfmt=1)\n");
        return 1;
    }
    return 0;
}

static int run_image(char **argv)
{
    const char *image = argv[0];
    struct fault fault;
    int log, status;
    pid_t child;

    if (!native_tasks_enabled()) {
        fprintf(stderr, "capstone-vexec: this kernel does not run native Capstone tasks "
                        "(capstone_vm.native_task=1 capstone_vm.dom_binfmt=1)\n");
        return 125;
    }

    /* Positioned at the end before the child exists: every record read later
     * is newer than this point. */
    log = open_kernel_log();

    child = fork();
    if (child < 0) {
        perror("capstone-vexec: fork");
        return 125;
    }
    if (child == 0) {
        prctl(PR_SET_PDEATHSIG, SIGKILL);
        if (getppid() == 1) {
            _exit(125);
        }
        execv(image, argv);
        perror("capstone-vexec: image");
        _exit(errno == ENOENT ? 127 : 126);
    }

    image_pid = child;
    forward_signals();

    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) {
            perror("capstone-vexec: wait");
            return 125;
        }
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }

    if (WIFSIGNALED(status)) {
        if (log >= 0 && find_fault(log, child, &fault)) {
            publish_fault(&fault, image);
        }
        die_as_child(status);
    }

    fprintf(stderr, "capstone-vexec: the image ended with wait status 0x%x\n", status);
    return 125;
}

static int native_tasks_enabled(void)
{
    char task[8], binfmt[8];

    if (read_parameter("native_task", task, sizeof task) ||
        read_parameter("dom_binfmt", binfmt, sizeof binfmt)) {
        return 0;
    }
    return task[0] == 'Y' && binfmt[0] == 'Y';
}

static int read_parameter(const char *name, char *value, size_t size)
{
    char path[160];
    ssize_t n;
    int fd;

    snprintf(path, sizeof path, PARAMETERS "%s", name);
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return -1;
    }

    n = read(fd, value, size - 1);
    close(fd);
    if (n <= 0) {
        return -1;
    }

    value[n] = 0;
    value[strcspn(value, "\n")] = 0;
    return 0;
}

static int open_kernel_log(void)
{
    int log = open("/dev/kmsg", O_RDONLY | O_NONBLOCK | O_CLOEXEC);

    if (log < 0) {
        return -1;
    }
    if (lseek(log, 0, SEEK_END) < 0) {
        close(log);
        return -1;
    }
    return log;
}

static void forward_signals(void)
{
    static const int forwarded[] = {SIGHUP, SIGINT, SIGQUIT, SIGTERM, SIGUSR1, SIGUSR2, SIGCONT, SIGALRM};
    struct sigaction action = {0};

    action.sa_handler = forward;
    sigemptyset(&action.sa_mask);
    for (size_t i = 0; i < sizeof forwarded / sizeof *forwarded; ++i) {
        sigaction(forwarded[i], &action, NULL);
    }
}

static void forward(int signal_number)
{
    int saved = errno;

    if (image_pid > 0) {
        kill(image_pid, signal_number);
    }
    errno = saved;
}

/* The newest record that names the child. A record the ring buffer dropped
 * reads as EPIPE, and the read after it continues with the next one. */
static int find_fault(int log, pid_t child, struct fault *fault)
{
    char record[1024];
    int found = 0;

    for (;;) {
        ssize_t n = read(log, record, sizeof record - 1);

        if (n < 0 && errno == EPIPE) {
            continue;
        }
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            break;
        }

        record[n] = 0;
        if (parse_fault(record, child, fault)) {
            found = 1;
        }
    }
    return found;
}

static int parse_fault(const char *record, pid_t child, struct fault *fault)
{
    const char *text = strstr(record, RECORD_MARK);
    struct fault parsed;
    uint64_t pid;
    const char *range;

    if (!text) {
        return 0;
    }
    if (!word(text, " pid=", 10, &pid) || (pid_t)pid != child) {
        return 0;
    }

    if (!word(text, " cause=", 10, &parsed.cause) || !word(text, " pc=", 16, &parsed.pc) ||
        !word(text, " address=", 16, &parsed.address) || !word(text, " ra=", 16, &parsed.ra) ||
        !word(text, " fp=", 16, &parsed.fp) || !word(text, " entry=", 16, &parsed.entry) ||
        !word(text, " code=", 16, &parsed.code_base)) {
        return 0;
    }

    range = strstr(text, " code=");
    range = strchr(range, '-');
    if (!range) {
        return 0;
    }
    parsed.code_end = strtoull(range + 1, NULL, 16);

    *fault = parsed;
    return 1;
}

static int word(const char *record, const char *key, int base, uint64_t *value)
{
    const char *at = strstr(record, key);
    char *end;

    if (!at) {
        return 0;
    }

    at += strlen(key);
    errno = 0;
    *value = strtoull(at, &end, base);
    return end != at && !errno;
}

static void publish_fault(const struct fault *fault, const char *image)
{
    const char *record = getenv("CAPSTONE_FAULT_RECORD");

    if (record && *record) {
        int fd = open(record, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);

        if (fd >= 0) {
            write_fault_line(fd, fault, image);
            close(fd);
        }
    }

    if (isatty(2) || getenv("CAPSTONE_EXEC_DIAGNOSTICS")) {
        write_fault_line(2, fault, image);
    }
}

/* One write, without blocking: stderr may be a closed or a full pipe. */
static void write_fault_line(int fd, const struct fault *fault, const char *image)
{
    char line[1024];
    int flags, n;

    n = snprintf(line, sizeof line,
                 "capstone-exec: domain fault cause=%llu pc=0x%llx address=0x%llx ra=0x%llx fp=0x%llx"
                 " entry=0x%llx code=0x%llx-0x%llx image=%s\n",
                 (unsigned long long)fault->cause, (unsigned long long)fault->pc,
                 (unsigned long long)fault->address, (unsigned long long)fault->ra,
                 (unsigned long long)fault->fp, (unsigned long long)fault->entry,
                 (unsigned long long)fault->code_base, (unsigned long long)fault->code_end, image);
    if (n < 0) {
        return;
    }
    if ((size_t)n >= sizeof line) {
        n = sizeof line - 1;
        line[n - 1] = '\n';
    }

    flags = fcntl(fd, F_GETFL);
    if (flags >= 0) {
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }
    (void)!write(fd, line, (size_t)n);
    if (flags >= 0) {
        fcntl(fd, F_SETFL, flags);
    }
}

/* Runners require death by the signal, not an exit status that looks like it. */
static void die_as_child(int status)
{
    int signal_number = WTERMSIG(status);
    struct sigaction action = {0};
    sigset_t signals;

    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    sigaction(signal_number, &action, NULL);

    sigemptyset(&signals);
    sigaddset(&signals, signal_number);
    sigprocmask(SIG_UNBLOCK, &signals, NULL);

    raise(signal_number);
    _Exit(128 + signal_number);
}
