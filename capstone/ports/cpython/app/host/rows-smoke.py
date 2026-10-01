"""The plain rows from CPython: one line per check, a SMOKE total at the end.
Exit 0 when every check passed. Before the rows exist each of these is
ENOSYS (errno 38) from the domain's libc; after, Linux's own answer."""
import errno
import os
import resource
import stat
import sys
import time

results = []


def check(name, fn):
    try:
        ok, detail = bool(fn()), ""
    except OSError as e:
        ok, detail = False, f"errno {e.errno} {e.strerror}"
    except Exception as e:  # noqa: BLE001
        ok, detail = False, repr(e)
    results.append(ok)
    print("PASS" if ok else "FAIL", name, detail, flush=True)


p = f"/tmp/rows-{os.getpid()}"
with open(p, "wb") as f:
    f.write(b"hello")
fd = os.open(p, os.O_RDWR)

check("statvfs", lambda: os.statvfs("/").f_bsize > 0)
check("fstatvfs", lambda: os.fstatvfs(fd).f_bsize > 0)
check("getrusage", lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 0)
check("cpu_count", lambda: (os.cpu_count() or 0) >= 1)
check("sched_getaffinity", lambda: len(os.sched_getaffinity(0)) >= 1)
check("sched_setaffinity", lambda: os.sched_setaffinity(0, os.sched_getaffinity(0)) is None)
check("truncate", lambda: (os.truncate(p, 2), os.stat(p).st_size == 2)[1])
check("link", lambda: (os.link(p, p + ".l"), os.stat(p).st_nlink == 2, os.unlink(p + ".l"))[1])
check("fchmod", lambda: (os.fchmod(fd, 0o640), stat.S_IMODE(os.fstat(fd).st_mode) == 0o640)[1])
check("fchown", lambda: os.fchown(fd, os.getuid(), os.getgid()) is None)
check("chown", lambda: os.chown(p, -1, -1) is None)
check("mkfifo", lambda: (os.mkfifo(p + ".fifo"), stat.S_ISFIFO(os.stat(p + ".fifo").st_mode), os.unlink(p + ".fifo"))[1])
check("access-effective", lambda: os.access(p, os.R_OK, effective_ids=True))
check("posix_fallocate", lambda: (os.posix_fallocate(fd, 0, 8192), os.fstat(fd).st_size == 8192)[1])


def sendfile():
    r, w = os.pipe()
    n = os.sendfile(w, fd, 0, 2)
    data = os.read(r, 2)
    os.close(r)
    os.close(w)
    return n == 2 and data == b"he"


check("sendfile", sendfile)


def copy_file_range():
    out = os.open(p + ".c", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    n = os.copy_file_range(fd, out, 2, 0, 0)
    data = os.pread(out, 4, 0)
    os.close(out)
    os.unlink(p + ".c")
    return n == 2 and data == b"he"


check("copy_file_range", copy_file_range)
check("posix_fadvise", lambda: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_NORMAL) is None)
check("sync", lambda: os.sync() is None)
check("memfd_create", lambda: os.memfd_create("rows") >= 0)
check("clock_getres", lambda: time.clock_getres(time.CLOCK_MONOTONIC) > 0)
check("getgroups", lambda: isinstance(os.getgroups(), list))
check("getpriority", lambda: -20 <= os.getpriority(os.PRIO_PROCESS, 0) <= 19)
check("setpriority", lambda: os.setpriority(os.PRIO_PROCESS, 0, os.getpriority(os.PRIO_PROCESS, 0)) is None)
check("sched_get_priority_max", lambda: os.sched_get_priority_max(os.SCHED_FIFO) == 99)
check("sched_rr_get_interval", lambda: os.sched_rr_get_interval(0) >= 0)
check("fchdir", lambda: (os.fchdir(os.open("/tmp", os.O_RDONLY)), os.getcwd() == "/tmp")[1])
check("setpgid-own-group", lambda: (os.setpgid(0, 0), os.getpgrp() == os.getpid())[1])
check("getrlimit-via-prlimit64", lambda: resource.getrlimit(resource.RLIMIT_NOFILE)[0] > 0)

os.close(fd)
os.unlink(p)
passed = sum(results)
print("SMOKE", passed, "/", len(results), flush=True)
sys.exit(0 if passed == len(results) else 1)
