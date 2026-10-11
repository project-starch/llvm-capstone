"""subprocess in a delegated domain, over the posix_spawn fork_exec: one line
per check, a SMOKE total at the end."""
import os
import signal
import subprocess
import sys
import time

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print("PASS" if ok else "FAIL", name, detail, flush=True)


# 1. run, capture, exit status
p = subprocess.run(["sh", "-c", "echo out; echo err >&2; exit 3"], capture_output=True, text=True)
check("run-capture", p.returncode == 3 and p.stdout == "out\n" and p.stderr == "err\n", f"rc={p.returncode} out={p.stdout!r} err={p.stderr!r}")

# 2. check_output and check_call
check("check_output", subprocess.check_output(["echo", "hi"]) == b"hi\n")
try:
    subprocess.check_call(["sh", "-c", "exit 2"])
    check("check_call-raises", False)
except subprocess.CalledProcessError as e:
    check("check_call-raises", e.returncode == 2)

# 3. stdin from a pipe, communicate
p = subprocess.run(["tr", "a-z", "A-Z"], input="from a pipe\n", capture_output=True, text=True)
check("communicate-stdin", p.stdout == "FROM A PIPE\n", repr(p.stdout))

# 4. Popen with pipes, write then read, wait
proc = subprocess.Popen(["cat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
out, _ = proc.communicate(b"x" * 10000)
check("popen-cat-10k", out == b"x" * 10000 and proc.returncode == 0, f"len={len(out)} rc={proc.returncode}")

# 5. env and cwd
p = subprocess.run(["sh", "-c", "echo $MARK; pwd"], env={"MARK": "marker", "PATH": os.environ["PATH"]}, cwd="/tmp", capture_output=True, text=True)
check("env-cwd", p.stdout == "marker\n/tmp\n", repr(p.stdout))

# 6. missing program: the error, named after the program
try:
    subprocess.run(["no-such-program-xyz"])
    check("enoent", False, "no exception")
except FileNotFoundError as e:
    check("enoent", e.errno == 2 and e.filename == "no-such-program-xyz", f"{e!r}")

# 7. missing cwd
try:
    subprocess.run(["true"], cwd="/no/such/dir")
    check("enoent-cwd", False, "no exception")
except (FileNotFoundError, NotADirectoryError) as e:
    check("enoent-cwd", True, f"{type(e).__name__}")

# 8. pass_fds: the child sees a descriptor we keep, and not one we do not
r, w = os.pipe()
p = subprocess.run(["sh", "-c", f"echo kept >&{w}"], pass_fds=(w,), capture_output=True, text=True)
os.close(w)
kept = os.read(r, 100)
os.close(r)
check("pass_fds", p.returncode == 0 and kept == b"kept\n", f"rc={p.returncode} read={kept!r}")
r, w = os.pipe()
os.set_inheritable(w, True)
p = subprocess.run(["sh", "-c", f"echo leaked >&{w} 2>/dev/null; exit 0"], capture_output=True, text=True)
os.close(w)
os.set_blocking(r, False)
try:
    leaked = os.read(r, 100)
except BlockingIOError:
    leaked = b""
os.close(r)
check("close_fds", leaked == b"", f"read={leaked!r}")

# 9. restore_signals: SIGPIPE ignored here, default in the child
signal.signal(signal.SIGPIPE, signal.SIG_IGN)
p = subprocess.run(["sh", "-c", "kill -PIPE $$; echo alive"], capture_output=True, text=True)
check("restore_signals", p.returncode == -signal.SIGPIPE and p.stdout == "", f"rc={p.returncode} out={p.stdout!r}")
p = subprocess.run(["sh", "-c", "kill -PIPE $$; echo alive"], capture_output=True, text=True, restore_signals=False)
check("inherit-ign", p.returncode == 0 and p.stdout == "alive\n", f"rc={p.returncode} out={p.stdout!r}")
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# 10. kill, terminate, wait, timeout
proc = subprocess.Popen(["sleep", "30"])
proc.terminate()
check("terminate", proc.wait() == -signal.SIGTERM, f"rc={proc.returncode}")
proc = subprocess.Popen(["sleep", "30"])
try:
    proc.wait(timeout=0.5)
    check("wait-timeout", False)
except subprocess.TimeoutExpired:
    proc.kill()
    check("wait-timeout", proc.wait() == -signal.SIGKILL)
try:
    subprocess.run(["sleep", "30"], timeout=0.5)
    check("run-timeout", False)
except subprocess.TimeoutExpired:
    check("run-timeout", True)

# 11. start_new_session and process_group
p = subprocess.run(["sh", "-c", "echo $$"], start_new_session=True, capture_output=True, text=True)
check("new-session", p.returncode == 0 and p.stdout.strip().isdigit(), repr(p.stdout))
p = subprocess.run(["sh", "-c", "ps -o pgid= -p $$ || echo skip"], process_group=0, capture_output=True, text=True)
check("process-group", p.returncode == 0, f"rc={p.returncode} out={p.stdout.strip()!r}")

# 12. what posix_spawn cannot do is refused up front
try:
    subprocess.run(["true"], preexec_fn=lambda: None)
    check("preexec-refused", False, "ran")
except OSError as e:
    check("preexec-refused", e.errno == 38, f"errno={e.errno}")

# 13. a child that signals the parent while the parent waits (subprocess path this time)
got = []
signal.signal(signal.SIGUSR1, lambda s, f: got.append(1))
p = subprocess.run(["sh", "-c", f"kill -USR1 {os.getpid()}; echo done"], capture_output=True, text=True)
check("child-signals-parent", got == [1] and p.stdout == "done\n", f"got={got} out={p.stdout!r}")

# 14. os.popen and shell=True
check("os.popen", os.popen("echo popen").read() == "popen\n")
p = subprocess.run("echo $((6*7))", shell=True, capture_output=True, text=True)
check("shell", p.stdout == "42\n", repr(p.stdout))

passed = sum(1 for _, ok in results if ok)
print("SMOKE", passed, "/", len(results), flush=True)
sys.exit(0 if passed == len(results) else 1)
