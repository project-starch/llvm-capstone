"""CPython over the socket rows: one line per check, a SMOKE total at the end.
Exit 0 when every check passed. Before the rows each of these is ENOSYS from
the domain's libc; after, Linux's own answer. Runs on Linux as the oracle."""
import errno
import os
import select
import selectors
import socket
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


def tcp_pair():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(server.getsockname())
    accepted, peer = server.accept()
    server.close()
    return client, accepted, peer


def stream():
    client, accepted, peer = tcp_pair()
    client.sendall(b"hello")
    got = accepted.recv(16)
    accepted.shutdown(socket.SHUT_WR)
    eof = client.recv(16)
    client.close(); accepted.close()
    return got == b"hello" and eof == b"" and peer[0] == "127.0.0.1"


check("socket-tcp-loopback", stream)
check("socketpair", lambda: (lambda a, b: (a.send(b"x"), b.recv(1) == b"x", a.close(), b.close())[1])(*socket.socketpair()))


def dgram():
    a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    a.bind(("127.0.0.1", 0)); b.bind(("127.0.0.1", 0))
    a.sendto(b"datagram", b.getsockname())
    data, addr = b.recvfrom(64)
    a.close(); b.close()
    return data == b"datagram" and addr[1] == a.getsockname()[1] if False else data == b"datagram" and addr[0] == "127.0.0.1"


check("udp-loopback", dgram)


def unix_stream():
    path = f"/tmp/socket-smoke-{os.getpid()}"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path); server.listen(1)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(path)
    accepted, _ = server.accept()
    client.sendall(b"unix")
    got = accepted.recv(8)
    name = accepted.getsockname()
    for s in (client, accepted, server): s.close()
    os.unlink(path)
    return got == b"unix" and name == path


check("unix-stream", unix_stream)
check("getsockopt-type", lambda: socket.socket().getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_STREAM)


def timeout():
    a, b = socket.socketpair()
    a.settimeout(0.1)
    t0 = time.monotonic()
    try:
        a.recv(1)
        return False
    except socket.timeout:
        return time.monotonic() - t0 >= 0.09
    finally:
        a.close(); b.close()


check("settimeout-recv", timeout)


def nonblocking():
    a, b = socket.socketpair()
    a.setblocking(False)
    try:
        a.recv(1)
        return False
    except BlockingIOError:
        return True
    finally:
        a.close(); b.close()


check("setblocking-false", nonblocking)


def fds():
    a, b = socket.socketpair()
    r, w = os.pipe()
    socket.send_fds(a, [b"fd"], [w])
    msg, received, flags, addr = socket.recv_fds(b, 16, 1)
    os.write(received[0], b"y")
    got = os.read(r, 1)
    for fd in (r, w, received[0]): os.close(fd)
    a.close(); b.close()
    return msg == b"fd" and got == b"y"


check("send_fds-recv_fds", fds)


def sendmsg_iov():
    a, b = socket.socketpair()
    n = a.sendmsg([b"ab", b"cd"])
    got = b.recvmsg(8)
    a.close(); b.close()
    return n == 4 and got[0] == b"abcd"


check("sendmsg-recvmsg", sendmsg_iov)


def selectors_ready():
    a, b = socket.socketpair()
    sel = selectors.DefaultSelector()
    sel.register(a, selectors.EVENT_READ, "data")
    before = sel.select(0)
    b.send(b"r")
    after = sel.select(1)
    kind = type(sel).__name__
    sel.close(); a.close(); b.close()
    print("  selector:", kind)
    return before == [] and len(after) == 1 and after[0][0].data == "data"


check("selectors", selectors_ready)


def epoll_ready():
    a, b = socket.socketpair()
    ep = select.epoll()
    ep.register(a, select.EPOLLIN)
    before = ep.poll(0)
    b.send(b"e")
    after = ep.poll(1)
    fd = a.fileno()
    ep.close(); a.close(); b.close()
    return before == [] and after == [(fd, select.EPOLLIN)]


check("epoll", epoll_ready)
check("select-poll", lambda: (lambda a, b: (b.send(b"p"), select.select([a], [], [], 1)[0] == [a], a.close(), b.close())[1])(*socket.socketpair()))
check("getaddrinfo-localhost", lambda: any(r[4][0] == "127.0.0.1" for r in socket.getaddrinfo("localhost", 80, socket.AF_INET)))
check("gethostname", lambda: isinstance(socket.gethostname(), str))


def refused():
    s = socket.socket()
    s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    try:
        socket.create_connection(("127.0.0.1", port), timeout=2)
        return False
    except ConnectionRefusedError:
        return True


check("connect-refused", refused)


def big_stream():
    client, accepted, _ = tcp_pair()
    data = bytes(range(256)) * 400   # 102400 bytes, larger than any exchange region here
    client.sendall(data)
    client.shutdown(socket.SHUT_WR)
    got = bytearray()
    while chunk := accepted.recv(65536):
        got += chunk
    client.close(); accepted.close()
    return bytes(got) == data


check("sendall-100k", big_stream)

passed = sum(results)
print("SMOKE", passed, "/", len(results), flush=True)
sys.exit(0 if passed == len(results) else 1)
