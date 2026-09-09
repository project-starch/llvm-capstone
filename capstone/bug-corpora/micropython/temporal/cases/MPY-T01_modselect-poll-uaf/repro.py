# MPY-T01 / CVE-2023-7152, MPY-T04 / issue 12887.
# The fix is titled "Handle growing the pollfds allocation correctly": m_renew
# moves the pollfds array once more than POLL_SET_ALLOC_INCREMENT (4) descriptors
# are registered, and poll_set_add_fd kept using a pointer into the old array.
# So the trigger is simply to register well past four.
import select, sys
p = select.poll()
files = []
for i in range(16):
    f = open("/dev/null", "rb")
    files.append(f)
    p.register(f, select.POLLIN)
sys.stderr.write("T01 registered %d\n" % len(files))
res = p.poll(0)
sys.stderr.write("T01 polled, %d ready\n" % len(res))
sys.stderr.write("T01 survived\n")
