#define _GNU_SOURCE
#include "../wire.h"
#include "../native-grant.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

/* Runs as ordinary Linux, with bootstrap-only grants. This tests actual
 * VMA and lifetime-table state, not execution through a saved capability. */
static void one(unsigned inject)
{
    unsigned char resident;
    struct cv_stats before, after;
    int fd = open("/dev/capstone-vm", O_RDWR | O_CLOEXEC);
    assert(fd >= 0);
    char *guard = mmap(NULL, 3*4096, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    assert(guard != MAP_FAILED);
    char *p = guard+4096;
    assert(!mprotect(p,4096,PROT_READ|PROT_WRITE));
    memset(p,0x5a,4096);
    struct cv_map map = {.address=(unsigned long)p, .bytes=4096,
        .permissions=6, .reg=13, .cursor=(unsigned long)p, .linear=1};
    assert(!ioctl(fd,CV_ADD,&map));
    assert(!ioctl(fd,CV_STATS,&before));
    assert(before.arenas == 1);
    struct cv_grant_transaction r = {.version=CV_GRANT_ABI_VERSION,
        .operation=CV_GRANT_UNMAP, .id=map.id, .address=map.address,
        .bytes=map.bytes, .inject=inject};
    errno=0;
    int rc=ioctl(fd,CV_GRANT_TRANSACTION,&r), error=errno;
    assert(!ioctl(fd,CV_STATS,&after));
    if (inject) {
        assert(rc == -1);
        assert(error == (inject==1 ? EINTR : inject==2 ? EACCES :
                         inject==5 ? EIO : ENOMEM));
        assert(!mincore(p,4096,&resident));
        assert(after.arenas == before.arenas);
        assert(after.nodes_live == before.nodes_live);
        assert(after.nodes_retired == before.nodes_retired);
        for (unsigned i=0;i<4096;++i) assert((unsigned char)p[i] == 0x5a);
        r.inject=0;
        assert(!ioctl(fd,CV_GRANT_TRANSACTION,&r));
        assert(!ioctl(fd,CV_STATS,&after));
    } else assert(rc == 0);
    errno=0;
    assert(mincore(p,4096,&resident) == -1 && errno == ENOMEM);
    assert(after.arenas == 0 && after.nodes_live < before.nodes_live);
    r.inject=0;
    assert(ioctl(fd,CV_GRANT_TRANSACTION,&r) == -1 && errno == EINVAL);
    assert(!close(fd));
    assert(!munmap(guard,3*4096));
    printf("grant-probe injection=%u PASS\n", inject);
}

static void refusals(void)
{
    struct cv_stats before, after;
    int fd=open("/dev/capstone-vm",O_RDWR|O_CLOEXEC);
    assert(fd>=0);
    char *guard=mmap(NULL,4*4096,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    assert(guard!=MAP_FAILED);
    char *p=guard+4096;
    assert(!mprotect(p,2*4096,PROT_READ|PROT_WRITE));
    struct cv_map map={.address=(unsigned long)p,.bytes=8192,
        .permissions=6,.reg=13,.cursor=(unsigned long)p,.linear=1};
    assert(!ioctl(fd,CV_ADD,&map));
    assert(!ioctl(fd,CV_STATS,&before));
    struct cv_grant_transaction r={.version=CV_GRANT_ABI_VERSION,
        .operation=CV_GRANT_UNMAP,.id=map.id,.address=map.address,.bytes=map.bytes};
    r.bytes=4096;
    assert(ioctl(fd,CV_GRANT_TRANSACTION,&r)==-1 && errno==EINVAL);
    r.bytes=8192; r.id++;
    assert(ioctl(fd,CV_GRANT_TRANSACTION,&r)==-1 && errno==EINVAL);
    r.id--; r.operation=CV_GRANT_MAP;
    assert(ioctl(fd,CV_GRANT_TRANSACTION,&r)==-1 && errno==EOPNOTSUPP);
    r.operation=CV_GRANT_UNMAP; r.flags=1;
    assert(ioctl(fd,CV_GRANT_TRANSACTION,&r)==-1 && errno==EINVAL);
    r.flags=0; r.version++;
    assert(ioctl(fd,CV_GRANT_TRANSACTION,&r)==-1 && errno==EINVAL);
    r.version--;
    /* Same descriptor inherited by another address space cannot retire. */
    pid_t child=fork(); assert(child>=0);
    if (!child) _exit(ioctl(fd,CV_GRANT_TRANSACTION,&r)==-1 && errno==EPERM ? 0:1);
    int status; assert(waitpid(child,&status,0)==child && status==0);
    assert(!ioctl(fd,CV_STATS,&after));
    assert(after.arenas==before.arenas && after.nodes_live==before.nodes_live);
    /* Splitting a whole grant into multiple VMAs is refused before retirement. */
    assert(!mprotect(p+4096,4096,PROT_READ));
    assert(ioctl(fd,CV_GRANT_TRANSACTION,&r)==-1 && errno==EOPNOTSUPP);
    assert(!ioctl(fd,CV_STATS,&after));
    assert(after.arenas==before.arenas && after.nodes_live==before.nodes_live);
    assert(!mprotect(p+4096,4096,PROT_READ|PROT_WRITE));
    assert(!ioctl(fd,CV_GRANT_TRANSACTION,&r));
    assert(!close(fd));
    assert(!munmap(guard,4*4096));
    puts("grant-probe identity/partial/unsupported/foreign-mm/split PASS");
}
int main(void)
{
    for (unsigned inject=0;inject<=CV_GRANT_INJECT_RETIRE;++inject) one(inject);
    refusals();
    puts("grant-probe PASS");
    return 0;
}
