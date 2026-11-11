#define _GNU_SOURCE
#include <errno.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
static int failures;
static void check(const char *name, int ok) { printf("%s %s\n",ok?"PASS":"FAIL",name); failures += !ok; }
static int done(pid_t pid) { int st; return pid>0 && waitpid(pid,&st,0)==pid && WIFEXITED(st) && !WEXITSTATUS(st); }
static pid_t launch(const char *image, int in[2], int out[2])
{
    pid_t pid=fork();
    if (!pid) {
        if (dup2(in[0],0)<0 || dup2(out[1],1)<0) _exit(99);
        /* Extra inherited descriptors must never become application grants. */
        execl(image,image,(char *)0); _exit(98);
    }
    close(in[0]); close(out[1]); return pid;
}
int main(int argc,char **argv)
{
    if (argc!=2) return 2;
    alarm(20);
    pid_t pid=fork();
    if (!pid) {
        struct sock_filter insn=BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW);
        struct sock_fprog prog={1,&insn};
        if (prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0) || prctl(PR_SET_SECCOMP,SECCOMP_MODE_FILTER,&prog)) _exit(90);
        execl(argv[1],argv[1],(char *)0); _exit(errno==EOPNOTSUPP?0:1);
    }
    check("inherited seccomp refused",done(pid));
    pid=fork();
    if (!pid) {
        if (setresuid(65534,65534,65534)) _exit(90);
        execl(argv[1],argv[1],(char *)0); _exit(errno==EACCES?0:1);
    }
    check("unprivileged native exec refused",done(pid));
    int ia[2],oa[2],ib[2],ob[2]; char byte;
    if (pipe(ia)||pipe(oa)||pipe(ib)||pipe(ob)) return 3;
    pid_t a=launch(argv[1],ia,oa);
    check("first native task ready",read(oa[0],&byte,1)==1&&byte=='R');
    pid_t b=launch(argv[1],ib,ob);
    check("second native task ready",read(ob[0],&byte,1)==1&&byte=='R');
    check("wake first blocked read",write(ia[1],"A",1)==1);
    check("first snapshot restored",read(oa[0],&byte,1)==1&&byte=='A');
    check("wake second blocked read",write(ib[1],"B",1)==1);
    check("second snapshot restored",read(ob[0],&byte,1)==1&&byte=='B');
    check("first task exits",done(a)); check("second task exits",done(b));
    printf("native entry failures=%d\n",failures);
    return failures!=0;
}
