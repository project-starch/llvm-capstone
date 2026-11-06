/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

static int in, out, fd, checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr,"DRIVER_FAIL line=%d errno=%d\n",__LINE__,errno); exit(1); } } while (0)
static int lock(int type) {
    struct flock f={.l_type=type,.l_whence=SEEK_SET,.l_start=100,.l_len=10};
    return fcntl(fd,F_SETLK,&f);
}
static pid_t start(const char *binary,const char *mode) {
    int a[2],b[2]; CHECK(pipe(a)==0 && pipe(b)==0);
    pid_t p=fork(); CHECK(p>=0);
    if(!p) {
        char parent[32]; snprintf(parent,sizeof parent,"%d",getppid());
        if(dup2(a[0],0)<0 || dup2(b[1],1)<0) _exit(125);
        close(a[0]); close(a[1]); close(b[0]); close(b[1]);
        if(dup2(fd,100)<0) _exit(125);
        execl(binary,binary,mode,parent,NULL); _exit(126);
    }
    close(a[0]);close(b[1]);out=a[1];in=b[0];return p;
}
static void receive(char want) {
    struct pollfd p={.fd=in,.events=POLLIN}; char c=0;
    CHECK(poll(&p,1,10000)==1 && read(in,&c,1)==1 && c==want);
}
static void answer(char c) {CHECK(write(out,&c,1)==1);}
static void done(pid_t child,int killed) {
    int status; CHECK(waitpid(child,&status,0)==child);
    CHECK(killed ? WIFSIGNALED(status)&&WTERMSIG(status)==SIGKILL : WIFEXITED(status)&&WEXITSTATUS(status)==0);
    close(in);close(out);
}
static void blocked(pid_t child) {
    struct pollfd p={.fd=in,.events=POLLIN};
    CHECK(poll(&p,1,100)==0);
    /* Lack of output alone could mean the child has not reached F_SETLKW.
     * Require Linux's actual blocked-lock record before wakeup or SIGKILL. */
    FILE *locks=fopen("/proc/locks","r"); char line[512]; int found=0;
    CHECK(locks!=NULL);
    while(fgets(line,sizeof line,locks)) {
        char *waiter=strstr(line,"->"); long owner=0;
        if(waiter && sscanf(waiter+2," %*s %*s %*s %ld",&owner)==1 && owner==child) found=1;
    }
    fclose(locks);CHECK(found);
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    alarm(45);
    struct rlimit lim={2048,2048}; CHECK(setrlimit(RLIMIT_NOFILE,&lim)==0);
    fd=open("file-data",O_CREAT|O_RDWR,0600); CHECK(fd>=0);
    CHECK(lock(F_WRLCK)==0);
    pid_t p=start(argv[1],"contend");
    receive('R'); CHECK(lock(F_RDLCK)==0);answer('R');
    receive('S'); CHECK(lock(F_UNLCK)==0);answer('S');
    receive('L');errno=0;CHECK(lock(F_WRLCK)==-1&&(errno==EAGAIN||errno==EACCES));answer('L');
    receive('D');errno=0;CHECK(lock(F_WRLCK)==-1&&(errno==EAGAIN||errno==EACCES));answer('D');
    receive('U');CHECK(lock(F_WRLCK)==0);CHECK(lock(F_UNLCK)==0);answer('U');done(p,0);
    CHECK(lock(F_WRLCK)==0);p=start(argv[1],"wait");receive('W');answer('W');blocked(p);
    CHECK(lock(F_UNLCK)==0);receive('A');errno=0;CHECK(lock(F_WRLCK)==-1&&(errno==EAGAIN||errno==EACCES));answer('A');done(p,0);
    CHECK(lock(F_WRLCK)==0);p=start(argv[1],"wait");receive('W');answer('W');blocked(p);
    CHECK(kill(p,SIGKILL)==0);done(p,1);CHECK(lock(F_UNLCK)==0);
    CHECK(lock(F_WRLCK)==0);CHECK(lock(F_UNLCK)==0);
    p=start(argv[1],"basic");done(p,0);
    CHECK(close(fd)==0);
    printf("NATIVE_FILES_DRIVER checks=%d failures=0\n",checks);return 0;
}
