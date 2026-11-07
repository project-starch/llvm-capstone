/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr, "FILE_FAIL line=%d errno=%d\n", __LINE__, errno); } } while (0)
static void exchange(char c) {
    char answer = 0;
    CHECK(write(1, &c, 1) == 1);
    CHECK(read(0, &answer, 1) == 1 && answer == c);
}
static int lock(int fd, int cmd, int type) {
    struct flock f = {.l_type=type, .l_whence=SEEK_SET, .l_start=100, .l_len=10};
    return fcntl(fd, cmd, &f);
}
#ifdef FILES_RAW
extern long __capstone_native_syscall(long, void *, void *, void *, void *, void *, void *);
static void *scalar(unsigned long n) {
    void *r;
    __asm__ volatile("movc %0, zero\n\tmv %0, %1" : "=&r"(r) : "r"(n));
    return r;
}
static long raw_lock(int fd, unsigned long cmd, void *p) {
    return __capstone_native_syscall(0xc0de0003UL,scalar(fd),scalar(cmd),p,0,0,0);
}
static void raw_seek_checks(int fd) {
    char tag;
    _Alignas(16) unsigned long words[2] = {(unsigned long)fd, 1};
    void *dirty;
    __asm__ volatile("ldc %0, 0(%1)" : "=r"(dirty) : "r"(words) : "memory");
    CHECK(__capstone_native_syscall(62,scalar(fd),scalar(-1),scalar(SEEK_SET),0,0,0)==-EINVAL);
    CHECK(__capstone_native_syscall(62,scalar(fd),0,scalar((1UL<<32)|SEEK_SET),0,0,0)==-EINVAL);
    CHECK(__capstone_native_syscall(62,&tag,0,0,0,0,0)==-EINVAL);
    CHECK(__capstone_native_syscall(62,scalar(fd),&tag,0,0,0,0)==-EINVAL);
    CHECK(__capstone_native_syscall(62,scalar(fd),0,0,&tag,0,0)==-EINVAL);
    CHECK(__capstone_native_syscall(62,dirty,0,0,0,0,0)==-EINVAL);
    CHECK(__capstone_native_syscall(62,scalar(fd),scalar(-1),scalar(SEEK_END),0,0,0)==11);
    CHECK(__capstone_native_syscall(62,scalar(fd),scalar(6),scalar(SEEK_SET),0,0,0)==6);
}
/* Raw entry must reject malformed scalar arguments before any file effect. */
static void raw_fd_checks(int fd) {
    char tag;
    struct stat before, after;
    const unsigned long calls[] = {82, 83, 46, 52, 55}; /* sync, truncate, metadata */
    _Alignas(16) unsigned long words[2] = {(unsigned long)fd, 1};
    void *dirty;
    __asm__ volatile("ldc %0, 0(%1)" : "=r"(dirty) : "r"(words) : "memory");
    CHECK(fstat(fd,&before)==0);
    for (unsigned i=0; i<sizeof(calls)/sizeof(calls[0]); ++i) {
        unsigned long nr=calls[i];
        CHECK(__capstone_native_syscall(nr,&tag,0,0,0,0,0)==-EINVAL);
        CHECK(__capstone_native_syscall(nr,scalar((1UL<<32)|(unsigned)fd),0,0,0,0,0)==-EBADF);
        CHECK(__capstone_native_syscall(nr,dirty,0,0,0,0,0)==-EINVAL);
        CHECK(__capstone_native_syscall(nr,scalar(100),0,0,0,0,0)==-EBADF);
        CHECK(__capstone_native_syscall(nr,scalar(fd),0,0,0,0,&tag)==-EINVAL);
    }
    CHECK(__capstone_native_syscall(46,scalar(fd),&tag,0,0,0,0)==-EINVAL);
    CHECK(__capstone_native_syscall(52,scalar(fd),&tag,0,0,0,0)==-EINVAL);
    CHECK(__capstone_native_syscall(55,scalar(fd),&tag,scalar(-1),0,0,0)==-EINVAL);
    CHECK(__capstone_native_syscall(55,scalar(fd),scalar(-1),&tag,0,0,0)==-EINVAL);
    CHECK(__capstone_native_syscall(46,scalar(fd),dirty,0,0,0,0)==-EINVAL);
    CHECK(__capstone_native_syscall(46,scalar(fd),scalar(-1),0,0,0,0)==-EINVAL);
    CHECK(__capstone_native_syscall(55,scalar(fd),scalar(-1),scalar(-1),0,0,0)==0);
    CHECK(fstat(fd,&after)==0 && after.st_size==before.st_size &&
          after.st_mode==before.st_mode && after.st_uid==before.st_uid && after.st_gid==before.st_gid);
    CHECK(lseek(fd,0,SEEK_CUR)==6);
}
static void raw_stat_checks(int fd) {
    struct stat st;
    char name[] = "file-data";
    unsigned long base = __builtin_capstone_cap_get_cursor(&st);
    void *tiny = __builtin_capstone_cap_shrink(&st,base,base+127);
    void *ro = __builtin_capstone_cap_tighten(&st,4);
    CHECK(sizeof(st)==128);
    CHECK(__capstone_native_syscall(80,scalar(fd),tiny,0,0,0,0)==-EFAULT);
    CHECK(__capstone_native_syscall(80,scalar(fd),ro,0,0,0,0)==-EFAULT);
    CHECK(__capstone_native_syscall(80,scalar(100),&st,0,0,0,0)==-EBADF);
    base = __builtin_capstone_cap_get_cursor(name);
    void *tight = __builtin_capstone_cap_shrink(name,base,base+sizeof(name));
    void *unterminated = __builtin_capstone_cap_shrink(name,base,base+sizeof(name)-1);
    CHECK(__capstone_native_syscall(79,scalar(AT_FDCWD),tight,&st,0,0,0)==0 && st.st_size==12);
    CHECK(__capstone_native_syscall(79,scalar(AT_FDCWD),unterminated,&st,0,0,0)==-EFAULT);
    CHECK(__capstone_native_syscall(79,scalar(AT_FDCWD),name,ro,0,0,0)==-EFAULT);
    CHECK(__capstone_native_syscall(79,scalar(AT_FDCWD),name,&st,scalar(1UL<<32),0,0)==-EINVAL);
    CHECK(__capstone_native_syscall(79,scalar(AT_FDCWD),0,&st,0,0,0)==-EFAULT);
    CHECK(__capstone_native_syscall(79,scalar(AT_FDCWD),name,&st,0,name,0)==-EINVAL);
}
static void raw_checks(int fd) {
    struct flock f={.l_type=F_UNLCK,.l_whence=SEEK_SET,.l_start=100,.l_len=10};
    CHECK(sizeof(f)==32);
    unsigned long base=__builtin_capstone_cap_get_cursor(&f);
    void *tiny=__builtin_capstone_cap_shrink(&f,base,base+31);
    CHECK(raw_lock(fd,F_SETLK,tiny)==-EFAULT); /* must not unlock */
    CHECK(raw_lock(fd,F_SETLKW,NULL)==-EFAULT);
    CHECK(raw_lock(fd,(1UL<<32)|F_SETLK,&f)==-EINVAL);
    f.l_type=F_WRLCK;
    void *ro=__builtin_capstone_cap_tighten(&f,4);
    CHECK(raw_lock(fd,F_SETLK,ro)==0);
    CHECK(raw_lock(fd,F_SETLKW,ro)==0);
    CHECK(raw_lock(fd,F_GETLK,ro)==-EFAULT);
    CHECK(raw_lock(fd,F_GETLK,&f)==0 && f.l_type==F_UNLCK);
}
#endif
int main(int argc, char **argv) {
    if (argc < 2) return 2;
    int fd = open("file-data", O_CREAT|O_RDWR, 0600);
    CHECK(fd >= 0);
    if (!strcmp(argv[1], "contend")) {
        struct flock f = {.l_type=F_WRLCK, .l_whence=SEEK_SET, .l_start=100, .l_len=10};
        CHECK(fcntl(fd, F_GETLK, &f) == 0);
        CHECK(f.l_type == F_WRLCK && f.l_start == 100 && f.l_len == 10 && f.l_pid == atoi(argv[2]));
        errno=0; CHECK(lock(fd, F_SETLK, F_WRLCK) == -1 && (errno==EAGAIN || errno==EACCES));
        exchange('R'); /* parent replaces write lock by read lock */
        CHECK(lock(fd, F_SETLK, F_RDLCK) == 0);
        errno=0; CHECK(lock(fd, F_SETLK, F_WRLCK) == -1 && (errno==EAGAIN || errno==EACCES));
        exchange('S'); /* parent releases read lock */
        CHECK(lock(fd, F_SETLK, F_WRLCK) == 0);
        f.l_type=F_WRLCK;
        CHECK(fcntl(fd, F_GETLK, &f)==0 && f.l_type==F_UNLCK);
#ifdef FILES_RAW
        raw_checks(fd);
#endif
        exchange('L'); /* ordinary Linux must now observe our write lock */
        /* Exhaust the highest admissible slot, then fail without closing a
         * temporary duplicate: such a close would release ALL our locks. */
        int high=fcntl(fd,F_DUPFD,1023);
        CHECK(high==1023);
        errno=0; CHECK(fcntl(fd,F_DUPFD,1023)==-1 && errno==EMFILE);
        exchange('D');
        CHECK(close(high)==0); /* POSIX close of any same-inode fd releases locks */
        exchange('U');
    } else if (!strcmp(argv[1], "wait")) {
        exchange('W');
        CHECK(lock(fd,F_SETLKW,F_WRLCK)==0);
        exchange('A');
    } else if (!strcmp(argv[1], "basic")) {
        char buf[16]; struct stat st;
        CHECK(ftruncate(fd,0)==0);
        CHECK(write(fd,"abcdef",6)==6);
        CHECK(ftruncate(fd,3)==0 && lseek(fd,0,SEEK_CUR)==6);
        CHECK(ftruncate(fd,12)==0 && lseek(fd,0,SEEK_CUR)==6);
        memset(buf,0x55,sizeof buf);
        CHECK(pread(fd,buf,16,0)==12 && !memcmp(buf,"abc",3));
        for(int i=3;i<12;i++) CHECK(buf[i]==0);
        errno=0; CHECK(ftruncate(fd,-1)==-1 && errno==EINVAL);
        CHECK(fstat(fd,&st)==0 && st.st_size==12);
        {
            struct stat byname;
            CHECK(fstatat(AT_FDCWD,"file-data",&byname,0)==0 && !memcmp(&st,&byname,sizeof(st)));
            errno=0; CHECK(stat("missing-file",&byname)==-1 && errno==ENOENT);
            errno=0; CHECK(stat("../outside-sentinel",&byname)==-1 && errno==EACCES);
            errno=0; CHECK(stat("/tmp/outside-sentinel",&byname)==-1 && errno==ENOENT);
            errno=0; CHECK(stat("escape/outside-sentinel",&byname)==-1 && errno==ENOTDIR);
            int directory=open(".",O_RDONLY|O_DIRECTORY);
            CHECK(directory>=0);
            CHECK(fstatat(directory,"file-data",&byname,AT_SYMLINK_NOFOLLOW)==0 && byname.st_size==12);
            CHECK(close(directory)==0);
        }
        errno=0; CHECK(lseek(fd,-1,SEEK_SET)==-1 && errno==EINVAL);
        CHECK(lseek(fd,0,SEEK_CUR)==6);
        CHECK(lseek(fd,-1,SEEK_END)==11);
        CHECK(lseek(fd,1,SEEK_CUR)==12);
        errno=0; CHECK(lseek(fd,-20,SEEK_CUR)==-1 && errno==EINVAL);
        CHECK(lseek(fd,0,SEEK_CUR)==12);
        errno=0; CHECK(lseek(fd,0,777)==-1 && errno==EINVAL);
        errno=0; CHECK(lseek(0,0,SEEK_SET)==-1 && errno==ESPIPE);
        errno=0; CHECK(lseek(100,0,SEEK_SET)==-1 && errno==EBADF);
        CHECK(lseek(fd,6,SEEK_SET)==6);
#ifdef FILES_RAW
        raw_stat_checks(fd);
        raw_seek_checks(fd);
        raw_fd_checks(fd);
#endif
        CHECK(fsync(fd)==0 && fdatasync(fd)==0);
        int ro=open("file-data",O_RDONLY);
        CHECK(ro>=0);
        errno=0; CHECK(ftruncate(ro,0)==-1 && errno==EINVAL);
        CHECK(close(ro)==0);
        int du=fcntl(fd,F_DUPFD_CLOEXEC,20);
        CHECK(du>=20 && fcntl(du,F_GETFD)==FD_CLOEXEC && fcntl(fd,F_GETFD)==0);
        CHECK(fcntl(du,F_SETFD,0)==0 && fcntl(du,F_GETFD)==0);
        CHECK(lseek(du,2,SEEK_SET)==2 && lseek(fd,0,SEEK_CUR)==2);
        CHECK(fcntl(du,F_SETFL,O_APPEND)==0 && (fcntl(fd,F_GETFL)&O_APPEND));
        CHECK(fcntl(fd,F_SETFL,0)==0 && !(fcntl(du,F_GETFL)&O_APPEND));
        CHECK(close(du)==0);
        CHECK(fchmod(fd,0640)==0 && fstat(fd,&st)==0 && (st.st_mode&0777)==0640);
        CHECK(fchown(fd,-1,-1)==0);
        int dir=open(".",O_RDONLY|O_DIRECTORY);
        CHECK(dir>=0 && fsync(dir)==0 && close(dir)==0);
        errno=0; CHECK(fsync(-1)==-1 && errno==EBADF);
        errno=0; CHECK(fsync(0)==-1 && errno==EINVAL); /* supervisor pipe */
        errno=0; CHECK(fcntl(100,F_GETFD)==-1 && errno==EBADF); /* inherited but ungranted */
        errno=0; CHECK(unlink("../outside-sentinel")==-1);
        errno=0; CHECK(unlink("/tmp/outside-sentinel")==-1);
        errno=0; CHECK(unlink("escape/outside-sentinel")==-1);
        CHECK(unlink("file-data")==0);
        errno=0; CHECK(unlink("file-data")==-1 && errno==ENOENT);
        CHECK(fstat(fd,&st)==0 && st.st_nlink==0 && st.st_size==12);
    } else return 2;
    CHECK(close(fd)==0);
    fprintf(stderr,"NATIVE_FILES mode=%s checks=%d failures=%d\n",argv[1],checks,failures);
    return failures!=0;
}
