#define _GNU_SOURCE
#include "application-image.h"
#include "capstone/delegate.h"
#include <assert.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

_Alignas(Elf64_Shdr) static unsigned char image[1024];
static char path[] = "/tmp/capstone-image-XXXXXX";
static int source;
static Elf64_Ehdr *header(void) { return (void *)image; }
static Elf64_Shdr *sections(void) { return (void *)(image + 256); }

static void fixture(void) {
  memset(image, 0, sizeof image);
  Elf64_Ehdr *h = header();
  memcpy(h->e_ident, ELFMAG, SELFMAG);
  h->e_ident[EI_CLASS] = ELFCLASS64;
  h->e_ident[EI_DATA] = ELFDATA2LSB;
  h->e_ident[EI_VERSION] = EV_CURRENT;
  h->e_version = EV_CURRENT;
  h->e_machine = 259;
  h->e_type = ET_EXEC;
  h->e_ehsize = sizeof *h;
  h->e_phoff = 64;
  h->e_phnum = 1;
  h->e_phentsize = sizeof(Elf64_Phdr);
  h->e_shoff = 256;
  h->e_shnum = 4;
  h->e_shstrndx = 1;
  h->e_shentsize = sizeof(Elf64_Shdr);
  Elf64_Phdr *ph = (void *)(image + 64);
  *ph = (Elf64_Phdr){.p_type = PT_LOAD, .p_flags = PF_X, .p_filesz = 128, .p_memsz = 128};
  const char names[] = "\0.shstrtab\0.capstone_application\0.capstone_domreq";
  memcpy(image + 512, names, sizeof names);
  sections()[1] = (Elf64_Shdr){.sh_name = 1, .sh_type = SHT_STRTAB,
      .sh_offset = 512, .sh_size = sizeof names};
  sections()[2] = (Elf64_Shdr){.sh_name = 11, .sh_type = SHT_PROGBITS,
      .sh_offset = 640, .sh_size = sizeof(struct capstone_application_descriptor_v2)};
  sections()[3] = (Elf64_Shdr){.sh_name = 33, .sh_type = SHT_PROGBITS,
      .sh_offset = 704, .sh_size = 24};
  struct capstone_application_descriptor_v2 d = {{CAPSTONE_APPLICATION_MAGIC,
      CAPSTONE_LAUNCH_VERSION, CAPSTONE_APPLICATION_RECOVERY | CAPSTONE_APPLICATION_DELEGATE,
      CAPSTONE_LAUNCH_BYTES, 0}, 262144};
  memcpy(image + 640, &d, sizeof d);
  const uint64_t req[] = {UINT64_C(0x5145524d4f445043), 4096 + 256, 4096};
  memcpy(image + 704, req, sizeof req);
}

static struct capstone_application_descriptor_v2 loaded;
static int load(void) {
  assert(pwrite(source, image, sizeof image, 0) == sizeof image);
  return capstone_application_image(path, &loaded);
}

/* Rewrite the fixture's descriptor as v2: 48 bytes, the delegate flag, an
   exchange size. */
static void v2(uint64_t flags, uint64_t exchange) {
  struct capstone_application_descriptor_v2 d = {{CAPSTONE_APPLICATION_MAGIC,
      CAPSTONE_LAUNCH_VERSION, flags, CAPSTONE_LAUNCH_BYTES, 0}, exchange};
  sections()[2].sh_size = sizeof d;
  memcpy(image + 640, &d, sizeof d);
}

static void reject(void) {
  assert(load() == -1);
  assert(errno == ENOEXEC);
  fixture();
}

int main(void) {
  source = mkstemp(path);
  assert(source >= 0);
  char hash[65];
  assert(!capstone_application_hash(source, hash));
  assert(!strcmp(hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
  assert(write(source, "abc", 3) == 3);
  assert(!capstone_application_hash(source, hash));
  assert(!strcmp(hash, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
  const char *long_input = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  assert(ftruncate(source, 0) == 0 && lseek(source, 0, SEEK_SET) == 0);
  assert(write(source, long_input, strlen(long_input)) == (ssize_t)strlen(long_input));
  assert(!capstone_application_hash(source, hash));
  assert(!strcmp(hash, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
  fixture();
  int snapshot = load();
  assert(snapshot >= 0);
  assert(ftruncate(source, 0) == 0);
  unsigned char copy[sizeof image];
  assert(pread(snapshot, copy, sizeof copy, 0) == sizeof copy);
  assert(!memcmp(image, copy, sizeof copy));
  assert(pwrite(snapshot, "x", 1, 0) == -1 && errno == EPERM);
  assert(ftruncate(snapshot, 0) == -1 && errno == EPERM);
  close(snapshot);
  header()->e_shoff = UINT64_MAX; reject();
  header()->e_phoff = UINT64_MAX; reject();
  header()->e_machine = EM_RISCV; reject();
  header()->e_entry = 128; reject();
  sections()[1].sh_size = UINT64_MAX; reject();
  sections()[2].sh_name = UINT32_MAX; reject();
  sections()[2].sh_offset = 1024; reject();
  sections()[3].sh_offset = UINT64_MAX; reject();
  sections()[3].sh_type = SHT_NOBITS; reject();
  image[640] ^= 1; reject();
  /* Legacy images must be rejected, never silently use a second runtime. */
  sections()[2].sh_size = sizeof(struct capstone_application_descriptor);
  ((struct capstone_application_descriptor *)(image + 640))->flags = CAPSTONE_APPLICATION_RECOVERY;
  reject();
  /* v2: 48 bytes with the flag and a sane exchange size */
  v2(CAPSTONE_APPLICATION_RECOVERY | CAPSTONE_APPLICATION_DELEGATE, 262144);
  snapshot = load();
  assert(snapshot >= 0 && loaded.exchange_bytes == 262144 &&
         (loaded.v1.flags & CAPSTONE_APPLICATION_DELEGATE));
  close(snapshot);
  /* the flag without the size, the size without the flag, an absurd size */
  v2(CAPSTONE_APPLICATION_RECOVERY, 262144); reject();
  fixture();
  struct capstone_application_descriptor d1 = {CAPSTONE_APPLICATION_MAGIC,
      CAPSTONE_LAUNCH_VERSION, CAPSTONE_APPLICATION_RECOVERY | CAPSTONE_APPLICATION_DELEGATE,
      CAPSTONE_LAUNCH_BYTES, 0};
  sections()[2].sh_size = sizeof d1;
  memcpy(image + 640, &d1, sizeof d1); reject();
  v2(CAPSTONE_APPLICATION_RECOVERY | CAPSTONE_APPLICATION_DELEGATE, 4095); reject();
  v2(CAPSTONE_APPLICATION_RECOVERY | CAPSTONE_APPLICATION_DELEGATE, UINT64_C(2) << 30); reject();
  close(source);
  unlink(path);
  return 0;
}
