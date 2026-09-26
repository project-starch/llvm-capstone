#define _GNU_SOURCE
#include "application-image.h"
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
      .sh_offset = 640, .sh_size = sizeof(struct capstone_application_descriptor)};
  sections()[3] = (Elf64_Shdr){.sh_name = 33, .sh_type = SHT_PROGBITS,
      .sh_offset = 704, .sh_size = 24};
  struct capstone_application_descriptor d = {CAPSTONE_APPLICATION_MAGIC,
      CAPSTONE_LAUNCH_VERSION, CAPSTONE_APPLICATION_RECOVERY, CAPSTONE_LAUNCH_BYTES, 0};
  memcpy(image + 640, &d, sizeof d);
  const uint64_t req[] = {UINT64_C(0x5145524d4f445043), 4096 + 256, 4096};
  memcpy(image + 704, req, sizeof req);
}

static int load(void) {
  assert(pwrite(source, image, sizeof image, 0) == sizeof image);
  struct capstone_application_descriptor d;
  return capstone_application_image(path, &d);
}

static void reject(void) {
  assert(load() == -1);
  assert(errno == ENOEXEC);
  fixture();
}

int main(void) {
  source = mkstemp(path);
  assert(source >= 0);
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
  close(source);
  unlink(path);
  return 0;
}
