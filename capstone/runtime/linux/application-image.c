#define _GNU_SOURCE
#include "application-image.h"
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int within(uint64_t offset, uint64_t bytes, size_t size) {
  return offset <= size && bytes <= size - offset;
}

static int inspect(const unsigned char *data, size_t size,
                    struct capstone_application_descriptor *out) {
  Elf64_Ehdr h;
  if (size < sizeof h)
    return ENOEXEC;
  memcpy(&h, data, sizeof h);
  if (memcmp(h.e_ident, ELFMAG, SELFMAG) || h.e_ident[EI_CLASS] != ELFCLASS64 ||
      h.e_ident[EI_DATA] != ELFDATA2LSB || h.e_machine != 259 ||
      h.e_type != ET_EXEC || h.e_ehsize != sizeof h ||
      h.e_version != EV_CURRENT || h.e_ident[EI_VERSION] != EV_CURRENT ||
      h.e_shentsize != sizeof(Elf64_Shdr) || !h.e_shnum ||
      h.e_shoff % _Alignof(Elf64_Shdr) || h.e_phoff % _Alignof(Elf64_Phdr) ||
      h.e_shstrndx >= h.e_shnum ||
      !within(h.e_shoff, (uint64_t)h.e_shnum * sizeof(Elf64_Shdr), size) ||
      h.e_phentsize != sizeof(Elf64_Phdr) || !h.e_phnum ||
      !within(h.e_phoff, (uint64_t)h.e_phnum * sizeof(Elf64_Phdr), size))
    return ENOEXEC;
  uint64_t low = UINT64_MAX, high = 0;
  int executable = 0;
  for (unsigned i = 0; i < h.e_phnum; ++i) {
    Elf64_Phdr ph;
    memcpy(&ph, data + h.e_phoff + i * sizeof ph, sizeof ph);
    if (ph.p_type == PT_LOAD &&
        (ph.p_filesz > ph.p_memsz || ph.p_memsz > 512u * 1024u * 1024u ||
         ph.p_vaddr > UINT64_MAX - ph.p_memsz ||
         !within(ph.p_offset, ph.p_filesz, size)))
      return ENOEXEC;
    if (ph.p_type == PT_LOAD) {
      if (ph.p_vaddr < low) low = ph.p_vaddr;
      if (ph.p_vaddr + ph.p_memsz > high) high = ph.p_vaddr + ph.p_memsz;
      if ((ph.p_flags & PF_X) && h.e_entry >= ph.p_vaddr &&
          h.e_entry - ph.p_vaddr < ph.p_filesz)
        executable = 1;
    }
  }
  if (!executable || high <= low || high - low > 512u * 1024u * 1024u)
    return ENOEXEC;
  Elf64_Shdr names;
  memcpy(&names, data + h.e_shoff + h.e_shstrndx * sizeof names, sizeof names);
  if (names.sh_type != SHT_STRTAB || !within(names.sh_offset, names.sh_size, size))
    return ENOEXEC;
  int found = 0;
  for (unsigned i = 0; i < h.e_shnum; ++i) {
    Elf64_Shdr section;
    memcpy(&section, data + h.e_shoff + i * sizeof section, sizeof section);
    if (section.sh_name >= names.sh_size ||
        (section.sh_type != SHT_NOBITS && !within(section.sh_offset, section.sh_size, size)))
      return ENOEXEC;
    const char *name = (const char *)data + names.sh_offset + section.sh_name;
    if (!memchr(name, 0, names.sh_size - section.sh_name))
      return ENOEXEC;
    if (!strcmp(name, ".capstone_domreq")) {
      uint64_t req[3];
      if (section.sh_type != SHT_PROGBITS || section.sh_size != sizeof req ||
          section.sh_offset % _Alignof(uint64_t))
        return ENOEXEC;
      memcpy(req, data + section.sh_offset, sizeof req);
      if (req[0] != UINT64_C(0x5145524d4f445043) || req[1] < 256 ||
          req[1] > 256u * 1024u * 1024u || req[2] > req[1] - 256)
        return ENOEXEC;
    }
    if (!strcmp(name, ".capstone_gp_initdesc") &&
        (section.sh_addr < low || section.sh_addr >= high))
      return ENOEXEC;
    if (strcmp(name, ".capstone_application"))
      continue;
    if (found++ || section.sh_type != SHT_PROGBITS ||
        section.sh_size != sizeof *out ||
        !within(section.sh_offset, sizeof *out, size))
      return ENOEXEC;
    memcpy(out, data + section.sh_offset, sizeof *out);
  }
  if (!found || out->magic != CAPSTONE_APPLICATION_MAGIC ||
      out->version != CAPSTONE_LAUNCH_VERSION ||
      out->flags != CAPSTONE_APPLICATION_RECOVERY ||
      out->launch_bytes != CAPSTONE_LAUNCH_BYTES ||
      out->heap_bytes > 256u * 1024u * 1024u)
    return ENOEXEC;
  return 0;
}

int capstone_application_image(const char *path,
                               struct capstone_application_descriptor *out) {
  int source = open(path, O_RDONLY | O_CLOEXEC);
  if (source < 0)
    return -1;
  int image = -1, error = 0;
  struct stat st;
  if (fstat(source, &st)) {
    error = errno;
    goto done;
  }
  if (!S_ISREG(st.st_mode) || st.st_size < (off_t)sizeof(Elf64_Ehdr) ||
      st.st_size > 512 * 1024 * 1024) {
    error = ENOEXEC;
    goto done;
  }
  image = memfd_create("capstone-application", MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (image < 0) {
    error = errno;
    goto done;
  }
  char buf[65536];
  off_t remaining = st.st_size;
  while (remaining) {
    size_t wanted = remaining < (off_t)sizeof buf ? (size_t)remaining : sizeof buf;
    ssize_t n = read(source, buf, wanted);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0) {
      error = n ? errno : ENOEXEC;
      goto done;
    }
    size_t written = 0;
    while (written < (size_t)n) {
      ssize_t k = write(image, buf + written, (size_t)n - written);
      if (k < 0 && errno == EINTR)
        continue;
      if (k <= 0) {
        error = k ? errno : EIO;
        goto done;
      }
      written += (size_t)k;
    }
    remaining -= n;
  }
  if (fcntl(image, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL)) {
    error = errno;
    goto done;
  }
  void *data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, image, 0);
  if (data == MAP_FAILED) {
    error = errno;
    goto done;
  }
  error = inspect(data, (size_t)st.st_size, out);
  munmap(data, (size_t)st.st_size);
done:
  close(source);
  if (error) {
    if (image >= 0)
      close(image);
    errno = error;
    return -1;
  }
  return image;
}
