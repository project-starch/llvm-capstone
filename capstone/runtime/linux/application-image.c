#define _GNU_SOURCE
#include "application-image.h"
#include "capstone/delegate.h"
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
                    struct capstone_application_descriptor_v2 *out) {
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
        !within(section.sh_offset, section.sh_size, size))
      return ENOEXEC;
    memset(out, 0, sizeof *out);
    memcpy(out, data + section.sh_offset, section.sh_size);
  }
  if (!found || out->v1.magic != CAPSTONE_APPLICATION_MAGIC ||
      out->v1.version != CAPSTONE_LAUNCH_VERSION ||
      out->v1.launch_bytes != CAPSTONE_LAUNCH_BYTES ||
      out->v1.heap_bytes > 256u * 1024u * 1024u)
    return ENOEXEC;
  if (out->v1.flags != (CAPSTONE_APPLICATION_RECOVERY | CAPSTONE_APPLICATION_DELEGATE) ||
      out->exchange_bytes < 4096 || out->exchange_bytes > 1024u * 1024u * 1024u ||
      out->exchange_bytes % 4096 || out->contexts > CAPSTONE_DELEGATE_CONTEXTS_MAX ||
      (1 + out->contexts) * out->exchange_bytes > 1024u * 1024u * 1024u)
    return ENOEXEC;
  return 0;
}

int capstone_application_open(const char *path) {
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return -1;
  struct stat st;
  int error = fstat(fd, &st) ? errno : S_ISREG(st.st_mode) ? 0 : ENOEXEC;
  if (error) {
    close(fd);
    errno = error;
    return -1;
  }
  return fd;
}

int capstone_application_fixed(int fd) {
  /* A file with a name (tmpfs, ext4) was opened for writing through
   * that name, and Linux counted every such writer: the denial excludes
   * them all. A memfd is created unlinked (__shmem_file_setup clears its
   * link count) and its creating descriptor is not counted in i_writecount,
   * so an unlinked, sealable file must be sealed. That also refuses an
   * unsealed tmpfs file that was unlinked (or made with O_TMPFILE), whose
   * writers are counted: the link count cannot tell it from a memfd, and
   * refusing is the safe side. An ordinary tmpfs file answers F_GET_SEALS
   * too (F_SEAL_SEAL), so the seals alone cannot tell either. */
  struct stat st;
  if (fstat(fd, &st))
    return -1;
  if (st.st_nlink)
    return 0;
  int seals = fcntl(fd, F_GET_SEALS);
  int needed = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK;
  if (seals >= 0 && (seals & needed) != needed) {
    errno = ETXTBSY;
    return -1;
  }
  return 0;
}

int capstone_application_inspect(int fd, struct capstone_application_descriptor_v2 *out) {
  struct stat st;
  if (fstat(fd, &st))
    return -1;
  if (!S_ISREG(st.st_mode) || st.st_size < (off_t)sizeof(Elf64_Ehdr) ||
      st.st_size > 512 * 1024 * 1024) {
    errno = ENOEXEC;
    return -1;
  }
  void *data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED)
    return -1;
  int rc = capstone_application_validate(data, (size_t)st.st_size, out);
  int error = errno;
  munmap(data, (size_t)st.st_size);
  errno = error;
  return rc;
}

int capstone_application_validate(const void *data, size_t size,
                                  struct capstone_application_descriptor_v2 *out) {
  int error = inspect(data, size, out);
  if (error) {
    errno = error;
    return -1;
  }
  return 0;
}

int capstone_application_copy(int source, struct capstone_application_descriptor_v2 *out) {
  int image = -1, error = 0;
  struct stat st;
  if (fstat(source, &st)) {
    error = errno;
    goto done;
  }
  if (st.st_size < (off_t)sizeof(Elf64_Ehdr) || st.st_size > 512 * 1024 * 1024) {
    error = ENOEXEC;
    goto done;
  }
  image = memfd_create("capstone-application", MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (image < 0) {
    error = errno;
    goto done;
  }
  /* pread from the start: the copy is of this open file, whatever its offset,
   * never of whatever its path names by now. */
  char buf[65536];
  off_t at = 0;
  while (at < st.st_size) {
    size_t wanted = st.st_size - at < (off_t)sizeof buf ? (size_t)(st.st_size - at) : sizeof buf;
    ssize_t n = pread(source, buf, wanted, at);
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
    at += n;
  }
  if (fcntl(image, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL)) {
    error = errno;
    goto done;
  }
  if (capstone_application_inspect(image, out))
    error = errno;
done:
  if (error) {
    if (image >= 0)
      close(image);
    errno = error;
    return -1;
  }
  return image;
}

int capstone_application_image(const char *path,
                               struct capstone_application_descriptor_v2 *out) {
  int source = capstone_application_open(path);
  if (source < 0)
    return -1;
  int image = capstone_application_copy(source, out);
  int error = errno;
  close(source);
  errno = error;
  return image;
}
