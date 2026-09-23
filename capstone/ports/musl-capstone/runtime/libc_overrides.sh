# Sourced, not run. The libc overrides: files in this directory that replace a musl
# object whose code cannot run as written on a capability machine. Each defines every
# symbol of the musl object it replaces, and each must be linked AHEAD of musl's
# archive by every domain that uses musl, or the domain silently gets musl's version.
#
#   source "$MUSL_PORT/runtime/libc_overrides.sh"
#   build_musl_overrides <clang> <out-dir> <musl-src-dir> <compile flags...>
#   ... link "${MUSL_OVERRIDE_OBJS[@]}" before libc.a
#
# The list lives here, once, because it used to be copied into every build script
# that links a musl domain: a new override then had to be added to each copy, and a
# copy that missed it linked musl's broken object without a word.
MUSL_OVERRIDES=(
  string_bounds_safe      # string routines: word-at-a-time scans read past the object
  fputwc_null_safe        # __fputwc_unlocked: pointer arithmetic on a null cursor
  mbsrtowcs_bounds_safe   # mbsrtowcs: the same word-at-a-time scan
  atexit_capability_safe  # atexit: the handler through uintptr_t lost its tag
)
build_musl_overrides() {
  local cc=$1 out=$2 musl=$3; shift 3
  local here f
  here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  MUSL_OVERRIDE_OBJS=()
  for f in "${MUSL_OVERRIDES[@]}"; do
    local extra=()
    [[ $f == mbsrtowcs_bounds_safe ]] && extra=(-I"$musl/src/multibyte")
    "$cc" "$@" "${extra[@]}" -c "$here/$f.c" -o "$out/$f.o" || return 1
    MUSL_OVERRIDE_OBJS+=("$out/$f.o")
  done
}
