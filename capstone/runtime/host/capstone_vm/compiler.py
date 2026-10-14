#!/usr/bin/env python3
"""Compiler driver for a CMake-built Capstone application SDK.

Use the generated SDK's capstone-cc as CC in an upstream build. Its adjacent
sdk.json records the compiler, headers and runtime that CMake actually built.
CAPSTONE_SDK may select another SDK directory. No port names or entry adapters.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import tempfile

MUSL_LIBS = {"c", "m", "pthread", "rt", "dl", "util", "crypt", "xnet", "resolv"}
PAIRED = {"-D", "-U", "-I", "-isystem", "-iquote", "-include", "-imacros",
          "-idirafter", "-Xclang", "-mllvm", "-isysroot"}

# A direct-call target must be non-linear before the register allocator can
# copy it. The old backend emitted a bare cincoffset; a later movc consumed the
# linear source and the next call faulted. This check qualifies the compiler
# binary used by every SDK, even when the source checkout is sparse or newer.
CALL_TARGET_PROBE = """
@total = internal addrspace(200) global i32 0, align 4
define internal void @f(i32 %x) addrspace(200) noinline {
  %t = load i32, ptr addrspace(200) @total
  %s = add i32 %t, %x
  store i32 %s, ptr addrspace(200) @total
  ret void
}
define void @g() addrspace(200) {
  call addrspace(200) void @f(i32 1)
  call addrspace(200) void @f(i32 2)
  ret void
}
"""

# The virtual profile runs F and D in hardware (cmake/VirtualCodegen.cmake).
# Its FP constants live in the constant pool, which only a compiler that
# addresses the pool through a capability compiles (C-80): older ones crash
# on both functions below.
FLOAT_PROBE = """
double scale(double x) { return x * 3.14159; }
double pick(int c) { return c ? 1.5 : 2.5; }
"""


def check_toolchain(cc: str, codegen: list[str] | tuple[str, ...] = ()) -> None:
    result = subprocess.run([cc, "-S", "-x", "ir", "-target", "capstone64-unknown-elf",
                             "-O2", "-o", "-", "-"], input=CALL_TARGET_PROBE,
                            text=True, capture_output=True)
    if result.returncode:
        raise ValueError(f"compiler codegen probe failed: {result.stderr.strip()}")
    if "\ng:" not in result.stdout:
        raise ValueError("compiler codegen probe did not emit the direct-call function")
    body = result.stdout.split("\ng:", 1)[1].split(".Lfunc_end", 1)[0]
    target = re.search(r"^\s*cincoffset\s+(\w+),\s*gp,\s*\w+\s*\n"
                       r"\s*delin\s+\1\s*$", body, re.MULTILINE)
    if not target or not re.search(r"cjalr\s+ra,\s*0\(" + target[1] + r"\)", body):
        raise ValueError("compiler emits a linear direct-call target (C-46); rebuild the toolchain")
    # Port patches use this feature macro to select capability-preserving
    # pointer arithmetic. Without it they compile an integer-only fallback
    # that links successfully and faults on the first symbol/Datum access.
    result = subprocess.run([cc, '-fsyntax-only', '-x', 'c', '-target',
                             'capstone64-unknown-elf', *codegen, '-'], text=True, capture_output=True,
                            input='#if __SIZEOF_INTCAP__ != 16\n#error intcap required\n#endif\n'
                                  '_Static_assert(sizeof(__uintcap_t) == 16, "intcap");\n'
                                  '__uintcap_t f(void *p) { return (__uintcap_t)p; }\n')
    if result.returncode:
        raise ValueError('application SDK requires the intcap compiler extensions (with the SDK '
                         'code generation); use a qualified toolchain: '
                         + (result.stderr.strip().splitlines() or [''])[0])
    if "+d" in codegen:
        for mode in (["-S", "-o", "-"], ["-c", "-o", os.devnull]):
            result = subprocess.run([cc, *mode, "-x", "c", "-target", "capstone64-unknown-elf",
                                     "-O2", *codegen, "-"], input=FLOAT_PROBE, text=True,
                                    capture_output=True)
            if result.returncode:
                lines = result.stderr.strip().splitlines()
                first = next((line for line in lines if "error" in line),
                             lines[-1] if lines else "no output")
                raise ValueError("compiler cannot compile hard-float code (C-80): " + first)
            if mode[0] == "-S" and not re.search(r"^\s*fmul\.d\s", result.stdout, re.MULTILINE):
                raise ValueError("compiler emits no hardware floating point for the SDK's codegen")


# ---- the profile ----------------------------------------------------------
# sdk.json names its profile, and its codegen fixes the profile's code
# generation: virtual C runs F and D in hardware under lp64 and is gp-free;
# physical has none of these. A caller may repeat a setting, never change it.
PROFILES = ("physical", "virtual")
# Features that select the floating-point or vector unit. Virtual C decodes F
# and D only; physical, none of them.
FP_FEATURE = re.compile(r"^(f|d|q|zfh|zfhmin|zfa|zfbfmin|zfinx|zdinx|zhinx|zhinxmin|v|zv.*)$")
PROFILE_LLVM_OPTIONS = ("capstone-gp-free", "capstone-image-gp")
# Every gp ABI option (gp-free, image-gp, gp-captable ...) is the profile's:
# absent from its codegen means off.
PROFILE_LLVM_OPTION = re.compile(r"^capstone-(gp-.*|image-gp)$")
# Assembler options that select the architecture, its extensions or the ABI.
ASSEMBLER_PROFILE = ("-march", "-mabi", "-mattr", "-mcpu", "-mfloat", "-target", "-triple", "-mllvm")


def llvm_bool(name: str, value: str, what: str) -> bool:
    """A boolean cl::opt value as LLVM parses it; anything else is refused."""
    if value in ("", "true", "TRUE", "True", "1"):
        return True
    if value in ("false", "FALSE", "False", "0"):
        return False
    raise ValueError(f"{what}: -{name}={value}: not a boolean")


def parse_codegen(words: list[str], what: str) -> tuple[dict, str | None, dict]:
    """Target features (name -> '+' or '-', last wins), target ABI and LLVM
    options named in a driver argument list, through -Xclang and -mllvm."""
    cc1, llvm = [], []
    items = iter(words)
    for word in items:
        if word == "-Xclang":
            cc1.append(next(items))
        elif word.startswith("-Xclang="):
            cc1.append(word[len("-Xclang="):])
        elif word == "-mllvm":
            llvm.append(next(items))
        elif word.startswith("-mllvm="):
            llvm.append(word[len("-mllvm="):])
    features, abi = {}, None
    items = iter(cc1)
    for word in items:
        if word == "-target-feature":
            value = next(items, "")
            if len(value) < 2 or value[0] not in "+-":
                raise ValueError(f"{what}: malformed -target-feature {value!r}")
            name = value[1:].removeprefix("experimental-")
            features[name] = value[0]
        elif word == "-target-abi":
            abi = next(items, None)
            if not abi:
                raise ValueError(f"{what}: -target-abi without a value")
        elif word == "-mllvm":
            llvm.append(next(items, ""))
        elif word in ("-target-cpu", "-triple", "-target-feature=", "-cc1", "-cc1as") or \
                word.startswith(("-target-cpu=", "-triple=", "-target-feature=", "-target-abi=")):
            raise ValueError(f"{what}: {word} selects the target; the SDK profile fixes it")
    options = {}
    for word in llvm:
        name, _, value = word.lstrip("-").partition("=")
        if PROFILE_LLVM_OPTION.match(name):
            options[name] = llvm_bool(name, value, what)
    return features, abi, options


def check_profile(config: dict) -> tuple[dict, str | None, dict]:
    """The SDK's own codegen must be its profile's, exactly."""
    profile = config.get("profile")
    if profile not in PROFILES:
        raise ValueError("sdk.json names no profile; rebuild the SDK")
    features, abi, options = parse_codegen(config.get("codegen", []), "sdk.json codegen")
    virtual = profile == "virtual"
    want = {"features": {"f": "+", "d": "+"} if virtual else {},
            "abi": "lp64" if virtual else None,
            "options": {name: True for name in PROFILE_LLVM_OPTIONS} if virtual else {}}
    got = {"features": {k: v for k, v in features.items() if FP_FEATURE.match(k)},
           "abi": abi, "options": options}
    if got != want:
        raise ValueError(f"sdk.json codegen is not the {profile} profile's ({got} != {want}); rebuild the SDK")
    return features, abi, options


def check_arguments(arguments: list[str], config: dict) -> None:
    """Refuse a caller argument that would change the SDK profile's code
    generation, on the compile and the link path alike, before anything runs."""
    features, abi, options = check_profile(config)
    profile = config["profile"]
    for word in arguments:
        # capstone-cc expanded its own response files; one left inside an
        # option's value (-Xclang=@x, -Wa,@x, -mllvm @x) would be expanded by
        # cc1 or the assembler after this check.
        if word.startswith("@") or "=@" in word or ",@" in word:
            raise ValueError(f"{word}: a response file inside an option is not supported")
    # C ABI switches: objects built with them disagree with the SDK's libc and
    # runtime on struct layout, enum size or wchar_t, and no ELF record shows it.
    cc1 = [w for w in arguments if w.startswith("-Xclang=")] + \
          [b for a, b in zip(arguments, arguments[1:]) if a == "-Xclang"]
    for word in [*arguments, *(w.removeprefix("-Xclang=") for w in cc1)]:
        if word.startswith(("-fpack-struct", "-fshort-enums", "-fshort-wchar", "-fwchar-type",
                            "-fno-signed-wchar", "-fms-layout-compatibility", "-mms-bitfields")):
            raise ValueError(f"{word}: changes the C ABI the SDK's libc and runtime were built with")
    # cc1 options through -Xclang: only those the ports need. Any other could
    # change layout or code generation below every check here.
    cc1, items = [], iter(arguments)
    for word in items:
        if word == "-Xclang":
            cc1.append(next(items, ""))
        elif word.startswith("-Xclang="):
            cc1.append(word[len("-Xclang="):])
    items = iter(cc1)
    for word in items:
        if word in ("-target-feature", "-target-abi", "-mllvm"):
            next(items, None)
        elif not word.startswith("-fexperimental-assignment-tracking="):
            raise ValueError(f"-Xclang {word}: not a cc1 option capstone-cc passes through")
    # Driver -m options select the target, its width and ABI (-m32 is capstone32):
    # only these pass.
    for word in arguments:
        if word.startswith("-m") and not word.startswith("-mllvm") and \
                word not in ("-mno-relax", "-mrelax") and not word.startswith("-mcmodel="):
            raise ValueError(f"{word}: the {profile} SDK profile fixes the target; not a -m option capstone-cc passes through")
    words = iter(arguments)
    for word in words:
        if word in ("-cc1", "-cc1as", "--config", "-target", "-Xarch_device", "-Xarch_host") or \
                word.startswith(("--driver-mode", "--config=", "--config-system-dir", "--config-user-dir", "--target=",
                                 "-march=", "-mabi=", "-mcpu=", "-mfloat-abi=", "-flto", "-emit-llvm",
                                 "-Xarch_")) or \
                word in ("-msoft-float", "-mhard-float", "-mno-soft-float", "-mno-hard-float"):
            raise ValueError(f"{word}: the {profile} SDK profile fixes the target and its code generation")
        assembler = []
        if word.startswith("-Wa,"):
            assembler = word[4:].split(",")
        elif word in ("-Xassembler", "-Xclangas"):
            assembler = [next(words, "")]
        elif word.startswith("-Xclangas="):
            assembler = [word[len("-Xclangas="):]]
        for item in assembler:
            if item.startswith(ASSEMBLER_PROFILE):
                raise ValueError(f"{word}: {item} would change the {profile} profile's architecture")
    caller_features, caller_abi, caller_options = parse_codegen(arguments, "argument")
    # Target features: the profile's own (repeated with the same sign), the
    # base +m and +a every compile has, and the MOVC copy rule; nothing else
    # (+e alone selects another ABI).
    base = {"m": "+", "a": "+", **features}
    for name, sign in caller_features.items():
        if name == "movc-keeps-integer-source":
            continue
        if base.get(name) != sign:
            raise ValueError(f"-target-feature {sign}{name}: the {profile} SDK profile "
                             + (f"sets {base[name]}{name}" if name in base else f"has no {name}"))
    if caller_abi is not None and caller_abi != abi:
        raise ValueError(f"-target-abi {caller_abi}: the {profile} SDK profile's ABI is {abi or 'the default'}")
    for name, value in caller_options.items():
        if value != options.get(name, False):
            raise ValueError(f"-mllvm -{name}={'true' if value else 'false'}: not the {profile} profile's")


# ---- link inputs -------------------------------------------------------------
# Every object and archive member a link can use, the SDK's own archives
# included, must be a Capstone relocatable object built for the profile.
# C objects record their ISA in .capstone.attributes; assembly records none
# and is accepted on its arch flags alone (it can still contain any
# instruction). The capability call convention and the gp choice are not
# recorded at all, so a physical object in a virtual link (or the reverse)
# passes when its FP use matches: this detects stale or mismatched builds,
# not every incompatible object.
EM_CAPSTONE = 259
EF_FLOAT_ABI = 0x6
EF_RVE = 0x8
# Called by soft-float code for binary32 and binary64; hardware F and D code
# never needs them. binary128 helpers (__addtf3 ...) stay legitimate.
SOFT_FLOAT_HELPERS = frozenset(
    f"__{op}{t}" for t in ("sf3", "df3") for op in ("add", "sub", "mul", "div")) | frozenset(
    f"__{op}{t}2" for t in ("sf", "df") for op in ("eq", "ne", "lt", "le", "gt", "ge", "unord", "neg")) | frozenset({
    "__fixsfsi", "__fixsfdi", "__fixunssfsi", "__fixunssfdi", "__fixdfsi", "__fixdfdi",
    "__fixunsdfsi", "__fixunsdfdi", "__floatsisf", "__floatdisf", "__floatunsisf",
    "__floatundisf", "__floatsidf", "__floatdidf", "__floatunsidf", "__floatundidf",
    "__extendsfdf2", "__truncdfsf2"})


def uleb(data: bytes, at: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if at >= len(data) or shift > 63:
            raise ValueError("malformed ULEB128")
        byte = data[at]; at += 1
        value |= (byte & 0x7f) << shift; shift += 7
        if not byte & 0x80:
            return value, at


def attribute_arches(data: bytes, what: str) -> list[str]:
    """Tag_arch strings of a .capstone.attributes section (RISC-V layout:
    'A', vendor subsections, file-scope records; even tags ULEB, odd strings)."""
    if not data or data[0] != ord("A"):
        raise ValueError(f"{what}: malformed .capstone.attributes")
    arches, at = [], 1
    while at < len(data):
        if at + 4 > len(data):
            raise ValueError(f"{what}: malformed .capstone.attributes")
        length = struct.unpack_from("<I", data, at)[0]
        end = at + length
        if length < 5 or end > len(data):
            raise ValueError(f"{what}: malformed .capstone.attributes")
        nul = data.index(0, at + 4, end)
        vendor = data[at + 4:nul].decode(errors="replace")
        inner = nul + 1
        while vendor == "capstone" and inner < end:
            if inner + 5 > end:
                raise ValueError(f"{what}: malformed .capstone.attributes")
            tag, size = data[inner], struct.unpack_from("<I", data, inner + 1)[0]
            stop = inner + size
            if size < 5 or stop > end:
                raise ValueError(f"{what}: malformed .capstone.attributes")
            cursor = inner + 5
            if tag != 1:
                raise ValueError(f"{what}: .capstone.attributes scope {tag} is not the file's")
            while cursor < stop:
                attribute, cursor = uleb(data, cursor)
                if attribute % 2 == 0:
                    _, cursor = uleb(data, cursor)
                else:
                    nul = data.index(0, cursor, stop)
                    if attribute == 5:
                        arches.append(data[cursor:nul].decode(errors="strict"))
                    cursor = nul + 1
            inner = stop
        at = end
    return arches


def arch_extensions(arch: str, what: str) -> set[str]:
    match = re.fullmatch(r"rv64([a-z0-9]*)((?:_[a-z][a-z0-9]*)*)", arch)
    if not match:
        raise ValueError(f"{what}: unexpected ISA string {arch!r}")
    extensions = set(re.findall(r"([a-z])(?:\d+p\d+)?", match.group(1)))
    for token in match.group(2).split("_")[1:]:
        extensions.add(re.sub(r"\d+p\d+$", "", token))
    return extensions


def check_object(data: bytes, what: str, virtual: bool, helpers_allowed: bool = False) -> None:
    try:
        check_elf(data, what, virtual, helpers_allowed)
    except (IndexError, struct.error, UnicodeDecodeError) as error:
        raise ValueError(f"{what}: malformed ELF object ({error})") from None


SHT_SYMTAB, SHT_NOBITS, SHT_LLVM_DEPENDENT_LIBRARIES = 2, 8, 0x6FFF4C04


def check_elf(data: bytes, what: str, virtual: bool, helpers_allowed: bool) -> None:
    if data[:4] == b"BC\xc0\xde" or data[:4] == b"\xde\xc0\x17\x0b":
        raise ValueError(f"{what}: LLVM bitcode (LTO) is not a supported link input")
    if data[:4] != b"\x7fELF":
        raise ValueError(f"{what}: not an ELF object")
    if len(data) < 64 or data[4] != 2 or data[5] != 1:
        raise ValueError(f"{what}: not a little-endian ELF64 object")
    e_type, e_machine = struct.unpack_from("<HH", data, 16)
    e_flags = struct.unpack_from("<I", data, 48)[0]
    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 58)
    if e_type != 1 or e_machine != EM_CAPSTONE:
        raise ValueError(f"{what}: not a Capstone relocatable object (type {e_type}, machine {e_machine})")
    if e_flags & EF_FLOAT_ABI:
        raise ValueError(f"{what}: built for a hardware-float calling convention; the SDK's is lp64")
    if e_flags & EF_RVE:
        raise ValueError(f"{what}: built for the E base ISA (lp64e); the SDK's is lp64")
    if e_shentsize != 64 or not e_shnum or e_shoff + e_shnum * 64 > len(data) or e_shstrndx >= e_shnum:
        raise ValueError(f"{what}: malformed section headers")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + i * 64) for i in range(e_shnum)]
    def contents(header):
        offset, size = header[4], header[5]
        if header[1] == SHT_NOBITS:
            return b""
        if offset + size > len(data):
            raise ValueError(f"{what}: section beyond the file")
        return data[offset:offset + size]
    names = contents(sections[e_shstrndx])
    def name(header):
        end = names.find(b"\0", header[0])
        return names[header[0]:end].decode(errors="replace")
    for header in sections:
        if header[1] == SHT_LLVM_DEPENDENT_LIBRARIES:
            raise ValueError(f"{what}: names dependent libraries (.deplibs); link them explicitly")
        if name(header) == ".capstone.attributes":
            for arch in attribute_arches(contents(header), what):
                extensions = arch_extensions(arch, what)
                if "e" in extensions:
                    raise ValueError(f"{what}: ISA {arch}: the E base ISA is not the SDK's")
                fp = {e for e in extensions if FP_FEATURE.match(e)}
                if virtual and fp != {"f", "d"}:
                    raise ValueError(f"{what}: ISA {arch}: the virtual profile needs F and D and nothing else of the FP or vector units")
                if not virtual and fp:
                    raise ValueError(f"{what}: ISA {arch}: the physical profile has no FP or vector unit")
        if virtual and not helpers_allowed and header[1] == SHT_SYMTAB:
            if header[6] >= e_shnum or header[9] != 24:
                raise ValueError(f"{what}: malformed symbol table")
            strtab = contents(sections[header[6]])
            table = contents(header)
            for offset in range(24, len(table) - 23, 24):
                st_name, _, _, st_shndx = struct.unpack_from("<IBBH", table, offset)
                if st_shndx == 0 and st_name:
                    symbol = strtab[st_name:strtab.find(b"\0", st_name)].decode(errors="replace")
                    if symbol in SOFT_FLOAT_HELPERS:
                        raise ValueError(f"{what}: calls {symbol}: soft-float code in a hard-float link")


def check_archive(path: Path, virtual: bool, helpers_allowed: bool = False) -> None:
    data = path.read_bytes()
    if data.startswith(b"!<thin>\n"):
        raise ValueError(f"{path}: thin archives are not supported link inputs")
    at, long_names = 8, b""
    while at < len(data):
        if at + 60 > len(data) or data[at + 58:at + 60] != b"`\n":
            raise ValueError(f"{path}: malformed archive")
        header = data[at:at + 60]
        raw = header[:16].decode(errors="replace").rstrip()
        size = int(header[48:58].decode().strip() or "-1")
        body = at + 60
        if size < 0 or body + size > len(data):
            raise ValueError(f"{path}: malformed archive")
        member = data[body:body + size]
        if raw.startswith("#1/"):  # BSD: the name precedes the contents
            length = int(raw[3:])
            raw, member = member[:length].rstrip(b"\0").decode(errors="replace"), member[length:]
        if raw == "//":
            long_names = member
        elif raw in ("/", "/SYM64/", "__.SYMDEF", "__.SYMDEF SORTED", "__.SYMDEF_64",
                     "__.SYMDEF_64 SORTED"):
            pass  # the symbol table, exactly; any other name is a member
        else:
            if raw.startswith("/") and raw[1:].isdigit():
                start = int(raw[1:])
                raw = long_names[start:long_names.find(b"\n", start)].decode(errors="replace")
            check_object(member, f"{path}({raw.rstrip('/')})", virtual, helpers_allowed)
        at = body + size + (size & 1)


def check_link_inputs(paths: list[Path], config: dict) -> None:
    virtual = config["profile"] == "virtual"
    # The SDK's builtins archive defines the soft-float helpers, and some call
    # each other (__subdf3 calls __adddf3); only its ISA and ABI are checked.
    provider = Path(config["builtins"]).resolve() if config.get("builtins") else None
    for path in paths:
        helpers_allowed = path.resolve() == provider
        with open(path, "rb") as stream:
            magic = stream.read(8)
        if magic.startswith(b"!<arch>\n") or magic.startswith(b"!<thin>\n"):
            check_archive(path, virtual, helpers_allowed)
        else:
            check_object(path.read_bytes(), str(path), virtual, helpers_allowed)


# Linker options a caller may pass (-Wl, -Xlinker): none of them adds an
# input. Anything else is refused, since lld has many spellings that do
# (--library=, -script=, INPUT() in a script, @file, --just-symbols ...).
LINKER_FLAGS = {"--gc-sections", "--no-gc-sections", "--no-undefined", "--noinhibit-exec",
                "--start-group", "--end-group", "--whole-archive", "--no-whole-archive",
                "--fatal-warnings", "--no-fatal-warnings", "--print-gc-sections", "-Bstatic",
                "-static", "--warn-common", "--no-warn-mismatch", "--enable-new-dtags",
                "--disable-new-dtags", "--export-dynamic", "-E", "--as-needed", "--no-as-needed",
                "--eh-frame-hdr", "--build-id", "-O0", "-O1", "-O2", "--strip-all", "-s",
                "--strip-debug", "-S", "--relax", "--no-relax"}
LINKER_VALUES = {"--wrap", "--undefined", "-u", "--defsym", "--unresolved-symbols", "--error-limit",
                 "-Map", "--Map", "-e", "--entry", "-z", "-rpath", "--rpath", "--hash-style",
                 "--build-id", "--icf"}


def link_files(inputs: list[str]) -> tuple[list[Path], list[str]]:
    """The files a link reads, and the linker arguments that read exactly
    those: objects, archives, and -l libraries resolved here as lld would
    (static only) and handed to it as paths, so lld never interprets a
    -l or -L spelling itself (it also reads single-dash long options such
    as -library=). Linker options outside the allowlist, which could add
    inputs the check cannot see, are refused, as is any response file."""
    for item in inputs:
        if item.startswith("@"):
            raise ValueError(f"linker response file {item}: inputs the SDK cannot check are not supported")
    files, search, libraries, arguments = [], [], [], []
    items = iter(inputs)
    for item in items:
        if item in LINKER_FLAGS:
            arguments.append(item)
            continue
        if item in LINKER_VALUES:
            value = next(items, None)
            if value is None:
                raise ValueError(f"linker {item} without a value")
            arguments.extend((item, value))
            continue
        if item.startswith("-") and not item.startswith(("-L", "-l")):
            option = item.split("=", 1)[0]
            if "=" in item and option in LINKER_VALUES:
                arguments.append(item)
                continue
            raise ValueError(f"linker option {item} is not supported by capstone-cc")
        if item == "-L" or item.startswith("-L"):
            directory = next(items, None) if item == "-L" else item[2:]
            if not directory or directory.startswith("="):
                raise ValueError(f"linker {item}: a search directory is required (no sysroot prefix)")
            search.append(directory)
        elif item == "-l" or item.startswith("-l"):
            name = next(items, None) if item == "-l" else item[2:]
            if not name:
                raise ValueError("linker -l without a library")
            libraries.append(name)
            arguments.append(("library", name))
        else:
            files.append(Path(item))
            arguments.append(item)
    resolved = {}
    for name in libraries:
        names = [name[1:]] if name.startswith(":") else [f"lib{name}.so", f"lib{name}.a"]
        found = next((Path(d) / n for d in search for n in names if (Path(d) / n).is_file()), None)
        if found is None:
            raise ValueError(f"-l{name}: not found in {search}")
        if found.suffix == ".so":
            raise ValueError(f"-l{name}: {found} is a shared library; application domains link statically")
        files.append(found)
        # Absolute, so a resolved name can never read as an option (-l:-Tx, -l:@x).
        resolved[name] = str(found.resolve())
    return files, [resolved[a[1]] if isinstance(a, tuple) else a for a in arguments]


# Driver options whose value is the next argument; a mode word such as -c
# in that position is the value, not the mode.
VALUE_OPTIONS = PAIRED | {"-o", "-x", "-MF", "-MT", "-MQ", "-Xlinker", "-Xassembler", "-Xclangas",
                          "-Xpreprocessor", "-L", "-l", "-T", "-u", "-e", "-z", "--param", "-arch"}
COMPILE_ONLY = ("-c", "-S", "-E", "-M", "-MM", "--version", "-dumpmachine", "-dumpversion",
                "--help", "-###")
# Options that would let clang link, or pick the tools it links with, outside
# capstone-cc's checked link.
LINK_FORWARDING = ("-Wl,", "-Xlinker", "-T", "-fuse-ld", "--ld-path", "-B", "--script")


def compile_only(arguments: list[str]) -> bool:
    """Whether clang is asked not to link: a mode word in an option position."""
    words = iter(arguments)
    for word in words:
        if word in VALUE_OPTIONS:
            value = next(words, "")
            if word == "-o" and value.startswith(("-", "@")):
                raise ValueError(f"-o {value}: an output named like an option is not supported")
            continue
        if word in COMPILE_ONLY or word.startswith(("-print-", "--print-")):
            return True
    return arguments == ["-v"]


LAYOUT_FLAGS = ("-fpack-struct", "-fshort-enums", "-fshort-wchar", "-fwchar-type", "-fno-signed-wchar",
                "-fms-layout-compatibility", "-mms-bitfields", "-fno-bitfield-type-align",
                "-fmax-type-align")


def check_cc1(job: list[str], config: dict) -> None:
    """The cc1/cc1as command clang plans must be the profile's, whatever
    route (-Xclang, -Wp, -Xpreprocessor, ...) its arguments took."""
    features, abi, options = check_profile(config)
    allowed = {"m": "+", "a": "+", "movc-keeps-integer-source": "+", **features}
    planned, planned_options, items = {}, {}, iter(job[1:])
    for word in items:
        if word == "-triple":
            triple = next(items, "")
            if triple != "capstone64-unknown-unknown-elf":
                raise ValueError(f"clang would compile for {triple}; the SDK's target is capstone64")
        elif word == "-target-feature":
            value = next(items, "")
            planned[value[1:].removeprefix("experimental-")] = value[:1]
        elif word == "-target-cpu" or word.startswith("-target-cpu="):
            # The SDK names no CPU; one would bring its own extensions.
            raise ValueError("clang would compile for a named CPU; the SDK profile fixes the extensions")
        elif word == "-target-abi":
            if next(items, "") != (abi or "lp64"):
                raise ValueError("clang would use another ABI than the SDK profile's")
        elif word == "-mllvm":
            name, _, value = next(items, "").lstrip("-").partition("=")
            if PROFILE_LLVM_OPTION.match(name):
                planned_options[name] = llvm_bool(name, value, "clang's plan")
        elif word.startswith(LAYOUT_FLAGS):
            raise ValueError(f"clang would compile with {word}, which changes the SDK's C ABI")
    for name, sign in planned.items():
        if allowed.get(name) != sign:
            raise ValueError(f"clang would compile with {sign}{name}, not the {config['profile']} profile's features")
    # cc1as receives no target features from this driver; cc1 must carry the profile's.
    if job[1] == "-cc1" and any(planned.get(name) != sign for name, sign in features.items()):
        raise ValueError(f"clang would compile without the {config['profile']} profile's features")
    if any(planned_options.get(name, False) != value for name, value in
           {**{n: False for n in planned_options}, **options}.items()):
        raise ValueError(f"clang would compile with gp options other than the {config['profile']} profile's")


def runs_only_clang(target: list[str], arguments: list[str], config: dict) -> None:
    """Refuse a direct invocation for which clang plans any job but its own
    cc1/cc1as, a link above all, or a cc1/cc1as job that is not the profile's.
    clang's own parser decides, not lists of options and their spellings."""
    plan = subprocess.run([*target, *arguments, "-###"], capture_output=True, text=True)
    if plan.returncode:
        return  # clang will refuse the same arguments itself
    own = os.path.realpath(target[0])
    for line in plan.stderr.splitlines():
        if line.startswith(' "'):
            job = shlex.split(line)
            if os.path.realpath(job[0]) != own:
                raise ValueError(f"this invocation would run {job[0]}; a link goes through capstone-cc's checked link")
            if len(job) > 1 and job[1] in ("-cc1", "-cc1as"):
                check_cc1(job, config)


def expand(arguments: list[str], depth: int = 0) -> list[str]:
    if depth > 8:
        raise ValueError("response files nest too deeply")
    result = []
    for arg in arguments:
        result.extend(expand(shlex.split(Path(arg[1:]).read_text()), depth + 1)
                      if arg.startswith("@") else [arg])
    return result


def link_arguments(arguments: list[str]) -> tuple[list[str], list[str], list[str], str]:
    """Separate compiler options, source files and ordered linker inputs."""
    compile_flags, sources, inputs = [], [], []
    output = "a.out"
    words = iter(arguments)
    for arg in words:
        if arg == "-o":
            output = next(words)
        elif arg.startswith("-o") and len(arg) > 2:
            output = arg[2:]
        elif arg in PAIRED:
            compile_flags.extend((arg, next(words)))
        elif arg == "-Xlinker":
            inputs.append(next(words))
        elif arg in ("-L", "-l"):
            value = next(words)
            if arg != "-l" or value not in MUSL_LIBS:
                inputs.extend((arg, value))
        elif arg.startswith("-l"):
            if arg[2:] not in MUSL_LIBS:
                inputs.append(arg)
        elif arg.startswith("-Wl,"):
            inputs.extend(arg[4:].split(","))
        elif arg.startswith("-L") or arg.endswith((".o", ".a", ".lo")):
            inputs.append(arg)
        elif arg.endswith((".c", ".s", ".S")):
            # Preserve source/library ordering, including repeated basenames.
            inputs.append("@source:" + str(len(sources)))
            sources.append(arg)
        elif arg in ("-shared", "-pie"):
            raise ValueError("application domains require a static executable")
        elif arg in ("-static", "-no-pie", "-rdynamic", "-nostdlib", "-nodefaultlibs"):
            continue
        elif arg.startswith(("-D", "-U", "-I", "-std=", "-O", "-g", "-f", "-W", "-m")) or arg in (
                "-pthread", "-v", "-Qunused-arguments", "-pedantic", "-pedantic-errors", "-pipe", "-ansi"):
            compile_flags.append(arg)
        else:
            raise ValueError(f"unsupported compiler driver argument: {arg}")
    return compile_flags, sources, inputs, output


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = expand(sys.argv[1:] if argv is None else argv)
        if os.environ.get("CCC_OVERRIDE_OPTIONS"):
            raise ValueError("CCC_OVERRIDE_OPTIONS would edit the compiler's arguments past the profile check; unset it")
        if len(arguments) >= 2 and arguments[0] == "--check-compiler":
            check_toolchain(arguments[1], arguments[2:])
            return 0
        sdk = Path(os.environ.get("CAPSTONE_SDK", Path(sys.argv[0]).absolute().parent))
        config = json.loads((sdk / "sdk.json").read_text())
        if config["version"] != 1 or config.get("application_abi") != 2:
            raise ValueError("unsupported SDK version; rebuild the SDK")
        if arguments == ["--check-toolchain"]:
            check_profile(config)
            check_toolchain(config["cc"], config.get("codegen", []))
            return 0
        check_arguments(arguments, config)
        target = [config["cc"], "--no-default-config", "-target", "capstone64-unknown-elf",
                  "-Xclang", "-target-feature", "-Xclang", "+m",
                  "-Xclang", "-target-feature", "-Xclang", "+a",
                  "-ffreestanding", "-fno-builtin", "-fno-jump-tables", "-nostdinc"]
        target.extend(config.get("codegen", []))
        for suffix in ("arch/capstone64", "arch/generic", "obj/include", "include"):
            target.extend(("-isystem", str(Path(config["musl"]) / suffix)))
        if compile_only(arguments):
            forwarded = next((a for a in arguments if a.startswith(LINK_FORWARDING)), None)
            if forwarded:
                raise ValueError(f"{forwarded}: linker options belong to a link through capstone-cc")
            runs_only_clang(target, arguments, config)
            log = os.environ.get("CAPSTONE_COMPILE_LOG")
            if log and "-c" in arguments and "-o" in arguments:
                obj = arguments[arguments.index("-o") + 1]
                result = subprocess.run([*target, *arguments, "-ferror-limit=0"],
                                        stderr=subprocess.PIPE, text=True)
                Path(obj + ".err").write_text(result.stderr)
                source = next((a for a in arguments if a.endswith((".c", ".S", ".s"))), "")
                first = next((line for line in result.stderr.splitlines() if "error:" in line), "")
                with open(log, "a") as stream:
                    stream.write(f"{obj}\t{result.returncode}\t{source}\t{first}\n")
                if result.returncode:
                    print(result.stderr, file=sys.stderr, end="")
                return result.returncode
            return subprocess.call([*target, *arguments])
        flags, sources, inputs, output = link_arguments(arguments)
        with tempfile.TemporaryDirectory(prefix="capstone-cc-") as scratch:
            for index, source in enumerate(sources):
                obj = str(Path(scratch) / f"{index}.o")
                runs_only_clang(target, [*flags, "-c", source, "-o", obj], config)
                result = subprocess.call([*target, *flags, "-c", source, "-o", obj])
                if result:
                    return result
                inputs[inputs.index("@source:" + str(index))] = obj
            files, linker_inputs = link_files(inputs)
            # The SDK's own inputs, absolute too: a path shaped like an option
            # (-Tx, @x) must reach lld as a file.
            sdk = {k: str(Path(config[k]).resolve()) for k in ("script", "runtime", "libc", "builtins")}
            check_link_inputs([*files, Path(sdk["runtime"]), Path(sdk["libc"]), Path(sdk["builtins"])], config)
            return subprocess.call([config["ld"], "--no-dependent-libraries", "--gc-sections", "-T", sdk["script"],
                                    "-o", str(Path(output).absolute()), "--whole-archive", sdk["runtime"],
                                    "--no-whole-archive", "--start-group", *linker_inputs,
                                    sdk["libc"], sdk["builtins"], "--end-group"])
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f"capstone-cc: {error or 'missing option argument'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
