import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from capstone_vm.compiler import expand, link_arguments


class CompilerTests(unittest.TestCase):
    def test_sources_keep_link_order_and_receive_late_compile_flags(self):
        flags, sources, inputs, output = link_arguments([
            "one/main.c", "libone.a", "two/main.c", "-D", "MESSAGE=two words",
            "-lm", "-lcustom", "-Xlinker", "--no-undefined", "-o", "app.dom"])
        self.assertEqual(flags, ["-D", "MESSAGE=two words"])
        self.assertEqual(sources, ["one/main.c", "two/main.c"])
        self.assertEqual(inputs, ["@source:0", "libone.a", "@source:1", "-lcustom", "--no-undefined"])
        self.assertEqual(output, "app.dom")

    def test_response_file_preserves_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arguments"
            path.write_text("'a b.o' -D 'VALUE=a b' -o 'output file.dom'")
            self.assertEqual(expand(["@" + str(path)]),
                             ["a b.o", "-D", "VALUE=a b", "-o", "output file.dom"])

    def test_unknown_and_dynamic_options_fail(self):
        for option in ("--imaginary", "-shared", "-pie"):
            with self.assertRaises(ValueError):
                link_arguments(["main.o", option])

    def test_response_cycles_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle"
            path.write_text("@" + str(path))
            with self.assertRaises(ValueError):
                expand(["@" + str(path)])


from capstone_vm.compiler import (check_arguments, check_profile, check_object,
                                  check_link_inputs, link_files)
import struct

VIRTUAL = {"profile": "virtual", "codegen": [
    "-mllvm", "-capstone-gp-free", "-mllvm", "-capstone-image-gp",
    "-Xclang", "-target-feature", "-Xclang", "+f", "-Xclang", "-target-feature", "-Xclang", "+d",
    "-Xclang", "-target-abi", "-Xclang", "lp64",
    "-Xclang", "-target-feature", "-Xclang", "+movc-keeps-integer-source"]}
PHYSICAL = {"profile": "physical", "codegen": []}


def attributes(arch: str) -> bytes:
    record = b"\x04\x10\x05" + arch.encode() + b"\0"
    scope = b"\x01" + struct.pack("<I", 5 + len(record)) + record
    vendor = b"capstone\0"
    return b"A" + struct.pack("<I", 4 + len(vendor) + len(scope)) + vendor + scope


def elf(arch: str | None = None, undefined: tuple[str, ...] = (), flags: int = 0,
        machine: int = 259, defined: tuple[str, ...] = ()) -> bytes:
    """A minimal ELF64 relocatable object: shstrtab, optional attributes,
    and a symbol table with the given undefined symbols and the given global
    definitions (placed in section 1, which any non-zero index stands for)."""
    names = b"\0.shstrtab\0.capstone.attributes\0.symtab\0.strtab\0"
    strtab = b"\0" + b"".join(s.encode() + b"\0" for s in undefined + defined)
    symtab = bytes(24)
    at = 1
    for s in undefined:
        symtab += struct.pack("<IBBHQQ", at, 0x10, 0, 0, 0, 0)
        at += len(s) + 1
    for s in defined:
        symtab += struct.pack("<IBBHQQ", at, 0x10, 0, 1, 0, 0)
        at += len(s) + 1
    blobs = [names, attributes(arch) if arch else b"", symtab, strtab]
    offsets, data = [], bytearray(64)
    for blob in blobs:
        offsets.append(len(data)); data += blob
    shoff = len(data)
    headers = [bytes(64),
               struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, offsets[0], len(names), 0, 0, 1, 0)]
    if arch:
        headers.append(struct.pack("<IIQQQQIIQQ", 11, 0x70000003, 0, 0, offsets[1], len(blobs[1]), 0, 0, 1, 0))
    strtab_index = len(headers) + 1
    headers.append(struct.pack("<IIQQQQIIQQ", 32, 2, 0, 0, offsets[2], len(symtab), strtab_index, 1, 8, 24))
    headers.append(struct.pack("<IIQQQQIIQQ", 40, 3, 0, 0, offsets[3], len(strtab), 0, 0, 1, 0))
    data += b"".join(headers)
    data[0:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
    struct.pack_into("<HHIQQQIHHHHHH", data, 16, 1, machine, 1, 0, 0, shoff, flags, 64, 0, 0, 64,
                     len(headers), 1)
    return bytes(data)


def archive(members: dict[str, bytes], thin: bool = False) -> bytes:
    long_names = b"".join(n.encode() + b"/\n" for n in members)
    out = bytearray(b"!<thin>\n" if thin else b"!<arch>\n")
    def member(name: str, body: bytes):
        out.extend(name.ljust(16).encode() + b"0".ljust(12) + b"0".ljust(6) + b"0".ljust(6)
                   + b"644".ljust(8) + str(len(body)).ljust(10).encode() + b"`\n" + body)
        if len(body) & 1:
            out.extend(b"\n")
    member("//", long_names)
    at = 0
    for name, body in members.items():
        member(f"/{at}", body)
        at += len(name) + 2
    return bytes(out)


class ProfileArgumentTests(unittest.TestCase):
    def refused(self, config, *arguments):
        with self.assertRaises(ValueError, msg=str(arguments)):
            check_arguments(list(arguments), config)

    def test_profile_codegen_must_match(self):
        check_profile(VIRTUAL)
        check_profile(PHYSICAL)
        # A physical SDK may never carry the virtual MOVC rule, opt-in or not.
        with patch.dict(os.environ, {"CAPSTONE_ALLOW_UNSAFE_MOVC_PROBE": "1"}):
            with self.assertRaises(ValueError):
                check_profile({"profile": "physical", "codegen": [
                    "-Xclang", "-target-feature", "-Xclang", "+movc-keeps-integer-source"]})
        # The MOVC copy rule is not part of the profile.
        check_profile({"profile": "virtual", "codegen": VIRTUAL["codegen"][:-4]})
        for config in ({"codegen": []}, {"profile": "virtual", "codegen": []},
                       {"profile": "physical", "codegen": VIRTUAL["codegen"]},
                       {"profile": "virtual", "codegen": VIRTUAL["codegen"][:-8]}):
            with self.assertRaises(ValueError):
                check_profile(config)

    def test_ordinary_and_repeated_settings_pass(self):
        check_arguments(["-O2", "-c", "a.c", "-mcmodel=medany", "-Wa,--noexecstack",
                         "-Xclang", "-target-feature", "-Xclang", "+movc-keeps-integer-source",
                         "-Xclang", "-target-feature", "-Xclang", "+d",
                         "-Xclang=-target-abi", "-Xclang=lp64", "-mllvm", "-capstone-gp-free=true"], VIRTUAL)
        check_arguments(["-O2", "-c", "a.c", "-mllvm", "-capstone-gp-free=false"], PHYSICAL)

    def test_the_movc_copy_rule_is_the_callers(self):
        # tests/runtime-qemu's movc-rule and movc-keep, through capstone-cc.
        def movc(*signs, equals=False):
            words = ["-O2", "-c", "a.c"]
            for sign in signs:
                words += (["-Xclang=-target-feature", "-Xclang=" + sign + "movc-keeps-integer-source"] if equals
                          else ["-Xclang", "-target-feature", "-Xclang", sign + "movc-keeps-integer-source"])
            return words
        with patch.dict(os.environ):
            os.environ.pop("CAPSTONE_ALLOW_UNSAFE_MOVC_PROBE", None)
            for equals in (False, True):
                for sign in "+-":
                    check_arguments(movc(sign, equals=equals), VIRTUAL)
                check_arguments(movc("-", equals=equals), PHYSICAL)
                # Physical silicon nulls an untagged MOVC source (Q-04 (b)): + is refused,
                # whichever spelling, unless a later - wins.
                self.refused(PHYSICAL, *movc("+", equals=equals))
                self.refused(PHYSICAL, *movc("-", "+", equals=equals))
                check_arguments(movc("+", "-", equals=equals), PHYSICAL)
            # The movc-keep control's opt-in, set for that compile alone.
            os.environ["CAPSTONE_ALLOW_UNSAFE_MOVC_PROBE"] = "1"
            check_arguments(movc("+"), PHYSICAL)
            os.environ["CAPSTONE_ALLOW_UNSAFE_MOVC_PROBE"] = "yes"
            self.refused(PHYSICAL, *movc("+"))
        # Only as a -target-feature value: the driver word is still refused.
        self.refused(VIRTUAL, "-c", "a.c", "-movc-keeps-integer-source")
        self.refused(VIRTUAL, "-c", "a.c", "-Xclang", "-target-feature", "-movc-keeps-integer-source")

    def test_fp_and_abi_changes_are_refused(self):
        for config, feature in ((VIRTUAL, "-d"), (VIRTUAL, "-f"), (VIRTUAL, "+q"), (VIRTUAL, "+v"),
                                (VIRTUAL, "+zfinx"), (VIRTUAL, "+experimental-zfbfmin"),
                                (PHYSICAL, "+f"), (PHYSICAL, "+d"), (PHYSICAL, "+zve32x")):
            self.refused(config, "-c", "a.c", "-Xclang", "-target-feature", "-Xclang", feature)
            self.refused(config, "-c", "a.c", "-Xclang=-target-feature", "-Xclang=" + feature)
        self.refused(VIRTUAL, "-c", "a.c", "-Xclang", "-target-abi", "-Xclang", "lp64d")
        for config in (VIRTUAL, PHYSICAL):
            for feature in ("+e", "+c", "+relax", "-m", "-a"):
                self.refused(config, "-c", "a.c", "-Xclang", "-target-feature", "-Xclang", feature)
            check_arguments(["-c", "a.c", "-Xclang", "-target-feature", "-Xclang", "+m"], config)
        self.refused(PHYSICAL, "-c", "a.c", "-Xclang", "-target-abi", "-Xclang", "lp64")
        self.refused(VIRTUAL, "-c", "a.c", "-Xclang", "-target-feature")
        self.refused(VIRTUAL, "-c", "a.c", "-Xclang", "-target-cpu", "-Xclang", "x")
        self.refused(VIRTUAL, "-c", "a.c", "-Xclang", "-triple", "-Xclang", "riscv64")

    def test_driver_and_assembler_spellings_are_refused(self):
        for word in ("-march=rv64gc", "-mabi=lp64d", "-mcpu=x", "-mfloat-abi=hard", "-msoft-float",
                     "-mhard-float", "--target=riscv64", "--config=x.cfg", "-cc1", "-flto",
                     "-flto=thin", "-emit-llvm", "-Wa,-march=rv64gc", "-Wa,--noexecstack,-mattr=+f",
                     "--driver-mode=cl", "--driver-mode=gcc"):
            self.refused(VIRTUAL, "-c", "a.c", word)
        self.refused(VIRTUAL, "-c", "a.c", "-target", "riscv64")
        self.refused(VIRTUAL, "-c", "a.c", "--config", "x.cfg")
        self.refused(VIRTUAL, "-c", "a.S", "-Xassembler", "-mattr=+f")
        self.refused(VIRTUAL, "-c", "a.S", "-Xclangas", "-target-feature")
        for abi in (["-fpack-struct=1"], ["-fshort-enums"], ["-fshort-wchar"], ["-Xclang", "-fpack-struct=1"],
                    ["-Xclang=-fshort-enums"], ["-mms-bitfields"]):
            self.refused(VIRTUAL, "-c", "a.c", *abi)
        for cc1 in (["-Xclang", "-fno-bitfield-type-align"], ["-Xclang=-fno-bitfield-type-align"],
                    ["-Xclang", "-fmax-type-align=1"]):
            self.refused(VIRTUAL, "-c", "a.c", *cc1)
        check_arguments(["-c", "a.c", "-Xclang", "-fexperimental-assignment-tracking=disabled"], VIRTUAL)
        for width in ("-m32", "-m64", "-mbig-endian", "-mstrict-align"):
            self.refused(VIRTUAL, "-c", "a.c", width)
        check_arguments(["-c", "a.c", "-mno-relax", "-mcmodel=medany"], VIRTUAL)
        for nested in (["-Xclang=@x"], ["-Xclang", "@x"], ["-Wa,@x"], ["-mllvm", "@x"], ["-mllvm=@x"]):
            self.refused(VIRTUAL, "-c", "a.c", *nested)

    def test_gp_options_follow_the_profile(self):
        for spelling in ("-capstone-gp-free=false", "-capstone-gp-free=0", "--capstone-image-gp=false"):
            self.refused(VIRTUAL, "-c", "a.c", "-mllvm", spelling)
        for spelling in ("-capstone-gp-free", "-capstone-gp-free=1", "--capstone-image-gp=true",
                         "-capstone-gp-free=True"):
            self.refused(PHYSICAL, "-c", "a.c", "-mllvm", spelling)
            self.refused(PHYSICAL, "-c", "a.c", "-Xclang", "-mllvm", "-Xclang", spelling)
        self.refused(VIRTUAL, "-c", "a.c", "-mllvm", "-capstone-gp-free=maybe")
        for config in (VIRTUAL, PHYSICAL):
            self.refused(config, "-c", "a.c", "-mllvm", "-capstone-gp-captable")
        check_arguments(["-c", "a.c", "-mllvm", "-capstone-shrink-stack=false"], VIRTUAL)


class CompileOnlyTests(unittest.TestCase):
    def test_mode_words_in_value_positions_do_not_count(self):
        from capstone_vm.compiler import compile_only
        self.assertTrue(compile_only(["-O2", "-c", "a.c", "-o", "a.o"]))
        self.assertTrue(compile_only(["-o", "a.o", "-c", "a.c"]))
        self.assertFalse(compile_only(["main.o", "-x", "-c", "x.c"]))
        self.assertFalse(compile_only(["main.o", "-MF", "-c", "-o", "app.dom"]))
        for bad in (["main.o", "-o", "-c", "-Wl,-T,x.ld"], ["a.c", "-o", "@x", "-c"]):
            with self.assertRaises(ValueError):
                compile_only(bad)


class PlannedJobTests(unittest.TestCase):
    """clang's -### plan, whatever route its arguments took (-Wp, -Xpreprocessor ...)."""
    def job(self, *extra, kind="-cc1", features=("+m", "+a", "+f", "+d"),
            gp=("-capstone-gp-free", "-capstone-image-gp")):
        words = ["clang", kind, "-triple", "capstone64-unknown-unknown-elf", "-target-abi", "lp64"]
        for f in features:
            words += ["-target-feature", f]
        for g in gp:
            words += ["-mllvm", g]
        return words + list(extra)

    def test_the_profile_plan_passes(self):
        from capstone_vm.compiler import check_cc1
        check_cc1(self.job(), VIRTUAL)
        check_cc1(self.job("-target-feature", "+movc-keeps-integer-source"), VIRTUAL)
        # The SDK's +movc-keeps-integer-source, then the caller's -: the last wins.
        check_cc1(self.job("-target-feature", "+movc-keeps-integer-source",
                           "-target-feature", "-movc-keeps-integer-source"), VIRTUAL)
        check_cc1(self.job(), VIRTUAL)
        check_cc1(self.job("-target-feature", "-movc-keeps-integer-source", features=("+m", "+a"), gp=()),
                  PHYSICAL)
        physical_keep = self.job("-target-feature", "+movc-keeps-integer-source", features=("+m", "+a"), gp=())
        with patch.dict(os.environ):
            os.environ.pop("CAPSTONE_ALLOW_UNSAFE_MOVC_PROBE", None)
            with self.assertRaises(ValueError):
                check_cc1(physical_keep, PHYSICAL)
            os.environ["CAPSTONE_ALLOW_UNSAFE_MOVC_PROBE"] = "1"
            check_cc1(physical_keep, PHYSICAL)
        check_cc1(self.job(kind="-cc1as", features=()), VIRTUAL)
        check_cc1(self.job(features=("+m", "+a"), gp=()), PHYSICAL)

    def test_other_plans_are_refused(self):
        from capstone_vm.compiler import check_cc1
        for job, config in ((self.job("-target-feature", "+e"), VIRTUAL),
                            (self.job(features=("+m", "+a", "+f")), VIRTUAL),
                            (self.job("-fpack-struct=1"), VIRTUAL),
                            (self.job("-fno-bitfield-type-align"), VIRTUAL),
                            (self.job("-mllvm", "-capstone-gp-captable"), VIRTUAL),
                            (self.job(gp=("-capstone-gp-free",)), VIRTUAL),
                            (self.job(features=("+m", "+a"), gp=("-capstone-gp-free",)), PHYSICAL),
                            (self.job(features=("+m", "+a", "+f", "+d"), gp=()), PHYSICAL),
                            (self.job(kind="-cc1as", features=("+e",)), VIRTUAL),
                            (self.job(features=("+m", "+a"), gp=("-capstone-gp-free=True",)), PHYSICAL),
                            (self.job(gp=("-capstone-gp-free=False", "-capstone-image-gp")), VIRTUAL),
                            (self.job(gp=("-capstone-gp-free=yes", "-capstone-image-gp")), VIRTUAL),
                            (self.job("-target-cpu", "sifive-u74", features=("+m", "+a"), gp=()), PHYSICAL)):
            with self.assertRaises(ValueError, msg=str(job)):
                check_cc1(job, config)
        triple = self.job(); triple[3] = "x86_64-unknown-linux-gnu"
        with self.assertRaises(ValueError):
            check_cc1(triple, VIRTUAL)


class LinkInputTests(unittest.TestCase):
    HARD, SOFT = "rv64i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0", "rv64i2p1_m2p0_a2p1"

    def check(self, blob: bytes, config: dict):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            path.write_bytes(blob)
            check_link_inputs([path], config)

    def refused(self, blob: bytes, config: dict):
        with self.assertRaises(ValueError):
            self.check(blob, config)

    def test_objects_must_match_the_profile(self):
        self.check(elf(self.HARD), VIRTUAL)
        self.check(elf(self.SOFT), PHYSICAL)
        self.check(elf(None), VIRTUAL)            # assembly: no attributes
        self.check(elf(None), PHYSICAL)
        self.check(elf(self.HARD, ("__addtf3",)), VIRTUAL)
        self.refused(elf(self.SOFT), VIRTUAL)
        self.refused(elf(self.HARD), PHYSICAL)
        self.refused(elf(self.HARD + "_zfh1p0"), VIRTUAL)
        self.refused(elf(self.HARD + "_v1p0"), VIRTUAL)
        self.refused(elf(None, ("__muldf3",)), VIRTUAL)
        self.refused(elf(self.HARD, ("__adddf3",)), VIRTUAL)
        self.check(elf(None, ("__muldf3",)), PHYSICAL)
        self.refused(elf(self.HARD, flags=0x4), VIRTUAL)  # lp64d
        self.refused(elf(self.HARD, flags=0x8), VIRTUAL)  # RVE
        self.refused(elf("rv64e2p0_m2p0"), PHYSICAL)
        self.refused(elf(None, machine=243), VIRTUAL)     # RISC-V
        self.refused(b"BC\xc0\xde" + bytes(60), VIRTUAL)
        self.refused(elf(self.HARD)[:-40], VIRTUAL)       # truncated headers
        noheaders = bytearray(elf(self.HARD)); struct.pack_into("<HH", noheaders, 60, 0, 0)
        self.refused(bytes(noheaders), VIRTUAL)           # no sections at all
        deplibs = bytearray(elf(None)); shoff = struct.unpack_from("<Q", deplibs, 40)[0]
        struct.pack_into("<I", deplibs, shoff + 64 + 4, 0x6FFF4C04)  # shstrtab retyped as .deplibs
        self.refused(bytes(deplibs), PHYSICAL)
        badlink = bytearray(elf(None, ("__muldf3",)))
        shnum = struct.unpack_from("<H", badlink, 60)[0]; shoff = struct.unpack_from("<Q", badlink, 40)[0]
        struct.pack_into("<I", badlink, shoff + (shnum - 2) * 64 + 40, 99)  # symtab sh_link out of range
        self.refused(bytes(badlink), VIRTUAL)
        blob = elf(self.HARD)
        at = blob.index(attributes(self.HARD))
        self.refused(blob[:at] + b"B" + blob[at + 1:], VIRTUAL)            # not 'A'
        self.refused(blob[:at + 1] + b"\xff\xff" + blob[at + 3:], VIRTUAL)  # length past the end

    def test_the_builtins_archive_may_call_its_own_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            builtins = Path(directory) / "libbuiltins.a"
            builtins.write_bytes(archive({"subdf3.o": elf(self.HARD, ("__adddf3",))}))
            other = Path(directory) / "libother.a"
            other.write_bytes(builtins.read_bytes())
            config = {**VIRTUAL, "builtins": str(builtins)}
            check_link_inputs([builtins], config)
            with self.assertRaises(ValueError):
                check_link_inputs([other], config)
            builtins.write_bytes(archive({"subdf3.o": elf(self.SOFT, ("__adddf3",))}))
            with self.assertRaises(ValueError):
                check_link_inputs([builtins], config)

    def test_an_archive_may_satisfy_its_own_members_helpers(self):
        # A port that merges compiler-rt's builtins into its libc archive (cpython's
        # libc-and-builtins.a) carries compiler-rt's own internal calls: __fixdfdi calls
        # __fixunsdfdi. A member's reference to a soft-float helper is satisfied when the
        # SAME archive defines it AND the member attests hard float itself.
        fixdfdi = elf(self.HARD, ("__fixunsdfdi",))
        fixunsdfdi = elf(self.HARD, defined=("__fixunsdfdi",))
        self.check(archive({"fixdfdi.o": fixdfdi, "fixunsdfdi.o": fixunsdfdi}), VIRTUAL)
        # Nothing in the archive defines it: still refused (the board lane's negative test).
        self.refused(archive({"fixdfdi.o": fixdfdi}), VIRTUAL)
        # Defined beside it, but the caller does not attest hard float: still refused.
        self.refused(archive({"x.o": elf(None, ("__fixunsdfdi",)), "fixunsdfdi.o": fixunsdfdi}), VIRTUAL)
        self.refused(archive({"x.o": elf(self.SOFT, ("__fixunsdfdi",)), "fixunsdfdi.o": fixunsdfdi}), VIRTUAL)
        # The archive defines a DIFFERENT helper: still refused.
        self.refused(archive({"x.o": elf(self.HARD, ("__adddf3",)), "fixunsdfdi.o": fixunsdfdi}), VIRTUAL)
        # A definition in another input does not satisfy: each archive answers for itself.
        self.refused(fixdfdi, VIRTUAL)

    def test_archives_check_every_member(self):
        self.check(archive({"a.o": elf(self.HARD), "a-long-member-name.o": elf(None)}), VIRTUAL)
        self.refused(archive({"a.o": elf(self.HARD), "a-long-member-name.o": elf(self.SOFT)}), VIRTUAL)
        self.refused(archive({"a.o": elf(self.HARD)}, thin=True), VIRTUAL)
        # Only the exact symbol-table names are skipped.
        self.refused(archive({"__.SYMDEFbad": elf(self.SOFT)}), VIRTUAL)

    def test_libraries_resolve_as_the_linker_does(self):
        with tempfile.TemporaryDirectory() as directory:
            one, two = Path(directory, "one"), Path(directory, "two")
            one.mkdir(); two.mkdir()
            (two / "libx.a").write_bytes(b"")
            self.assertEqual(link_files(["-L", str(one), f"-L{two}", "-lx"]), ([two / "libx.a"], [str((two / "libx.a").resolve())]))
            (one / "libx.so").write_bytes(b"")
            with self.assertRaises(ValueError):
                link_files(["-L", str(one), f"-L{two}", "-lx"])
            self.assertEqual(link_files([f"-L{two}", "-l:libx.a", "obj.o"]),
                             ([Path("obj.o"), two / "libx.a"], [str((two / "libx.a").resolve()), "obj.o"]))
            # lld reads -library=x as --library=x; here it is library "ibrary=x", resolved and
            # handed over as a path, so lld never sees the spelling.
            (two / "libibrary=x.a").write_bytes(b"")
            self.assertEqual(link_files([f"-L{two}", "-library=x"])[1], [str((two / "libibrary=x.a").resolve())])
            # A library named like an option still reaches lld as an absolute path.
            cwd = os.getcwd(); os.chdir(two)
            try:
                (two / "-Tpayload").write_bytes(b"")
                self.assertTrue(link_files(["-L.", "-l:-Tpayload"])[1][0].startswith("/"))
            finally:
                os.chdir(cwd)
            for bad in (["-L=/sysroot", "-lx"], ["--wrap", "@args"], ["-u", "@x"], ["-l"]):
                with self.assertRaises(ValueError, msg=str(bad)):
                    link_files(bad)
            with self.assertRaises(ValueError):
                link_files([f"-L{two}", "-lmissing"])
            for item in ("@args", "-T", "--script=x.ld", "-Tx.ld", "-script=x.ld", "--library=x",
                         "--library-path=x", "--just-symbols=x", "-R", "--dependent-libraries",
                         "-dT", "--version-script=x", "-o"):
                with self.assertRaises(ValueError, msg=item):
                    link_files([item, "x"])
            self.assertEqual(link_files(["-Map", "map.txt", "-e", "start", "--defsym", "x=1", "-u", "sym",
                                         "--wrap=main", "--undefined=f", "-z", "noexecstack",
                                         "--gc-sections", "--error-limit=0", "-rpath", "/usr/lib", "obj.o"])[0], [Path("obj.o")])
            with self.assertRaises(ValueError):
                link_files(["-Map"])
