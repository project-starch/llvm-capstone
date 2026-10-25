"""The allocator gate must ignore data and distinguish protected from stock code."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'lifetime_instructions', Path(__file__).parents[1] / 'application/lifetime-instructions.py')
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)


def word(funct7):
    return struct.pack('<I', (funct7 << 25) | (1 << 12) | 0x5b)


def elf(code, data=b'', machine=259):
    header = bytearray(64)
    header[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<H', header, 18, machine)
    struct.pack_into('<Q', header, 40, 64)
    struct.pack_into('<HH', header, 58, 64, 3)
    sections = bytearray(3 * 64)
    struct.pack_into('<QQQQ', sections, 64 + 8, 4, 0, 256, len(code))
    struct.pack_into('<QQQQ', sections, 128 + 8, 0, 0, 256 + len(code), len(data))
    return header + sections + code + data


class LifetimeInstructions(unittest.TestCase):
    def scan(self, content):
        with tempfile.TemporaryDirectory() as directory:
            obj = Path(directory) / 'allocator.o'
            obj.write_bytes(content)
            return scanner.counts(obj)

    def test_compressed_prefix_and_register_operands(self):
        derive = struct.pack('<I', int.from_bytes(word(0x51), 'little') | (7 << 7) | (8 << 15) | (9 << 20))
        self.assertEqual(self.scan(elf(b'\x01\x00' + derive + word(0x52))), (1, 1))

    def test_data_and_old_opcodes_do_not_protect_allocator(self):
        self.assertEqual(self.scan(elf(word(0x50), word(0x51) + word(0x52))), (0, 0))

    def test_missing_revoke_is_distinguishable(self):
        self.assertEqual(self.scan(elf(word(0x51))), (1, 0))

    def test_wrong_machine_and_truncation_are_rejected(self):
        for content in (elf(word(0x51), machine=243), elf(word(0x51))[:-2], elf(b'\x03\x00')):
            with self.subTest(content=content[-8:]), self.assertRaises(ValueError):
                self.scan(content)


if __name__ == '__main__':
    unittest.main()
