#!/usr/bin/env python3
"""Reject ambiguous nested image records while accepting staged re-exec."""
import unittest
from prof_images import group_image_records


def invocation(index):
    stage = f'/tmp/stage-{index:04d}-program'
    return f'{index:04d}', {'detail': {'staged': stage,
                                      'guest_argv': ['../' * 24 + stage[1:]]}}


def record(index, prefix='../../', entry=0x108, lo=0x100, hi=0x200):
    return entry, lo, hi, (f'entry={entry:#x} code={lo:#x}-{hi:#x} '
                           f'image={prefix}tmp/stage-{index:04d}-program')


class ImageRecords(unittest.TestCase):
    def test_nested_reexec_is_counted_in_its_parent_window(self):
        rows = [record(0), record(1), record(1, '../' * 25), record(1, '/')]
        groups = group_image_records(rows, [invocation(0), invocation(1)])
        self.assertEqual(groups, [rows[:1], rows[1:]])

    def test_ordinary_nonstaged_invocations_keep_their_placements(self):
        rows = [record(0), record(1, entry=0x308, lo=0x300, hi=0x400)]
        self.assertEqual(group_image_records(rows, [('0', {}), ('1', {})]),
                         [[rows[0]], [rows[1]]])

    def test_slice_record_can_precede_the_launcher_record(self):
        row = (0x108, 0x100, 0x200, 'capprof-slice: domain image entry=0x108 '
               'code=0x100-0x200 (derived from stopped_at; the launcher never exited)')
        self.assertEqual(group_image_records([row], [invocation(0)]), [[row]])
        with self.assertRaises(ValueError):
            group_image_records([row, row], [invocation(0)])

    def test_unknown_child_image_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'unrecognised'):
            group_image_records([record(0), record(1), record(9)],
                                [invocation(0), invocation(1)])

    def test_different_child_placement_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'differing'):
            group_image_records([record(0), record(0, entry=0x118)], [invocation(0)])

    def test_missing_or_reordered_window_is_refused(self):
        for rows in ([record(0)], [record(0)] * 2, [record(0)] * 3,
                     [record(0), record(1), record(0)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                group_image_records(rows, [invocation(0), invocation(1)])

    def test_stage_identity_is_required_and_unique(self):
        for inv in ([('0', {})], [invocation(0), invocation(0)]):
            with self.subTest(inv=inv), self.assertRaises(ValueError):
                group_image_records([record(0)] * 3, inv)

    def test_entry_must_be_inside_the_code_range(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            group_image_records([record(0, entry=0x300)] * 2, [invocation(0)])


if __name__ == '__main__':
    unittest.main()
