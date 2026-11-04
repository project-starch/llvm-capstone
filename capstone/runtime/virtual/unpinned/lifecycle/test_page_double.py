#!/usr/bin/env python3
"""Contract checks and two deliberately broken lifecycle implementations."""
import random
import unittest
from page_double import GRANULE, PAGE, Lifecycle, Refused


def movement(model):
    rng = random.Random(0)
    source = model.allocate(0)
    clone = model.allocate(PAGE)
    for i in range(PAGE // GRANULE):
        model.write(source, i * GRANULE, rng.randbytes(GRANULE),
                    {'id': i + 1, 'live': bool(i % 2), 'bounds': [3, 139]})
    expected = dict(model.page(source).tags)
    for _ in range(12):
        model.copy(clone, source)
        if model.page(clone).tags != expected:
            return 'lost tag'
        if model.page(clone).data != model.page(source).data:
            return 'lost bytes'
        source, clone = clone, source
    return 'ok'


class Contract(unittest.TestCase):
    def test_moves_and_scalar_copy_mutant(self):
        self.assertEqual(movement(Lifecycle()), 'ok')
        self.assertEqual(movement(Lifecycle(scalar_copy=True)), 'lost tag')

    def test_reuse_and_skip_clear_mutant(self):
        for broken in [False, True]:
            model = Lifecycle(skip_clear=broken)
            handle = model.allocate(0)
            for off in [0, 16, 2032, 4080]:
                model.write(handle, off, bytes([42]) * GRANULE, {'live': False})
            if broken:
                with self.assertRaisesRegex(Refused, 'tag survived clear'):
                    model.release(handle)
                self.assertTrue(model.page(handle).tags)
            else:
                model.release(handle)
                new = model.allocate(0)
                self.assertEqual(model.query(new), 0)
                self.assertEqual(model.page(new).data, bytes(PAGE))
                with self.assertRaisesRegex(Refused, 'expired handle'):
                    model.clear(handle)

    def test_all_overlapping_granules_and_sticky_ack(self):
        model = Lifecycle()
        page = model.allocate(0)
        for off in range(0, 64, GRANULE):
            model.write(page, off, bytes(GRANULE), {'id': 1})
        model.write(page, 15, bytes(34))
        self.assertFalse(model.page(page).tags)
        self.assertEqual(model.query(page), 1)
        self.assertEqual(model.query(page, ack=True), 0)
        model.write(page, 4080, bytes(GRANULE), {'id': 2})
        self.assertEqual(model.query(page, ack=True), 1)
        model.clear(page)
        self.assertEqual(model.query(page), 1)
        self.assertEqual(model.query(page, ack=True), 0)

    def test_dma_and_failed_operations_do_not_publish(self):
        model = Lifecycle()
        page = model.allocate(0)
        dst = model.allocate(PAGE)
        model.write(page, 0, bytes(GRANULE), {'id': 1})
        model.dma(page, True)
        self.assertFalse(model.page(page).tags)
        for action in [lambda: model.copy(dst, page), lambda: model.clear(page),
                       lambda: model.release(page),
                       lambda: model.write(page, 0, bytes(GRANULE), {'id': 1})]:
            with self.assertRaises(Refused):
                action()
        model.write(page, 0, bytes([1]) * GRANULE)
        model.dma(page, False)
        self.assertEqual(model.query(page, ack=True), 0)
        with self.assertRaises(Refused):
            model.copy(page, page)
        for req in [{'version': 2}, {'version': 1, 'op': 'fork_namespace'}]:
            with self.assertRaises(Refused):
                model.dispatch(req)


if __name__ == '__main__':
    unittest.main()
