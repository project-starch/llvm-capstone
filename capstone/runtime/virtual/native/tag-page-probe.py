#!/usr/bin/env python3
"""Consume the page lane's double; this does not exercise physical hardware."""
import importlib.util
from pathlib import Path
import sys


def refused(peer, operation, reason):
    try:
        operation()
    except peer.Refused as error:
        assert str(error) == reason, error
    else:
        raise AssertionError("operation accepted")


def exercise(peer, mutant):
    model = peer.Lifecycle(skip_clear=mutant == "skip-clear",
                           scalar_copy=mutant == "scalar-copy")
    source, target = model.allocate(0), model.allocate(peer.PAGE)
    live = {"namespace": 9, "id": 17, "epoch": 3, "bounds": [3, 139]}
    stale = dict(live, epoch=2)
    model.write(source, 0, bytes([42]) * peer.GRANULE, live)
    model.write(source, peer.GRANULE, bytes([43]) * peer.GRANULE, stale)
    model.copy(target, source)
    assert model.page(target).tags == {0: live, 1: stale}, "copy lost identity"
    assert model.page(target).data == model.page(source).data

    # Page movement transports exact records; the grant layer alone decides
    # liveness. Retiring a grant never turns stale transported cargo live.
    grant_epoch = 3
    assert [t["epoch"] == grant_epoch for t in model.page(target).tags.values()] == [True, False]
    grant_epoch += 1
    assert not any(t["epoch"] == grant_epoch for t in model.page(target).tags.values())

    # A scalar write touching two granules must invalidate both records.
    model.write(target, peer.GRANULE - 1, bytes(2))
    assert not model.page(target).tags
    model.write(target, 0, bytes(peer.GRANULE), live)
    model.dma(target, True)
    assert not model.page(target).tags
    for operation in [lambda: model.clear(target), lambda: model.release(target),
                      lambda: model.copy(source, target)]:
        refused(peer, operation, "writable DMA mapping")
    model.dma(target, False)

    # Clearing on release must remove even already-stale capability records.
    try:
        model.release(source)
    except peer.Refused as error:
        assert str(error) == "tag survived clear", error
        raise AssertionError("release retained tags") from error
    replacement = model.allocate(0)
    assert replacement != source
    assert model.query(replacement) == 0 and not model.page(replacement).tags
    assert model.page(replacement).data == bytes(peer.PAGE)
    refused(peer, lambda: model.write(source, 0, bytes(1)), "expired handle")
    model.release(target)


def main():
    path = Path(sys.argv[1]) / "capstone/runtime/virtual/unpinned/lifecycle/page_double.py"
    spec = importlib.util.spec_from_file_location("page_peer", path)
    peer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = peer
    spec.loader.exec_module(peer)
    exercise(peer, "none")
    print("tag-page-peer normal PASS")
    for mutant, expected in [("skip-clear", "release retained tags"),
                             ("scalar-copy", "copy lost identity")]:
        try:
            exercise(peer, mutant)
        except AssertionError as error:
            assert str(error) == expected, error
            print(f"tag-page-peer {mutant} detected: {error}")
        else:
            raise AssertionError(f"undetected mutation: {mutant}")


if __name__ == "__main__":
    main()
