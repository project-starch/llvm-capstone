#!/usr/bin/env python3
"""Positive controls for transcript.py (ISSUES M-11).

Every test here asserts TWO things: that the old, line-by-line reading gets the synthetic log WRONG, and
that the module gets it right. A detector whose failure mode cannot be shown to exist proves nothing
(CLAUDE.md: "a CLEAN result is not evidence until the check is known to fire"), so if a test's
old-reading assertion ever stops holding, the failure mode has changed and the test must be revisited,
not relaxed.

Run:  python3 capstone/tests/rtl-smoke/fpga_driver/test_transcript.py      (exit 0 = all pass)
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fpga_driver import transcript as T  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.join(HERE, "transcript.py")

# A framed driver.log with every shape the archive showed (2026-09-15): the control's own line split
# across two chunks WITH an [event] line between them; a double-quoted frame (the payload holds a
# single quote); a monitor marker mid-token inside a cycle count; a SHA6 marker split across chunks;
# a mark split after four digits; the quoteless +0B frame; a previous boot's replay before load_image.
SYN = "\n".join([
    "[fpga] [uart] 'stale replay from the previous boot: RESULT k800 retval=9\\r\\n'",
    "[fpga] emit gdb_input <- {'cmd': 'monitor load_image /tmp/fw_payload.bin'}",
    "[fpga] [uart] 'OpenSBI v1.3\\r\\n'",
    "[fpga] [uart] '[    0.000000] Linux version 6.4.14\\r\\n'",
    "[stages] --> TEST 1/3  /test-domains/lpc|k800:/test-domains/k800.dom",
    "[fpga] [uart] \"echo '### TEST 1/3 START k800 ###'\\r\\n\"",
    "[fpga] [uart] 'SHA5:00000000\\r\\nSHA'",
    "[fpga] [event] led_state: {'states': [0, 0, 0, 0], 'server_epoch': 1}",
    "[fpga] [uart] '6:00000000\\r\\nladder-perf: RESULT k8'",
    "[fpga] [event] switch_state: {'states': [0, 1], 'server_epoch': 1}",
    "[fpga] [uart] '00 retval=4 cycles=4519 ran=53406 instret=1089 phase=2\\r\\n'",
    "[fpga] [uart] +0B",
    "[stages] <-- TEST 1/3  /test-domains/lpc|k800:/test-domains/k800.dom  returned in 1s",
    "[stages] --> TEST 2/3  /test-domains/speedtest1.dom:--speedtest1 --arena 2097152",
    "[fpga] [uart] 'SPEEDTEST1-CYCLES 1166ECSA:00000004\\r\\n'",
    "[fpga] [uart] '594074 HIGHWATER n/a HEAP 2097152 DROPPED 0 RC 0\\r\\n'",
    "[stages] <-- TEST 2/3  /test-domains/speedtest1.dom:--speedtest1 --arena 2097152  returned in 60s",
    "[stages] --> TEST 3/3  /test-domains/ngx-subpool.dom:--arena-linear",
    "[fpga] [uart] 'ngx retval = 1309'",
    "[fpga] [uart] '490692\\r\\n### TEST 3/3 END rc=0 ###\\r\\nDN_0\\r\\n'",
    "[stages] <-- TEST 3/3  /test-domains/ngx-subpool.dom:--arena-linear  returned in 5s",
    "",
])


def _clean():
    return T.strip_markers(T.uart_text(T.scope_to_run(SYN)))


def _old_seam_join(log):
    """e1-bundle's normaliser before this module: join ADJACENT frames only, unescape, delete markers."""
    log = re.sub(r"'\n\[fpga\] \[uart\] '", "", log)
    log = log.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "")
    return re.sub(r"[A-Z0-9]{4}:[0-9A-F]{8}\n?", "", log)


def test_control_split_with_event_between():
    scoped = T.scope_to_run(SYN)
    assert re.findall(r"RESULT k800 retval=[0-9-]+", scoped) == [], "old: the split control must NOT read line by line"
    assert re.findall(r"RESULT k800 retval=[0-9-]+", _old_seam_join(scoped)) == [], "old: an adjacency-only seam join must miss it too"
    assert T.find_all(r"RESULT k800 retval=[0-9-]+", _clean()) == ["RESULT k800 retval=4"]
    assert "retval=9" not in _clean(), "the previous boot's replay must be scoped out"


def test_double_quoted_frame():
    scoped = T.scope_to_run(SYN)
    single_only = "".join(m.group(1) for m in re.finditer(r"\[fpga\] \[uart\] '((?:[^'\\]|\\.)*)'", scoped))
    assert "### TEST 1/3 START" not in single_only, "old: a single-quote-only extractor drops the double-quoted frame"
    assert "### TEST 1/3 START k800 ###" in _clean()


def test_marker_mid_token():
    scoped = T.scope_to_run(SYN)
    old = re.search(r"SPEEDTEST1-CYCLES (\d+)", scoped)
    assert old and old.group(1) == "1166", "old: the marker cuts the number and a plausible wrong value reads"
    new = re.search(r"SPEEDTEST1-CYCLES (\d+) HIGHWATER", _clean())
    assert new and new.group(1) == "1166594074"


def test_sha6_split_across_chunks():
    scoped = T.scope_to_run(SYN)
    sed = subprocess.run(["sed", "-n", r"s/.*\(SHA[56]:[0-9A-F]*\).*/\1/p"], input=scoped, capture_output=True, text=True)
    assert sed.stdout.strip().splitlines()[-1] == "SHA5:00000000", "old: the per-line sed reads the stale marker (a false ENTRY-STALL)"
    assert T.last_marker(T.uart_text(scoped)) == "SHA6:00000000"
    cli = subprocess.run([sys.executable, MODULE, "last-marker"], input=SYN, capture_output=True, text=True)
    assert cli.returncode == 0 and cli.stdout.strip() == "SHA6:00000000"
    # a real stall: the marker is the last thing written and its newline never arrives -- the gate must still fire
    stall = SYN.rsplit("[stages] --> TEST 2/3", 1)[0] + "[fpga] [uart] 'SHA5:00000001'\n"
    assert T.last_marker(T.uart_text(T.scope_to_run(stall))) == "SHA5:00000001"


def test_mark_split_after_four_digits():
    arms = T.arm_segments(T.scope_to_run(SYN))
    a = arms[2]
    old = re.search(r"ngx retval = (\d+)", a.framed)
    assert old and (int(old.group(1)) & 0xFFFFFF) == 0x00051D, "old: the cut number reads as a plausible wrong mark (1309 -> mark 00051D)"
    new = re.search(r"ngx retval = (\d+)\n", a.uart)
    assert new and (int(new.group(1)) & 0xFFFFFF) == 0x0D3E04
    # with the second chunk missing (a log cut short), the terminator-anchored read must return NOTHING, never a mark
    cut = SYN.replace("[fpga] [uart] '490692\\r\\n### TEST 3/3 END rc=0 ###\\r\\nDN_0\\r\\n'\n", "")
    a2 = T.arm_segments(T.scope_to_run(cut))[2]
    assert re.search(r"ngx retval = (\d+)\n", a2.uart) is None


def test_arm_segments():
    arms = T.arm_segments(T.scope_to_run(SYN))
    assert [a.index for a in arms] == [1, 2, 3] and all(a.total == 3 for a in arms)
    assert arms[0].label.startswith("/test-domains/lpc|k800") and arms[0].returned is True
    assert "RESULT k800 retval=4" in arms[0].uart and "SHA5:" not in arms[0].uart
    wedged = SYN.replace("[stages] <-- TEST 3/3  /test-domains/ngx-subpool.dom:--arena-linear  returned in 5s",
                         "[wedge] sw=255 TRAP LOG {seen,mcause[6:0]} 0x99 10011001\n"
                         "[stages] <-- TEST 3/3  /test-domains/ngx-subpool.dom:--arena-linear  NO RETURN after 300s")
    w = T.arm_segments(T.scope_to_run(wedged))[2]
    assert w.returned is False
    assert re.search(r"TRAP LOG \{seen,mcause\[6:0\]\}\s+(0x[0-9a-f]+)", w.framed).group(1) == "0x99"


def test_require_and_frame_errors():
    try:
        T.require("", "UART after load_image")
        assert False, "require must raise on empty input"
    except T.TranscriptError:
        pass
    blocked = "\n".join(["=== preflight-board-run ===", "  BLOCK  first rung 'k800' ...", "preflight: BLOCKED",
                         "preflight BLOCKED -- not spending a boot.", ""])
    assert T.uart_text(T.scope_to_run(blocked)) == ""
    bad_middle = SYN.replace("[fpga] [uart] +0B", "[fpga] [uart] 'unterminated")
    try:
        T.uart_chunks(T.scope_to_run(bad_middle))
        assert False, "a malformed frame in the middle must raise"
    except T.TranscriptError as e:
        assert "unparseable UART frame" in str(e)
    live = SYN.rstrip("\n") + "\n[fpga] [uart] 'still being writ"
    assert T.uart_chunks(T.scope_to_run(live))[-1].startswith("490692"), "a malformed LAST frame is dropped, not fatal"


def test_archive_smoke():
    archive = os.path.expanduser("~/capstone-artifacts/unify")
    if not os.path.isdir(archive):
        print("  (archive absent; skipped)")
        return
    n = 0
    for d in sorted(os.listdir(archive)):
        p = os.path.join(archive, d, "driver.log")
        if d.startswith("board-") and os.path.exists(p):
            T.uart_chunks(T.scope_to_run(T.read(p)))
            n += 1
    assert n > 0
    print(f"  (archive: {n} transcripts, every frame parsed)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok    {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
