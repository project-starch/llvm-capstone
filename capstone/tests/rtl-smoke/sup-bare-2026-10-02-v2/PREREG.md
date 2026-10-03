# Supervised CALL on silicon, v2 tests -- pre-registered 2026-10-02 21:06:33, committed on the lane branch before the boot

The tests are capstone-ariane 7564c0945, repaired after the v1 audit; this is a test change only, no RTL change since
the synthesized 36a641e0b. Bitstream: caplifive_supcall_36a641e0b.bit (identified by csnodefree 0xFFCD). The harness is
the v1 harness with the recorder moved to x16/x17/x8, because the repaired sup-escape keeps its seal aliases in
s8/s9/x31. One board session per image.
Images (sha256[:16]): q64 2e4747f346eaad85, q16 57f1d4cd9e959232, sup-escape b45cd0aab8fe97ad.

PREDICTION: each board vector EQUALS the RTL lane's simulation vector of the same test (CAPPRINT readings in order,
expected-vectors.json):
- sup-quantum q64: 153 readings, from sim run aud3b (archive sup-run5-202556). Tail: 1, 0x1f64f7f55a20a182, 0x1234,
  0x49 (73 resumes), 0xA5, 0x888, 0.
- sup-quantum q16: 155 readings, also from aud3b. Tail: 1, 0x1f64f7f55a20a182, 0x1234, 0x4a (74), 0xA5, 0x888, 0.
  - The ladder value is acc := acc*7 + i over 100 steps. A skip and a repeat at different positions cannot cancel, so
    0x1f64f7f55a20a182 is exactly-once resume.
- sup-escape: 72 readings, from sim run aud3e. The address-dependent readings are relocated:
  - sim 0x80003401 (scratch + 1, the reference trap's mtval and arm 3's tval) becomes this image's 0x80005401.
  - The saved-mcause slot reads 0 in all five arms, meaning the trap was STRIPPED.
  - The alternative is the no-strip mutant's vector (expected-vectors.json, "sup-escape-if-NOT-stripped"). It
    differs ONLY at the five saved-mcause readings, which there read 11, 2, 2, 4, 2. So the board reading settles the
    strip on silicon.
- REFUTED if any reading differs. A difference confined to the five saved-mcause readings, with the mutant's values,
  means "trap not stripped on silicon".
