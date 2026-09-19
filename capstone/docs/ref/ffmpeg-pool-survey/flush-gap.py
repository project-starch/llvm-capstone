#!/usr/bin/env python3
"""Reference-holding context members released on close but not on flush.

The shape comes from upstream `a024f8c541`: `vp9_decode_flush` releases
`s->s.frames`, `s->s.refs` and `s->s.ref_frames` but leaves `s->next_refs`
referenced, while `vp9_decode_free` releases all four. A flush that clears only
part of a decoder's reference state leaves the rest reachable afterwards.

CALIBRATION. Run against a tree and check that `libavcodec/vp9.c` is reported
with `s->next_refs[]`. That defect is present at the 9.0.1 pin, so a run that
does not name it is measuring something else and its other hits mean nothing.

TWO EARLIER VERSIONS WERE WRONG, in ways worth not repeating:

1. Resolving helper calls only inside the defining file reported
   `libavcodec/h264dec.c` (`pic->f`, `pic->f_grain`) and
   `libavcodec/hevc/hevcdec.c`. Both are false: `h264_decode_flush` does release
   the DPB, through `ff_h264_unref_picture` in `h264_picture.c`, and
   `hevc_decode_flush` through `ff_hevc_flush_dpb` in `hevc/refs.c`.

2. Fixing that with a flat tree-wide name->body map was worse. `decode_close`,
   `decode_flush` and `flush` are `static` names defined in dozens of files, the
   first definition won, and the tool reported an `s->prev_frame` in `ralf.c`
   that does not exist in that file at all. Hence: resolve locally first, and use
   the tree-wide table only for names that are UNIQUE across it.

A gap is a candidate, not a defect. Retained references keep memory alive, so a
member missing from flush is memory-unsafe only where something else ends the
object's lifetime while the pointer survives, or where the stale reference is
reachable again afterwards. Each hit needs that second question answered at the
source before it counts.

Usage: flush-gap.py <ffmpeg-source-root>
"""
import re, sys, pathlib, collections

ROOT = pathlib.Path(sys.argv[1])
FRAME_REL = re.compile(
    r'\b(av_frame_unref|av_frame_free|ff_progress_frame_unref|'
    r'ff_thread_release_ext_buffer|av_refstruct_unref|ff_refstruct_unref)\s*\(')
MEMBER = re.compile(r'^\s*&?\s*([a-zA-Z_][a-zA-Z0-9_]*(?:(?:->|\.)[a-zA-Z0-9_]+(?:\[[^\]]*\])?)+)\s*$')
FNDEF = re.compile(r'^[a-zA-Z_][^\n;=]*?\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^;{]*\)\s*\{', re.M)

def bodies(text):
    out = {}
    for m in FNDEF.finditer(text):
        i = text.index('{', m.start()); d = 0
        for j in range(i, len(text)):
            if text[j] == '{': d += 1
            elif text[j] == '}':
                d -= 1
                if d == 0:
                    out.setdefault(m.group(1), text[i:j]); break
    return out

# A flat name->body map is wrong: decode_close/decode_flush/flush are defined
# in dozens of files, and the first one wins. Keep only names that are UNIQUE
# across the tree as the cross-file fallback; everything else resolves locally.
_seen = collections.Counter()
_body = {}
for p in ROOT.glob('libavcodec/**/*.c'):
    for k, v in bodies(p.read_text(errors='replace')).items():
        _seen[k] += 1
        _body.setdefault(k, v)
GLOBAL = {k: v for k, v in _body.items() if _seen[k] == 1}

def released(name, local, seen=None, depth=0):
    if seen is None: seen = set()
    table = local if name in local else GLOBAL
    if name in seen or depth > 3 or name not in table: return set()
    seen.add(name)
    body, got = table[name], set()
    for m in FRAME_REL.finditer(body):
        k = body.index('(', m.start()); d = 0; arg = ''
        for j in range(k, len(body)):
            if body[j] == '(': d += 1
            elif body[j] == ')':
                d -= 1
                if d == 0: arg = body[k+1:j]; break
        mm = MEMBER.search(arg)
        if mm: got.add(re.sub(r'\[[^\]]*\]', '[]', mm.group(1).strip()))
    for call in set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', body)):
        if call != name and any(w in call for w in ('unref', 'free', 'flush', 'reset', 'release')):
            got |= released(call, local, seen, depth + 1)
    return got

n = 0
for path in sorted(ROOT.glob('libavcodec/**/*.c')):
    text = path.read_text(errors='replace')
    fl = re.search(r'\.flush\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)', text)
    cl = re.search(r'\.close\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)', text)
    if not (fl and cl): continue
    local = bodies(text)
    f, c = released(fl.group(1), local), released(cl.group(1), local)
    gap = {g for g in c - f if not g.startswith(('avctx->', 'frame->', 'f->', 'pic->', 'src->', 'dst->'))}
    if gap:
        n += 1
        print(f'{path.relative_to(ROOT)}  flush={fl.group(1)} close={cl.group(1)}')
        for x in sorted(gap): print(f'    nur im close: {x}')
print(f'\n{n} Dateien mit Luecke (baumweit aufgeloest)')
