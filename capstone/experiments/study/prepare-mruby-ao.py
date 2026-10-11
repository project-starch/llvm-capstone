#!/usr/bin/env python3
"""Add phase boundaries to pinned upstream mruby ambient-occlusion render."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = 'e44efffd6415b9ef875bd92272ad2bd984ab4bc0f078b6e73aa791fd9865180f'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('expected pinned mruby 4.0.0-rc2 bm_ao_render.rb')
    code = raw.decode()
    before = '  printf("P6\\n")\n'
    after = '  Scene.new.render(IMAGE_WIDTH, IMAGE_HEIGHT, NSUBSAMPLES)\n'
    if code.count(before) != 1 or code.count(after) != 1:
        raise ValueError('expected AO output/render sites')
    code = code.replace(before, '  STDERR.syswrite("MEMPHASE before\\n")\n' + before)
    code = code.replace(after, after + '  STDERR.syswrite("MEMPHASE after\\n")\n')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'ao.rb').write_text(code)
    (args.out/'workload.json').write_text(json.dumps(dict(
        suite='mruby-upstream', case='bm_ao_render.rb', source_sha256=SOURCE_SHA256,
        generated_sha256=hashlib.sha256(code.encode()).hexdigest(),
        default_width=64, tested_widths=[8,16],
        adaptation='Two stderr phase markers only; render body, constants and binary PPM output unchanged.'),
        indent=2)+'\n')


if __name__ == '__main__':
    main()
