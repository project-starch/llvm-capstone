#!/usr/bin/env python3
"""Extract one case per candidate commit: the test lines the commit ADDED.

A candidate is a commit in the window that touches both a source file and a
Ruby test file -- the shape of "a fix and the test that proves it". The case is
the harness, the added test lines, and the footer, which is what survey.sh does
by hand for its twelve; this does it for every commit in the window.
"""
import subprocess, sys, os, re, json

W = '/tmp/capstone/mruby-corpus'
SRC = f'{W}/full'
PIN = '9d523e2f74f2e63ca02840937523de61398a617d'
TO  = subprocess.run(['git','-C',SRC,'rev-parse','origin/master'],capture_output=True,text=True).stdout.strip()

def git(*a):
    return subprocess.run(['git', '-C', SRC, *a], capture_output=True, text=True).stdout

def is_src(f):
    return (f.startswith('src/') or f.startswith('include/')
            or (f.startswith('mrbgems/') and ('/src/' in f or f.endswith('.c') or f.endswith('.h'))))

def is_test(f):
    return f.endswith('.rb') and ('/test/' in f or f.startswith('test/'))

out = git('log', '--format=%H', '--name-only', f'{PIN}..{TO}')
commits, cur, files = [], None, []
for line in out.splitlines():
    if len(line) == 40 and re.fullmatch(r'[0-9a-f]{40}', line):
        if cur: commits.append((cur, files))
        cur, files = line, []
    elif line.strip():
        files.append(line.strip())
if cur: commits.append((cur, files))

harness = open(f'{W}/probe/harness.rb').read()
footer  = open(f'{W}/probe/footer.rb').read()
os.makedirs(f'{W}/cases', exist_ok=True)

index = []
for sha, files in commits:
    tests = [f for f in files if is_test(f)]
    if not tests or not any(is_src(f) for f in files):
        continue
    added = []
    for t in tests:
        diff = git('show', '--format=', '-U0', sha, '--', t)
        for line in diff.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                added.append(line[1:])
    if not added:
        continue
    body = '\n'.join(added)
    # A case has to contain at least one assertion to be able to fail at all.
    if 'assert' not in body:
        continue
    name = sha[:9]
    with open(f'{W}/cases/{name}.rb', 'w') as fh:
        fh.write(harness + body + '\n' + footer)
    subject = git('log', '-1', '--format=%s', sha).strip()
    index.append({'sha': sha, 'name': name, 'subject': subject,
                  'test_files': tests, 'added_lines': len(added),
                  'src_files': [f for f in files if is_src(f)]})

json.dump(index, open(f'{W}/cases/index.json', 'w'), indent=1)
print(f'candidate commits: {len(index)}')
