"""Associate repeated launches of a staged image with their profiling window."""
import re


def staged_path_matches(path, staged):
    if path == staged:
        return True
    if not path.startswith('../'):
        return False
    while path.startswith('../'):
        path = path[3:]
    return path == staged.lstrip('/')


def group_image_records(records, invocations):
    """Return one group per invocation; refuse ambiguous nested placements.

    A nested exec can reuse its parent's staged image. The stage path binds it
    to a window, and identical entry/code ranges let the existing byte-checked
    attribution count every instruction once, including all nested processes.
    Different nested images or placements need richer recording, not guessing.
    """
    # stop-at.py can derive a slice's sole image from the byte-checked stop
    # address before the launcher has exited and printed an image path.
    if len(records) == len(invocations) and any(' image=' not in r[3] for r in records):
        return [[record] for record in records]
    if len(records) < len(invocations):
        raise ValueError(f'{len(records)} image records for {len(invocations)} guest invocations')
    stages = []
    for inv_id, inv in invocations:
        detail = inv.get('detail', {})
        stage = detail.get('staged')
        argv = detail.get('guest_argv', [])
        if (not isinstance(stage, str) or not stage.startswith('/tmp/stage-')
                or not argv or not staged_path_matches(argv[0], stage)):
            if len(records) == len(invocations):
                return [[record] for record in records]
            raise ValueError(f'nested images: invocation {inv_id} has no matching staged image')
        stages.append(stage)
    if len(set(stages)) != len(stages):
        raise ValueError('nested images: stage paths do not uniquely identify invocations')
    groups = [[] for _ in invocations]
    previous = -1
    for record in records:
        match = re.search(r' image=(.*)$', record[3])
        candidates = [i for i, stage in enumerate(stages)
                      if match and staged_path_matches(match[1], stage)]
        if len(candidates) != 1:
            raise ValueError('nested images: unrecognised or ambiguous staged image: ' + record[3])
        index = candidates[0]
        if index < previous:
            raise ValueError('nested images: invocation records are out of order')
        previous = index
        group = groups[index]
        if group and record[:3] != group[0][:3]:
            raise ValueError('nested images: differing entry/code placement within an invocation')
        if not record[1] <= record[0] < record[2]:
            raise ValueError('nested images: entry is outside the recorded code range')
        group.append(record)
    if not all(groups):
        raise ValueError('nested images: an invocation has no image record')
    return groups
