"""Reuse only the current successful session result, preserving the user's edits."""
import hashlib
from collections import Counter


def upload_identities(files):
    """Stable across reordering; duplicate name/content gets an occurrence suffix."""
    seen = Counter()
    identities = []
    for file in files:
        base = file.name + ':' + hashlib.sha256(file.getvalue()).hexdigest()
        identities.append(f'{base}:{seen[base]}')
        seen[base] += 1
    return tuple(identities)


def release_images(state):
    images = list(state.pop('preview', ())[:2])
    result = state.get('result')
    if result is not None and result.processed_preview is not None:
        images.append(result.processed_preview)
        result.processed_preview = None
    for image in {id(i): i for i in images}.values():
        image.close()
    state.pop('preview_key', None)


def prune_removed_files(states, active_keys):
    for key in list(states):
        if key not in active_keys:
            release_images(states.pop(key))


def get_or_compute(state, context, compute, force=False):
    result = state.get('result')
    if not force and result is not None and result.ok and state.get('result_context') == context:
        return result, True
    return compute(), False
