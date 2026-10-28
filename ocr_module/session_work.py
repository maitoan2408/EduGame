"""Reuse only the current successful session result, preserving the user's edits."""
def get_or_compute(state, context, compute, force=False):
    result = state.get('result')
    if not force and result is not None and result.ok and state.get('result_context') == context:
        return result, True
    return compute(), False
