"""Request-scoped measurements/progress. No global document cache or disk writes."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import logging
from time import perf_counter
from typing import Callable

_ACTIVE = ContextVar('ocr_diagnostics', default=None)


@dataclass
class Diagnostics:
    on_event: Callable | None = None
    on_partial: Callable | None = None
    seconds: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    started: float = field(default_factory=perf_counter)


def _notify(callback, *args):
    if callback:
        try:
            callback(*args)
        except Exception:
            logging.getLogger(__name__).warning('OCR progress callback failed; inference continues')


@contextmanager
def capture(on_event=None, on_partial=None):
    record = Diagnostics(on_event, on_partial)
    token = _ACTIVE.set(record)
    try:
        yield record
    finally:
        _ACTIVE.reset(token)


@contextmanager
def ensure_capture():
    record = _ACTIVE.get()
    if record is not None:
        yield record
    else:
        with capture() as record:
            yield record


def event(name, **metadata):
    record = _ACTIVE.get()
    if record is not None:
        entry = dict(stage=name, **metadata)
        record.events.append(entry)
        _notify(record.on_event, entry)


@contextmanager
def timed(name):
    before = perf_counter()
    event(name)
    try:
        yield
    finally:
        record = _ACTIVE.get()
        if record is not None:
            record.seconds[name] = record.seconds.get(name, 0.0) + perf_counter() - before


def make_streamer(tokenizer, num_beams):
    record = _ACTIVE.get()
    if record is None or record.on_partial is None:
        return None
    if num_beams != 1:
        event('stream_disabled', reason='beam_search')
        return None
    from transformers import TextStreamer

    class PreviewStreamer(TextStreamer):
        def __init__(self):
            super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
            self.visible = ''
            self.last_sent = 0.0
            self.output_token_count = 0
            self.last_output = None

        def put(self, value):
            if not self.next_tokens_are_prompt:
                self.output_token_count += value.numel()
                self.last_output = value.reshape(-1)[-1:].detach()
            super().put(value)

        def on_finalized_text(self, text, stream_end=False):
            self.visible += text
            now = perf_counter()
            if self.visible and 'first_visible_text' not in record.seconds:
                record.seconds['first_visible_text'] = now - record.started
            if stream_end or (text and now - self.last_sent >= 0.15):
                _notify(record.on_partial, self.visible)
                self.last_sent = now

    _notify(record.on_partial, '')
    return PreviewStreamer()
