import unittest
from unittest.mock import Mock, patch

from ocr_module.diagnostics import capture, timed, event
from ocr_module.pdf_text import read_pdf_text
from ocr_module.pipeline import OCRResult
from ocr_module.session_work import get_or_compute
from test_pdf_io import make_one_page_pdf


def text_pdf():
    stream = b'BT /F1 16 Tf 30 100 Td (This is exact selectable learning text.) Tj ET'
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
               b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
               b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'\nendstream']
    document = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(document))
        document.extend(f'{index} 0 obj\n'.encode() + obj + b'\nendobj\n')
    xref = len(document)
    document.extend(b'xref\n0 6\n0000000000 65535 f \n')
    for offset in offsets[1:]: document.extend(f'{offset:010d} 00000 n \n'.encode())
    document.extend(f'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    return bytes(document)


class FastPathTests(unittest.TestCase):
    def test_native_text_is_exact_without_rendering_or_model(self):
        with patch('pypdfium2.PdfPage.render', side_effect=AssertionError('must not render')), patch('ocr_module.pipeline.ocr_image_vintern', side_effect=AssertionError('must not infer')):
            result = read_pdf_text(text_pdf(), max_pages=1)
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.text, 'This is exact selectable learning text.')
        self.assertTrue(result.metrics['requires_review'])
        self.assertEqual(result.metrics['method'], 'pdf_text')

    def test_scanned_blank_or_invalid_pdf_never_returns_success(self):
        self.assertFalse(read_pdf_text(make_one_page_pdf()).ok)
        self.assertFalse(read_pdf_text(b'%PDF-corrupt').ok)
        self.assertFalse(read_pdf_text(text_pdf(), start_page=3).ok)

    def test_successful_current_result_reused_but_failures_force_or_new_context_recompute(self):
        result = OCRResult(text='unchanged text', ok=True)
        state = dict(result=result, result_context='current', user_text='user edit')
        compute = Mock(return_value=OCRResult(text='new', ok=True))
        self.assertEqual(get_or_compute(state, 'current', compute), (result, True))
        compute.assert_not_called()
        get_or_compute(state, 'other', compute)
        get_or_compute(state, 'current', compute, force=True)
        result.ok = False
        get_or_compute(state, 'current', compute)
        self.assertEqual(compute.call_count, 3)
        self.assertEqual(state['user_text'], 'user edit')

    def test_request_local_timings_and_failing_progress_do_not_change_results(self):
        with capture(on_event=Mock(side_effect=ValueError)) as outer:
            with timed('outer'): pass
            with capture() as inner:
                with timed('inner'): pass
            event('done')
        self.assertIn('outer', outer.seconds)
        self.assertNotIn('inner', outer.seconds)
        self.assertIn('inner', inner.seconds)
