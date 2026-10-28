import unittest
from unittest.mock import patch
from PIL import Image

from ocr_module.pipeline import run_pipeline
from ocr_module.pdf_text import read_pdf_text
import test_pdf_streaming as fixtures


class PageRecoveryTests(unittest.TestCase):
    def test_failed_middle_page_keeps_position_and_attempts_last_page(self):
        doc, pages = fixtures.PdfStreamingTests().document(3)
        pages[1].render.side_effect = RuntimeError('page 2 render failed')
        def process(*args):
            return 'Valid page text.', True, '', [], {}, Image.new('RGB', (10, 10))
        with patch('ocr_module.io_utils.pdfium.PdfDocument', return_value=doc), patch('ocr_module.pipeline.process_single_image', side_effect=process):
            result = run_pipeline(b'%PDF-fixture')
        self.assertFalse(result.ok)
        self.assertEqual(result.pages, ['Valid page text.', '', 'Valid page text.'])
        self.assertIn('[Trang 2]', result.reason)
        self.assertEqual([p['ok'] for p in result.metrics['per_page']], [True, False, True])
        pages[2].render.assert_called_once()
        doc.close.assert_called_once()

    def test_native_pdf_text_also_recovers_after_page_error(self):
        doc, pages = fixtures.PdfStreamingTests().document(3)
        for page in pages:
            page.get_textpage.return_value.count_chars.return_value = 25
            page.get_textpage.return_value.get_text_range.return_value = 'Valid native lesson text.'
        pages[1].get_textpage.side_effect = RuntimeError('broken text page')
        with patch('pypdfium2.PdfDocument', return_value=doc):
            result = read_pdf_text(b'%PDF-fixture')
        self.assertFalse(result.ok)
        self.assertEqual(result.pages, ['Valid native lesson text.', '', 'Valid native lesson text.'])
        self.assertIn('[Trang 2]', result.reason)
        pages[2].get_textpage.assert_called_once()
        doc.close.assert_called_once()
