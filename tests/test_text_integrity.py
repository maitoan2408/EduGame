import unittest
from unittest.mock import Mock

from ocr_module.config import QualityGateConfig
from ocr_module.handoff import deliver_text
from ocr_module.pipeline import OCRResult
from ocr_module.postprocess import clean_and_gate_text, validate_extracted_text


class TextIntegrityTests(unittest.TestCase):
    def test_formulas_code_and_answer_layout_survive_cleanup(self):
        cases = ['Công thức: 2 * 3 * 4 = 24.', 'Power: x**y**z and __init__', 'Power: x **y** z',
                 '#include <stdio.h>\nint main() { return 0; }',
                 'A. Photosynthesis\nB. Respiration\nC. Transpiration',
                 'Name\tScore\nAlice\t10\nBob\t20',
                 'Đáp án: A\n\nĐáp án: A', 'Nội dung bài học\nNội dung bài học']
        for source in cases:
            with self.subTest(source=source):
                cleaned, ok, reason, _ = clean_and_gate_text(source, QualityGateConfig())
                self.assertEqual(cleaned, source)
                self.assertTrue(ok, reason)

    def test_whitespace_cannot_hide_garbage_or_punctuation_only_content(self):
        consumer = Mock()
        for text in ['?' * 30, '.' * 30, '\n' * 100 + '💥' * 10,
                     'word' + ' ' * 1000 + '💥' * 100]:
            with self.subTest(text=text[:20]):
                self.assertFalse(validate_extracted_text(text, QualityGateConfig())[0])
                with self.assertRaises(ValueError):
                    deliver_text(result=OCRResult(text='valid original text', ok=True), text=text,
                                 consumer=consumer, gate=QualityGateConfig(), reviewed=True,
                                 result_context='x', current_context='x')
        consumer.assert_not_called()

    def test_numeric_and_bilingual_material_still_passes(self):
        for text in ['1234567890', 'x * y + z = 100', 'Tiếng Việt and English material.']:
            self.assertTrue(validate_extracted_text(text, QualityGateConfig())[0])
