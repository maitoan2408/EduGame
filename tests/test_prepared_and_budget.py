import io
import unittest
from unittest.mock import patch
from PIL import Image

from ocr_module.config import OCRModuleConfig
from ocr_module.pipeline import OCRResult, run_pipeline
from ocr_module.preprocess import run_preprocessing_pipeline
from ocr_module.session_work import image_cache_bytes, enforce_image_budget
from ocr_module.ui_upload import prepare_uploaded_preview


class PreparedImageTests(unittest.TestCase):
    def test_strip_config_change_rejects_stale_prepared_regions(self):
        from ocr_module.pipeline import prepare_image
        cfg = OCRModuleConfig()
        prepared = prepare_image(Image.new('RGB', (400, 400)), cfg)
        cfg.strip.overlap_margin_px += 10
        with patch('ocr_module.pipeline.ocr_image_vintern') as infer:
            result = run_pipeline(prepared, cfg)
        self.assertFalse(result.ok)
        self.assertIn('không khớp', result.reason)
        infer.assert_not_called()

    def test_preview_and_inference_share_exact_processed_image(self):
        cfg = OCRModuleConfig()
        cfg.preprocess.enable_quality_check = False
        cfg.preprocess.enable_illumination_fix = False
        stream = io.BytesIO()
        Image.new('RGB', (1800, 1200), (122, 140, 160)).save(stream, format='PNG')
        with patch('ocr_module.pipeline.run_preprocessing_pipeline', wraps=run_preprocessing_pipeline) as preprocess:
            original, prepared = prepare_uploaded_preview(stream.getvalue(), 0, cfg)
            before = prepared.image.tobytes()
            with patch('ocr_module.pipeline.ocr_image_vintern', return_value='Valid shared prepared output.') as infer:
                result = run_pipeline(prepared, cfg)
            self.assertTrue(result.ok, result.reason)
            preprocess.assert_called_once()
            self.assertIs(infer.call_args.args[0], prepared.image)
            self.assertEqual(result.processed_preview.tobytes(), before)
            self.assertLessEqual(max(original.size), 1600)
        cfg.preprocess.max_dimension = 1200
        with patch('ocr_module.pipeline.ocr_image_vintern') as infer:
            self.assertFalse(run_pipeline(prepared, cfg).ok)
            infer.assert_not_called()

    def test_cache_budget_evicts_rasters_but_retains_text_and_draft(self):
        old_image = Image.new('RGB', (100, 100))
        new_image = Image.new('RGB', (100, 100))
        states = dict(old=dict(result=OCRResult(text='old text', ok=True, processed_preview=old_image),
                               draft_text='old edit', _last_touched=1),
                      new=dict(result=OCRResult(text='new text', ok=True, processed_preview=new_image),
                               _last_touched=2))
        self.assertEqual(image_cache_bytes(states), 60000)
        self.assertEqual(enforce_image_budget(states, 30000), 30000)
        self.assertEqual(states['old']['result'].text, 'old text')
        self.assertEqual(states['old']['draft_text'], 'old edit')
        self.assertIsNone(states['old']['result'].processed_preview)
        self.assertIs(states['new']['result'].processed_preview, new_image)
