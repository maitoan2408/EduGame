import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from ocr_module.config import OCRModuleConfig
from ocr_module.pipeline import prepare_image, run_pipeline
from ocr_module.session_work import enforce_image_budget


class PageRegionTests(unittest.TestCase):
    def config(self):
        cfg = OCRModuleConfig()
        cfg.preprocess.enable_quality_check = False
        cfg.preprocess.enable_deskew = False
        cfg.preprocess.enable_illumination_fix = False
        return cfg

    def test_default_unchanged_opt_in_reaches_raw_page_regions(self):
        cfg = self.config()
        source = Image.fromarray(np.full((3000, 1200, 3), 140, dtype=np.uint8))
        baseline = prepare_image(source, cfg)
        self.assertEqual(baseline.regions, [])
        cfg.preprocess.enable_page_regions = True
        candidate = prepare_image(source, cfg)
        self.assertEqual(candidate.image.tobytes(), baseline.image.tobytes())
        self.assertGreater(len(candidate.regions), 1)
        self.assertTrue(all(max(r.image.size) <= 1600 for r in candidate.regions))
        responses = [f'Region {index} contains unique valid learning content.' for index in range(len(candidate.regions))]
        with patch('ocr_module.pipeline.ocr_image_vintern', side_effect=responses) as infer:
            result = run_pipeline(candidate, cfg)
        self.assertTrue(result.ok, result.reason)
        self.assertTrue(result.metrics['requires_review'])
        self.assertEqual(infer.call_count, len(candidate.regions))
        self.assertEqual(result.text, '\n'.join(responses))
        states = dict(current=dict(preview=(source, candidate), result=result))
        self.assertEqual(enforce_image_budget(states, 0), 0)
        self.assertEqual(states['current']['result'].text, '\n'.join(responses))
        with self.assertRaises(ValueError):
            candidate.regions[0].image.getpixel((0, 0))

    def test_manual_rotation_is_applied_once_before_region_planning(self):
        cfg = self.config()
        cfg.preprocess.enable_page_regions = True
        prepared = prepare_image(Image.new('RGB', (3000, 1200), 'white'), cfg, 90)
        self.assertGreater(len(prepared.regions), 1)
        with patch('ocr_module.pipeline.ocr_image_vintern', return_value='Valid region content.'):
            result = run_pipeline(prepared, cfg, manual_rotation_deg=90)
        self.assertTrue(result.ok, result.reason)
