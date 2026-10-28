import importlib.util
import unittest
from unittest.mock import patch

from ocr_module.config import OCRModuleConfig
from ocr_module.pipeline import OCRResult


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU inference CI job')
class ProfileTests(unittest.TestCase):
    def test_profile_reports_metrics_without_document_text(self):
        from tools.profile_ocr import measure
        cfg = OCRModuleConfig()
        cfg.model.device = 'cpu'
        with patch('ocr_module.pipeline.run_pipeline', return_value=OCRResult(text='private lesson', ok=True, pages=['private lesson'])):
            result = measure(b'fixture', cfg, truth='private lesson')
        self.assertEqual(result['cer'], 0)
        self.assertEqual(result['page_count'], 1)
        self.assertFalse(result['cuda_available'])
        self.assertIsNone(result['cuda_peak_allocated_bytes'])
        self.assertNotIn('private lesson', str(result))

    def test_rejection_is_not_scored_as_successful_cer(self):
        from tools.profile_ocr import measure
        cfg = OCRModuleConfig()
        cfg.model.device = 'cpu'
        with patch('ocr_module.pipeline.run_pipeline', return_value=OCRResult(text='', ok=False, reason='failed')):
            result = measure(b'fixture', cfg, truth='ground truth')
        self.assertFalse(result['ok'])
        self.assertIsNone(result['cer'])


class MemoryMetricsTests(unittest.TestCase):
    def test_process_peak_is_available_on_supported_platform(self):
        from tools.process_memory import peak_rss_bytes
        self.assertGreater(peak_rss_bytes(), 0)

    def test_windows_api_keeps_pointer_sized_counters(self):
        from tools.process_memory import peak_rss_bytes
        from unittest.mock import Mock
        kernel, psapi = Mock(), Mock()
        def query(handle, pointer, size):
            pointer._obj.PeakWorkingSetSize = 5 * 1024**3
            return True
        psapi.GetProcessMemoryInfo.side_effect = query
        with patch('sys.platform', 'win32'), patch('ctypes.WinDLL', side_effect=[kernel, psapi], create=True):
            self.assertEqual(peak_rss_bytes(), 5 * 1024**3)
