"""CPU torch + fake model: no weights, GPU or Hugging Face network calls."""
import gc
import importlib.util
import unittest
import weakref
from unittest.mock import Mock, patch

from PIL import Image

HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'ML tests run in the separate CPU inference CI job')
class InferenceMockTests(unittest.TestCase):
    def setUp(self):
        import torch
        from ocr_module import ocr_vintern as engine
        from ocr_module.config import ModelConfig, StripSplittingConfig
        self.engine, self.torch = engine, torch
        self.cfg = ModelConfig(device='cpu', torch_dtype=torch.float32)
        self.strip = StripSplittingConfig(enable_strip_splitting=False)
        self.image = Image.new('RGB', (400, 400))

    def test_oom_releases_failed_tensor_before_half_tile_retry(self):
        torch, engine = self.torch, self.engine
        refs, tiles = [], []
        class Pixels:
            def to(self, dtype): return self
        def tensor(image, input_size, max_num):
            if refs:
                gc.collect()
                self.assertIsNone(refs[0](), 'failed tensor still alive before retry')
            value = Pixels()
            refs.append(weakref.ref(value))
            tiles.append(max_num)
            return value
        class Model:
            calls = 0
            def chat(model, *args, **kwargs):
                model.calls += 1
                if model.calls == 1:
                    raise torch.cuda.OutOfMemoryError('simulated')
                return 'safe output', None
        with patch.object(engine, 'get_vintern_model', return_value=(Model(), object())), patch.object(engine, 'load_image_tensor', side_effect=tensor):
            self.assertEqual(engine.ocr_image_vintern(self.image, self.cfg, self.strip), 'safe output')
        self.assertEqual(tiles, [2, 1])

    def test_retry_failure_is_bounded_and_propagates(self):
        model = Mock()
        model.chat.side_effect = self.torch.cuda.OutOfMemoryError('still out of memory')
        with patch.object(self.engine, 'get_vintern_model', return_value=(model, object())), patch.object(self.engine, 'load_image_tensor', return_value=self.torch.zeros(1)):
            with self.assertRaises(self.torch.cuda.OutOfMemoryError):
                self.engine.ocr_image_vintern(self.image, self.cfg, self.strip)
        self.assertEqual(model.chat.call_count, 2)

    def test_normal_path_preserves_generation_and_dtype(self):
        model = Mock()
        model.chat.return_value = ('safe output', None)
        with patch.object(self.engine, 'get_vintern_model', return_value=(model, object())), patch.object(self.engine, 'load_image_tensor', return_value=self.torch.zeros(1)):
            result = self.engine.ocr_image_vintern(self.image, self.cfg, self.strip)
        self.assertEqual(result, 'safe output')
        args = model.chat.call_args.args
        self.assertEqual(args[1].dtype, self.torch.float32)
        self.assertEqual(args[2], self.cfg.prompt)
        self.assertEqual(args[3], dict(max_new_tokens=1024, do_sample=False, num_beams=1, repetition_penalty=1.2))

    def test_estimated_token_limit_marks_result_for_review(self):
        from ocr_module.config import OCRModuleConfig
        from ocr_module.pipeline import run_pipeline
        from ocr_module.handoff import deliver_text
        cfg = OCRModuleConfig()
        cfg.model.max_new_tokens = 4
        cfg.preprocess.enable_quality_check = False
        cfg.preprocess.enable_illumination_fix = False
        class Tokenizer:
            def encode(self, text, **kwargs): return [1, 2, 3, 4]
        model = Mock()
        model.chat.return_value = ('Valid but possibly incomplete lesson.', None)
        with patch.object(self.engine, 'get_vintern_model', return_value=(model, Tokenizer())), patch.object(self.engine, 'load_image_tensor', return_value=self.torch.zeros(1)):
            result = run_pipeline(self.image, cfg)
        self.assertTrue(result.ok)
        self.assertTrue(result.metrics['requires_review'])
        self.assertTrue(any('giới hạn token' in w for w in result.warnings))
        with self.assertRaises(ValueError):
            deliver_text(result=result, text=result.text, consumer=Mock(), gate=cfg.gate,
                         result_context='x', current_context='x')

    def test_stream_token_budget_distinguishes_eos_from_potential_cutoff(self):
        from ocr_module.diagnostics import capture
        torch = self.torch
        self.cfg.max_new_tokens = 2
        class Tokenizer:
            def __init__(self, eos): self.eos_token_id = eos
            def decode(self, values, **kwargs): return 'Valid streamed text '
        class Model:
            def chat(model, tokenizer, pixels, prompt, generation, **kwargs):
                streamer = generation['streamer']
                for value in [torch.tensor([[1]]), torch.tensor([2]), torch.tensor([3])]:
                    streamer.put(value)
                streamer.end()
                return 'Valid streamed text', None
        for eos in (3, 9):
            with capture(on_partial=lambda text: None) as record, patch.object(self.engine, 'get_vintern_model', return_value=(Model(), Tokenizer(eos))), patch.object(self.engine, 'load_image_tensor', return_value=torch.zeros(1)):
                self.engine.ocr_image_vintern(self.image, self.cfg, self.strip)
            observed = next(e for e in record.events if e['stage'] == 'generation_output')
            self.assertEqual(observed['output_tokens'], 2)
            self.assertFalse(observed['estimated'])
            self.assertEqual(any(e['stage'] == 'possible_truncation' for e in record.events), eos != 3)

    def test_splitting_and_exact_overlap_merge(self):
        e = self.engine
        self.assertEqual(e.merge_strip_texts(['alpha\nbeta', 'beta\ngamma']), 'alpha\nbeta\ngamma')
        self.assertEqual(e.merge_strip_texts([]), '')
        self.assertEqual(e.merge_strip_texts(['a', 'b']), 'a\nb')
        from ocr_module.config import StripSplittingConfig
        strips = e.split_image_into_strips(Image.new('RGB', (600, 2600), 'white'), StripSplittingConfig())
        self.assertGreater(len(strips), 1)

    def test_reserved_flag_is_not_a_silent_noop(self):
        self.strip.strip_target_lines = 20
        with self.assertRaisesRegex(ValueError, 'strip_target_lines'):
            self.engine.split_image_into_strips(self.image, self.strip)

    def test_real_tensor_shape_and_normalization(self):
        value = self.engine.load_image_tensor(self.image, input_size=448, max_num=2)
        self.assertEqual(tuple(value.shape[1:]), (3, 448, 448))
        self.assertTrue(self.torch.isfinite(value).all())

    def test_streaming_preview_does_not_replace_final_response_or_generation_settings(self):
        from ocr_module.diagnostics import capture
        class Tokenizer:
            def decode(self, tokens, **kwargs):
                return ''.join({1: 'Private prompt', 2: 'Early ', 3: 'text.'}[int(t)] for t in tokens)
        torch = self.torch
        class Model:
            def chat(model, tokenizer, pixels, prompt, generation, **kwargs):
                streamer = generation['streamer']
                streamer.put(torch.tensor([[1]]))  # skipped prompt
                streamer.put(torch.tensor([2]))
                streamer.put(torch.tensor([3]))
                streamer.end()
                return 'Final canonical response.', None
        previews = []
        with capture(on_partial=previews.append) as record, patch.object(self.engine, 'get_vintern_model', return_value=(Model(), Tokenizer())), patch.object(self.engine, 'load_image_tensor', return_value=torch.zeros(1)):
            result = self.engine.ocr_image_vintern(self.image, self.cfg, self.strip)
        self.assertEqual(result, 'Final canonical response.')
        self.assertEqual(previews[-1], 'Early text.')
        self.assertNotIn('Private prompt', ''.join(previews))
        self.assertIn('generation', record.seconds)

    def test_streaming_disabled_for_beams_without_changing_generation(self):
        from ocr_module.diagnostics import capture, make_streamer
        with capture(on_partial=Mock()) as record:
            self.assertIsNone(make_streamer(object(), 3))
        self.assertEqual(record.events[-1]['stage'], 'stream_disabled')

    def test_streamer_preserves_actual_transformers_generated_token_ids(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from ocr_module.diagnostics import capture, make_streamer
        torch = self.torch
        class Tokenizer:
            def decode(self, tokens, **kwargs):
                return ' '.join(str(int(t)) for t in tokens) + ' '
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_embd=8, n_head=1, n_layer=1,
                                             n_positions=32, eos_token_id=None, pad_token_id=0)).eval()
            inputs = torch.tensor([[1, 2]])
            args = dict(max_new_tokens=4, do_sample=False, num_beams=1, pad_token_id=0)
            with torch.inference_mode():
                baseline = model.generate(inputs, **args)
                previews = []
                with capture(on_partial=previews.append):
                    candidate = model.generate(inputs, streamer=make_streamer(Tokenizer(), 1), **args)
            self.assertTrue(torch.equal(baseline, candidate))
            self.assertTrue(previews[-1].strip())

    def test_model_cache_reuses_and_passes_explicit_revision(self):
        engine = self.engine
        self.cfg.revision = 'a' * 40
        model = Mock()
        model.eval.return_value = model
        model.to.return_value = model
        tokenizer = object()
        engine.clear_model_cache()
        try:
            with patch('transformers.AutoModel.from_pretrained', return_value=model) as loader, patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer):
                first = engine.get_vintern_model(self.cfg)
                second = engine.get_vintern_model(self.cfg)
                self.assertEqual(first, second)
                loader.assert_called_once()
                self.assertEqual(loader.call_args.kwargs['revision'], 'a' * 40)
                self.assertTrue(loader.call_args.kwargs['trust_remote_code'])
                self.cfg.revision = 'b' * 40
                engine.get_vintern_model(self.cfg)
                self.assertEqual(loader.call_count, 2)
        finally:
            engine.clear_model_cache()

    def test_loader_serializes_concurrent_sessions(self):
        from concurrent.futures import ThreadPoolExecutor
        import time
        engine = self.engine
        engine.clear_model_cache()
        self.cfg.revision = 'a' * 40
        model = Mock()
        model.eval.return_value = model
        model.to.return_value = model
        def slow_load(*args, **kwargs):
            time.sleep(0.02)
            return model
        try:
            with patch('transformers.AutoModel.from_pretrained', side_effect=slow_load) as loader, patch('transformers.AutoTokenizer.from_pretrained', return_value=object()):
                with ThreadPoolExecutor(2) as pool:
                    results = list(pool.map(lambda _: engine.get_vintern_model(self.cfg), range(2)))
                self.assertIs(results[0][0], results[1][0])
                loader.assert_called_once()
        finally:
            engine.clear_model_cache()
