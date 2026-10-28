"""Explicit, immutable-revision OCR measurement; prints metadata, never document text."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def measure(source, cfg, truth=None):
    import torch
    from ocr_module.pipeline import run_pipeline
    from ocr_module.eval.run_eval import calculate_cer
    from ocr_module.diagnostics import capture
    cuda = cfg.model.device == 'cuda' and torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    before, cpu = time.perf_counter(), time.process_time()
    with capture() as diagnostic:
        result = run_pipeline(source, cfg)
    if cuda:
        torch.cuda.synchronize()
    record = dict(ok=result.ok, reason=result.reason, page_count=len(result.pages), char_count=len(result.text),
                  elapsed_seconds=time.perf_counter()-before, cpu_seconds=time.process_time()-cpu,
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None,
                  cuda_available=cuda, torch=str(torch.__version__), config=asdict(cfg),
                  python=platform.python_version(), platform=platform.platform())
    record['stages_seconds'] = diagnostic.seconds
    record['execution'] = diagnostic.events
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record['process_peak_rss_bytes'] = peak if sys.platform == 'darwin' else peak * 1024
    except ImportError:
        record['process_peak_rss_bytes'] = None
    # Rejection is reported separately; do not score a rejected result as an accurate OCR.
    record['cer'] = calculate_cer(result.text, truth) if truth is not None and result.ok else None
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--revision', required=True, help='Approved immutable 40-character model SHA')
    parser.add_argument('--truth', type=Path)
    parser.add_argument('--runs', type=int, default=1, help='Repeat in the same process to separate cold/warm runs')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-fA-F]{40}', args.revision):
        parser.error('--revision must be an immutable 40-character SHA')
    if not 1 <= args.runs <= 10:
        parser.error('--runs must be between 1 and 10')
    from ocr_module.config import OCRModuleConfig
    cfg = OCRModuleConfig()
    cfg.model.revision = args.revision
    truth = args.truth.read_text(encoding='utf-8') if args.truth else None
    all_ok = True
    for index in range(args.runs):
        report = measure(args.source, cfg, truth)
        report['run'] = index + 1
        print(json.dumps(report, ensure_ascii=False, default=str))
        all_ok = all_ok and report['ok']
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
