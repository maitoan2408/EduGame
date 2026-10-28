"""
Module điều phối toàn bộ Pipeline (pipeline.py).
Ghép nối các tầng:
IO -> Preprocess -> Quality Check -> OCR Engine (Vintern-1B v3.5) -> Postprocess & Quality Gate.
Trả về dataclass chuẩn OCRResult sẵn sàng tích hợp với module tạo câu hỏi ôn tập.
"""

from dataclasses import dataclass, field, replace
from typing import List, Union, Optional, Dict, Any, Tuple
from PIL import Image

from .config import OCRModuleConfig
from .io_utils import (
    load_image_from_bytes, 
    pil_to_cv2, 
    cv2_to_pil, 
    iter_pdf_page_records,
)
from .preprocess import run_preprocessing_pipeline
from .postprocess import clean_and_gate_text
from .diagnostics import timed, event, ensure_capture


def ocr_image_vintern(image, model_cfg, strip_cfg):
    """Load ML dependencies only when inference is actually requested."""
    from .ocr_vintern import ocr_image_vintern as infer
    return infer(image, model_cfg, strip_cfg)


@dataclass
class OCRResult:
    """Đầu ra chuẩn để bàn giao cho module tạo câu hỏi ôn tập / sinh game."""
    text: str
    ok: bool                                 # Có vượt qua các cổng kiểm tra chất lượng không
    reason: str = ""                         # Lý do cụ thể nếu ok=False (để giao diện hướng dẫn người dùng chụp lại)
    warnings: List[str] = field(default_factory=list)  # Các cảnh báo (VD: "Ảnh hơi mờ", "Đã xoay 90°", v.v.)
    pages: List[str] = field(default_factory=list)     # Nội dung text theo từng trang / từng ảnh
    metrics: Dict[str, Any] = field(default_factory=dict) # Các chỉ số đo lường ảnh (độ mờ, sáng, tương phản)
    processed_preview: Optional[Image.Image] = None     # Ảnh PIL sau tiền xử lý để hiển thị so sánh trên UI


@dataclass
class PreparedImage:
    image: Image.Image
    ok: bool
    reason: str
    warnings: List[str]
    metrics: Dict[str, Any]
    signature: str
    rotation: int
    regions: List['PreparedImage'] = field(default_factory=list)


def preparation_signature(cfg):
    return repr((cfg.preprocess, cfg.quality, cfg.limits, cfg.strip))


def prepare_image(pil_img, cfg, manual_rotation_deg=0):
    if manual_rotation_deg % 90:
        raise ValueError("Góc xoay phải là bội số của 90 độ.")
    if pil_img.width * pil_img.height > cfg.limits.max_page_pixels:
        raise ValueError("Ảnh vượt giới hạn pixel.")
    with timed('preprocess'):
        processed, ok, reason, warnings, metrics = run_preprocessing_pipeline(
            pil_to_cv2(pil_img), cfg.preprocess, cfg.quality, manual_rotation_deg,
        )
    prepared = PreparedImage(cv2_to_pil(processed), ok, reason, list(warnings), dict(metrics),
                             preparation_signature(cfg), manual_rotation_deg)
    prepared.metrics.update(input_width=pil_img.width, input_height=pil_img.height)
    source_height = pil_img.width if manual_rotation_deg % 180 else pil_img.height
    if cfg.strip.enable_strip_splitting and source_height >= cfg.strip.split_height_threshold:
        if cfg.preprocess.enable_page_regions:
            from .regions import find_horizontal_split_points
            source = pil_img
            if manual_rotation_deg % 360:
                rotation = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}[manual_rotation_deg % 360]
                source = pil_img.transpose(rotation)
            child_cfg = region_config(cfg)
            try:
                for y0, y1 in find_horizontal_split_points(source, min_strip_height=800, overlap_px=cfg.strip.overlap_margin_px):
                    crop = source.crop((0, y0, source.width, y1))
                    try:
                        prepared.regions.append(prepare_image(crop, child_cfg))
                    finally:
                        crop.close()
            except Exception:
                for child in prepared.regions:
                    child.image.close()
                prepared.image.close()
                raise
            finally:
                if source is not pil_img:
                    source.close()
            prepared.warnings.append('Đang thử nghiệm chia trang dài trước resize; cần đối chiếu đầy đủ nội dung.')
        elif processed.shape[0] < cfg.strip.split_height_threshold:
            prepared.warnings.append('Trang dài đã thu nhỏ dưới ngưỡng chia dải. Có thể thử chia trang dài trước resize hoặc cung cấp ảnh từng vùng để kiểm tra.')
    return prepared


def region_config(cfg):
    return replace(cfg, preprocess=replace(cfg.preprocess, enable_page_regions=False),
                   strip=replace(cfg.strip, enable_strip_splitting=False))


def process_single_image(pil_img, cfg, manual_rotation_deg=0):
    reused = isinstance(pil_img, PreparedImage)
    prepared = pil_img if reused else prepare_image(pil_img, cfg, manual_rotation_deg)
    if prepared.signature != preparation_signature(cfg) or prepared.rotation != manual_rotation_deg:
        raise ValueError("Ảnh chuẩn bị không khớp cấu hình/góc xoay; hãy chuẩn bị lại.")
    event('preprocess_cache', hit=reused)
    preview = prepared.image
    metrics = dict(prepared.metrics)
    warnings = list(prepared.warnings)
    if prepared.regions:
        from .regions import merge_strip_texts
        outputs, errors, details = [], [], []
        for index, region in enumerate(prepared.regions, 1):
            text, ok, reason, child_warnings, child_metrics, _ = process_single_image(region, region_config(cfg))
            outputs.append(text if ok else '')
            warnings.extend(f'[Vùng {index}] {w}' for w in child_warnings)
            details.append(dict(region=index, ok=ok, reason=reason, metrics=child_metrics))
            if not ok:
                errors.append(f'[Vùng {index}] {reason}')
        text, gate_ok, reason, gate_warnings = clean_and_gate_text(merge_strip_texts(outputs), cfg.gate)
        metrics['regions'] = details
        metrics['requires_review'] = True  # Experimental assembly must be reviewed, even if gates pass.
        return text, gate_ok and not errors, '; '.join(errors + ([reason] if reason else [])), warnings + gate_warnings, metrics, preview
    if not prepared.ok:
        return "", False, prepared.reason, warnings, metrics, preview
    with ensure_capture() as diagnostic:
        start_event = len(diagnostic.events)
        try:
            raw_text = ocr_image_vintern(preview, cfg.model, cfg.strip)
        except Exception as exc:
            return "", False, f"Lỗi trong quá trình nhận dạng mô hình Vintern: {exc}", warnings, metrics, preview
        execution = diagnostic.events[start_event:]
        metrics['inference'] = execution
        if any(e['stage'] == 'possible_truncation' for e in execution):
            metrics['requires_review'] = True
            warnings.append("Đầu ra có thể chạm giới hạn token và thiếu nội dung. Hãy đối chiếu tài liệu hoặc chia nhỏ trang trước khi chuyển.")
    with timed('postprocess'):
        text, ok, reason, post_warnings = clean_and_gate_text(raw_text, cfg.gate)
    return text, ok, reason, warnings + post_warnings, metrics, preview


def _iter_inputs(source, cfg, pdf_start_page, pdf_max_pages):
    """Yield (image, warnings, owned) without pre-rendering an entire PDF."""
    from pathlib import Path
    if isinstance(source, str):
        path = Path(source)
        if path.stat().st_size > cfg.limits.max_file_bytes:
            raise ValueError("Tập tin vượt giới hạn dung lượng.")
        if path.suffix.lower() == ".pdf":
            yield from _iter_pdf(source, cfg, pdf_start_page, pdf_max_pages)
            return
        source = path.read_bytes()
    if isinstance(source, bytes) and source.startswith(b"%PDF-"):
        if len(source) > cfg.limits.max_file_bytes:
            raise ValueError("PDF vượt giới hạn dung lượng.")
        yield from _iter_pdf(source, cfg, pdf_start_page, pdf_max_pages)
        return
    items = source if isinstance(source, list) else [source]
    if len(items) > cfg.limits.max_pages:
        raise ValueError("Số ảnh vượt giới hạn số trang.")
    for item in items:
        if isinstance(item, PreparedImage):
            yield item, [], False, ''
        elif isinstance(item, Image.Image):
            if item.width * item.height > cfg.limits.max_page_pixels:
                raise ValueError("Ảnh vượt giới hạn pixel.")
            yield item, [], False, ''
        elif isinstance(item, bytes):
            if len(item) > cfg.limits.max_file_bytes:
                raise ValueError("Ảnh vượt giới hạn dung lượng.")
            image, warnings = load_image_from_bytes(
                item, apply_exif=cfg.preprocess.enable_exif, max_pixels=cfg.limits.max_page_pixels,
            )
            yield image, warnings, True, ''
        else:
            raise ValueError("Kiểu đầu vào không hỗ trợ; dùng bytes, path, PIL hoặc list ảnh.")


def _iter_pdf(source, cfg, start, count):
    from contextlib import closing
    with closing(iter_pdf_page_records(
        source, start_page=start, max_pages=count,
        max_render_pixels=cfg.limits.max_page_pixels, page_limit=cfg.limits.max_pages,
    )) as pages:
        while True:
            with timed('pdf_read_render'):
                try:
                    record = next(pages)
                except StopIteration:
                    return
            yield record.image, [], True, record.error


def run_pipeline(
    source: Union[bytes, str, Image.Image, PreparedImage, List[Any]],
    cfg: Optional[OCRModuleConfig] = None,
    manual_rotation_deg: int = 0,
    pdf_start_page: int = 0,
    pdf_max_pages: Optional[int] = None,
) -> OCRResult:
    """Sequential OCR. Failed pages remain empty; any failure makes ok=False."""
    from contextlib import closing
    cfg = cfg or OCRModuleConfig()
    is_pdf = ((isinstance(source, bytes) and source.startswith(b"%PDF-"))
              or (isinstance(source, str) and source.lower().endswith(".pdf")))
    page_texts, warnings, failures, page_details = [], [], [], []
    last_preview = None
    last_metrics = {}
    try:
        if manual_rotation_deg % 90:
            raise ValueError("Góc xoay phải là bội số của 90 độ.")
        if min(cfg.limits.max_pages, cfg.limits.max_page_pixels, cfg.limits.max_file_bytes) <= 0:
            raise ValueError("Giới hạn tài nguyên phải lớn hơn 0.")
        with closing(_iter_inputs(source, cfg, pdf_start_page, pdf_max_pages)) as images:
            for index, (image, input_warnings, owned, read_error) in enumerate(images):
                number = index + 1 + (pdf_start_page if is_pdf else 0)
                event('page', number=number)
                try:
                    if read_error:
                        raise ValueError(read_error)
                    text, ok, reason, page_warnings, metrics, preview = process_single_image(
                        image, cfg, manual_rotation_deg,
                    )
                except Exception as exc:
                    text, ok, reason = "", False, f"Lỗi xử lý trang: {exc}"
                    page_warnings, metrics, preview = [], {}, None
                finally:
                    if owned and image is not None:
                        image.close()
                prefix = f"[Trang {number}] "
                warnings.extend(prefix + w for w in input_warnings + page_warnings)
                if not ok:
                    failures.append(prefix + reason)
                page_texts.append(text if ok else "")
                page_details.append(dict(page=number, ok=ok, reason=reason, metrics=dict(metrics)))
                last_metrics = dict(metrics)
                if last_preview is not None and last_preview is not preview:
                    last_preview.close()
                last_preview = preview
    except (ValueError, OSError, RuntimeError) as exc:
        failures.append(f"Không thể đọc ảnh/PDF: {exc}")
    if not page_texts and not failures:
        failures.append("Không tìm thấy ảnh hợp lệ để xử lý.")
    if is_pdf:
        warnings.insert(0, f"Đã xử lý {len(page_texts)} trang PDF, từ trang {pdf_start_page + 1}.")
    # Keep legacy top-level last-page metrics and preview, add page-indexed diagnostics.
    if page_details:
        last_metrics["per_page"] = page_details
        if any(p['metrics'].get('requires_review') for p in page_details):
            last_metrics['requires_review'] = True
    return OCRResult(
        text="\n\n--- HẾT TRANG ---\n\n".join(p for p in page_texts if p).strip(),
        ok=bool(page_texts) and not failures,
        reason="; ".join(failures), warnings=warnings, pages=page_texts,
        metrics=last_metrics, processed_preview=last_preview,
    )
