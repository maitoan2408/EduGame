"""
Module điều phối toàn bộ Pipeline (pipeline.py).
Ghép nối các tầng:
IO -> Preprocess -> Quality Check -> OCR Engine (Vintern-1B v3.5) -> Postprocess & Quality Gate.
Trả về dataclass chuẩn OCRResult sẵn sàng tích hợp với module tạo câu hỏi ôn tập.
"""

from dataclasses import dataclass, field
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
from .diagnostics import timed, event


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


def process_single_image(
    pil_img: Image.Image,
    cfg: OCRModuleConfig,
    manual_rotation_deg: int = 0
) -> Tuple[str, bool, str, List[str], Dict[str, Any], Image.Image]:
    """
    Xử lý tiền xử lý, OCR và hậu xử lý cho một ảnh PIL duy nhất.
    """
    # 1. Chuyển sang OpenCV BGR
    img_bgr = pil_to_cv2(pil_img)
    
    # 2. Tiền xử lý ảnh (Resize, Quality, Deskew, Illumination, ...)
    with timed('preprocess'):
        proc_bgr, q_passed, q_reason, prep_warnings, metrics = run_preprocessing_pipeline(
            img_bgr, prep_cfg=cfg.preprocess, quality_cfg=cfg.quality,
            manual_rotation_deg=manual_rotation_deg,
        )
    
    preview_pil = cv2_to_pil(proc_bgr)
    
    # Nếu ảnh không qua cổng chất lượng đầu vào (quá mờ, quá tối, hỏng nặng)
    if not q_passed:
        return "", False, q_reason, prep_warnings, metrics, preview_pil

    # 3. Thực thi nhận dạng chữ bằng Vintern-1B v3.5
    try:
        raw_text = ocr_image_vintern(preview_pil, cfg.model, cfg.strip)
    except Exception as e:
        return "", False, f"Lỗi trong quá trình nhận dạng mô hình Vintern: {str(e)}", prep_warnings, metrics, preview_pil

    # 4. Hậu xử lý văn bản và Cổng chất lượng đầu ra
    with timed('postprocess'):
        clean_text, gate_ok, gate_reason, post_warnings = clean_and_gate_text(raw_text, cfg.gate)
    
    all_warnings = prep_warnings + post_warnings
    return clean_text, gate_ok, gate_reason, all_warnings, metrics, preview_pil


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
        if isinstance(item, Image.Image):
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
    source: Union[bytes, str, Image.Image, List[Any]],
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
    return OCRResult(
        text="\n\n--- HẾT TRANG ---\n\n".join(p for p in page_texts if p).strip(),
        ok=bool(page_texts) and not failures,
        reason="; ".join(failures), warnings=warnings, pages=page_texts,
        metrics=last_metrics, processed_preview=last_preview,
    )
