"""Explicit opt-in for PDF text layers, never silently used for scanned pages."""
from pathlib import Path
from time import perf_counter

from .config import OCRModuleConfig
from .io_utils import PDF_AVAILABLE
from .pipeline import OCRResult
from .postprocess import clean_and_gate_text


def read_pdf_text(source: bytes | str, cfg: OCRModuleConfig | None = None,
                  start_page: int = 0, max_pages: int | None = None) -> OCRResult:
    cfg = cfg or OCRModuleConfig()
    before = perf_counter()
    pages, failures, details = [], [], []
    try:
        if not PDF_AVAILABLE:
            raise ValueError('Thiếu pypdfium2.')
        size = len(source) if isinstance(source, bytes) else Path(source).stat().st_size
        if size > cfg.limits.max_file_bytes:
            raise ValueError('PDF vượt giới hạn dung lượng.')
        if start_page < 0 or (max_pages is not None and max_pages <= 0):
            raise ValueError('Phạm vi trang không hợp lệ.')
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(source)
        try:
            end = len(pdf) if max_pages is None else min(len(pdf), start_page + max_pages)
            if start_page >= len(pdf) or end - start_page > cfg.limits.max_pages:
                raise ValueError('Phạm vi ngoài tài liệu hoặc vượt giới hạn số trang.')
            for index in range(start_page, end):
                page = pdf.get_page(index)
                textpage = None
                try:
                    textpage = page.get_textpage()
                    if textpage.count_chars() > 200_000:
                        raise ValueError('Lớp chữ vượt giới hạn 200000 ký tự/trang.')
                    raw = textpage.get_text_range()
                    text, ok, reason, _ = clean_and_gate_text(raw, cfg.gate)
                    if not ok:
                        reason = 'Không có lớp chữ dùng được. Hãy chọn Nhận dạng ảnh bằng Vintern. ' + reason
                    pages.append(text if ok else '')
                    details.append(dict(page=index + 1, ok=ok, reason=reason))
                    if not ok:
                        failures.append(f'[Trang {index + 1}] {reason}')
                finally:
                    if textpage is not None:
                        textpage.close()
                    page.close()
        finally:
            pdf.close()
    except (ValueError, OSError, RuntimeError) as exc:
        failures.append(f'Không thể đọc lớp chữ PDF: {exc}')
    return OCRResult(text='\n\n--- HẾT TRANG ---\n\n'.join(p for p in pages if p),
                     ok=bool(pages) and not failures, reason='; '.join(failures), pages=pages,
                     warnings=['Đang dùng lớp văn bản có sẵn. Chữ trong hình ảnh có thể chưa được lấy; hãy đối chiếu và xác nhận.'],
                     metrics=dict(method='pdf_text', requires_review=True, per_page=details,
                                  total_seconds=perf_counter() - before))
