"""
Module tiện ích đọc/ghi dữ liệu ảnh (io_utils.py).
Hỗ trợ:
- Đọc ảnh từ đường dẫn, bytes (Streamlit UploadedFile), PIL Image, OpenCV BGR
- Tự động chuẩn hóa hướng xoay theo thông tin EXIF
- Đọc định dạng HEIC (từ iPhone) qua pillow-heif
- Đọc PDF scan thành danh sách ảnh PIL qua pypdfium2
- Chuyển đổi qua lại giữa PIL và OpenCV (BGR)
"""

import io
import math
from contextlib import closing
from dataclasses import dataclass
from typing import Union, List, Optional
import numpy as np
from PIL import Image, ImageOps

# Kiểm tra hỗ trợ HEIC
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

# Kiểm tra hỗ trợ PDF
try:
    import pypdfium2 as pdfium
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Chuyển PIL Image sang OpenCV BGR ndarray."""
    rgb = np.array(pil_img.convert("RGB"))
    import cv2
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Chuyển OpenCV BGR ndarray sang PIL Image."""
    import cv2
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def apply_exif_transpose(image: Image.Image) -> tuple[Image.Image, bool]:
    """
    Sửa hướng xoay tự động theo EXIF (rất phổ biến với ảnh chụp điện thoại).
    Trả về (ảnh_sau_khi_xoay, có_bị_xoay_không).
    """
    try:
        orientation = image.getexif().get(274, 1)
        return ImageOps.exif_transpose(image), orientation in (2, 3, 4, 5, 6, 7, 8)
    except Exception:
        return image, False


def load_image_from_bytes(data: bytes, apply_exif: bool = True,
                          max_pixels: int = 20_000_000) -> tuple[Image.Image, list[str]]:
    """
    Nạp ảnh từ mảng bytes (ví dụ từ Streamlit UploadedFile).
    Trả về (PIL.Image, danh_sách_cảnh_báo).
    """
    warnings = []
    bio = io.BytesIO(data)
    try:
        pil_img = Image.open(bio)
        if pil_img.width * pil_img.height > max_pixels:
            pil_img.close()
            raise ValueError(f"Ảnh vượt giới hạn {max_pixels} pixel.")
        pil_img.load()  # Tải toàn bộ pixel vào RAM
    except Exception as e:
        raise ValueError(f"Không thể đọc định dạng ảnh: {e}")

    if apply_exif:
        pil_img, rotated = apply_exif_transpose(pil_img)
        if rotated:
            warnings.append("Đã tự động chỉnh hướng ảnh theo thông tin EXIF.")

    return pil_img.convert("RGB"), warnings


@dataclass
class PDFPageRecord:
    number: int
    image: Optional[Image.Image] = None
    error: str = ''


def iter_pdf_page_records(
    pdf_bytes_or_path: Union[bytes, str],
    scale: float = 2.0,
    start_page: int = 0,
    max_pages: Optional[int] = None,
    max_render_pixels: int = 20_000_000,
    page_limit: int = 50,
):
    """
    Chuyển đổi từng trang của file PDF thành ảnh PIL với độ phân giải cao.
    `scale=2.0` tương đương khoảng 144 DPI (đủ rõ nét cho OCR).
    """
    if not PDF_AVAILABLE:
        raise RuntimeError("Thư viện pypdfium2 chưa được cài đặt. Vui lòng chạy: pip install pypdfium2")

    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale phải lớn hơn 0.")
    if start_page < 0:
        raise ValueError("start_page không được âm.")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages phải lớn hơn 0 nếu được đặt.")

    pdf = pdfium.PdfDocument(pdf_bytes_or_path)
    try:
        total_pages = len(pdf)
        if start_page >= total_pages:
            raise ValueError(f"Trang bắt đầu {start_page + 1} vượt quá tài liệu chỉ có {total_pages} trang.")

        end_page = total_pages if max_pages is None else min(total_pages, start_page + max_pages)
        if end_page - start_page > page_limit:
            raise ValueError(f"Phạm vi vượt giới hạn {page_limit} trang; hãy chọn phạm vi nhỏ hơn.")
        for page_idx in range(start_page, end_page):
            page = bitmap = image = None
            error = ''
            try:
                page = pdf.get_page(page_idx)
                width, height = page.get_size()
                if math.ceil(width * scale) * math.ceil(height * scale) > max_render_pixels:
                    raise ValueError(f"Trang {page_idx + 1} vượt giới hạn {max_render_pixels} pixel render.")
                bitmap = page.render(scale=scale)
                # convert() tạo ảnh độc lập trước khi page/bitmap bị giải phóng.
                image = bitmap.to_pil().convert("RGB")
            except (ValueError, RuntimeError, OSError) as exc:
                error = str(exc)
            finally:
                if bitmap is not None:
                    bitmap.close()
                if page is not None:
                    page.close()
            # No live PDFium page/bitmap is held while inference processes this page.
            yield PDFPageRecord(page_idx + 1, image, error)
    finally:
        pdf.close()


def iter_images_from_pdf(pdf_bytes_or_path, scale=2.0, start_page=0, max_pages=None,
                         max_render_pixels=20_000_000, page_limit=50):
    """Compatibility image iterator: fail on a bad page rather than silently skip."""
    with closing(iter_pdf_page_records(pdf_bytes_or_path, scale, start_page, max_pages,
                                       max_render_pixels, page_limit)) as records:
        for record in records:
            if record.error:
                raise ValueError(record.error)
            yield record.image


def load_images_from_pdf(pdf_bytes_or_path, scale=2.0, start_page=0, max_pages=None) -> List[Image.Image]:
    """Compatibility list API. Pipeline uses the streaming iterator instead."""
    images = []
    try:
        with closing(iter_images_from_pdf(pdf_bytes_or_path, scale, start_page, max_pages)) as pages:
            images.extend(pages)
    except Exception:
        for image in images:
            image.close()
        raise
    return images
