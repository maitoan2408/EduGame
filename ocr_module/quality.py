"""
Module kiểm tra chất lượng ảnh đầu vào (quality.py).
Đo các chỉ số vật lý của ảnh:
- Độ mờ (Blur / Sharpness qua phương sai toán tử Laplacian)
- Độ sáng trung bình (Mean luminance)
- Độ tương phản (Standard deviation của ảnh xám)
- Kích thước & Độ phân giải tối thiểu
Cổng kiểm tra chất lượng đưa ra cảnh báo hoặc từ chối xử lý ảnh hỏng nặng.
"""

from typing import Dict, Any, Tuple, List
import cv2
import numpy as np
from .config import QualityThresholds


def measure_image_quality(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Đo đạc các chỉ số chất lượng ảnh trên không gian màu xám.
    Trả về dict gồm: blur, brightness, contrast, width, height.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {
            "blur": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "w": 0,
            "h": 0,
            "valid": False
        }

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Đo độ mờ bằng phương sai Laplacian (Variance of Laplacian)
    # Giá trị thấp = mờ; giá trị cao = biên cạnh rõ nét
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # 2. Độ sáng trung bình (0: đen hoàn toàn -> 255: trắng lóa)
    brightness = float(gray.mean())
    
    # 3. Độ tương phản (Độ lệch chuẩn std)
    contrast = float(gray.std())
    
    return {
        "blur": laplacian_var,
        "brightness": brightness,
        "contrast": contrast,
        "w": w,
        "h": h,
        "valid": True
    }


def evaluate_quality(
    metrics: Dict[str, Any], 
    thresholds: QualityThresholds
) -> Tuple[bool, str, List[str]]:
    """
    Đánh giá dựa trên ngưỡng trong config.
    Trả về:
    - passed: bool (có thể tiếp tục đưa vào OCR hay không)
    - reason: str (lý do từ chối nếu không qua cổng)
    - warnings: list[str] (các cảnh báo cần hiển thị cho người dùng)
    """
    warnings: List[str] = []
    
    if not metrics.get("valid", False):
        return False, "Ảnh không hợp lệ hoặc dữ liệu điểm ảnh rỗng.", []

    w, h = metrics["w"], metrics["h"]
    blur = metrics["blur"]
    brightness = metrics["brightness"]
    contrast = metrics["contrast"]

    # Kiểm tra kích thước
    if w < thresholds.min_width or h < thresholds.min_height:
        return (
            False,
            f"Kích thước ảnh quá nhỏ ({w}x{h} px). Yêu cầu tối thiểu {thresholds.min_width}x{thresholds.min_height} px để đọc được chữ.",
            warnings
        )

    # Kiểm tra độ mờ nặng (không thể cứu được)
    # Nếu blur < 1/3 ngưỡng thì từ chối hẳn, nếu từ [1/3, 1] ngưỡng thì cảnh báo
    if blur < (thresholds.blur_threshold / 3.0):
        return (
            False, 
            f"Ảnh quá mờ (độ nét đo được: {blur:.1f}, ngưỡng: {thresholds.blur_threshold:.1f}). Chữ bị nhòe không thể nhận dạng chính xác, vui lòng chụp lại rõ nét hơn.",
            warnings
        )
    elif blur < thresholds.blur_threshold:
        warnings.append(f"Ảnh hơi mờ (độ nét: {blur:.1f} < {thresholds.blur_threshold:.1f}), văn bản có thể khó đọc.")

    # Kiểm tra độ sáng
    if brightness < thresholds.min_brightness:
        warnings.append(f"Ảnh khá tối (độ sáng: {brightness:.1f}/255), văn bản có thể khó đọc.")
    elif brightness > thresholds.max_brightness:
        warnings.append(f"Ảnh bị lóa sáng / cháy sáng (độ sáng: {brightness:.1f}/255).")

    # Kiểm tra độ tương phản
    if contrast < thresholds.min_contrast:
        warnings.append(f"Độ tương phản thấp (contrast: {contrast:.1f}), văn bản có thể mờ nhạt so với nền.")

    return True, "", warnings
