"""Image-only strip planning shared by experimental page preparation and Vintern."""
from typing import List, Tuple
from PIL import Image
import numpy as np


def find_horizontal_split_points(
    img_pil: Image.Image,
    min_strip_height: int = 600,
    overlap_px: int = 60
) -> List[Tuple[int, int]]:
    """
    Tìm vị trí cắt theo khoảng trống giữa các dòng chữ bằng histogram chiếu ngang.
    Trả về danh sách (y_start, y_end) của từng dải với vùng chồng nhau overlap_px.
    """
    w, h = img_pil.size
    if h <= min_strip_height * 1.5:
        return [(0, h)]

    import cv2
    gray = np.array(img_pil.convert("L"))
    # Điểm tối (chữ) -> nhị phân đảo
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # Histogram tổng điểm chữ theo từng hàng
    horizontal_proj = np.sum(binary, axis=1)

    split_points = [0]
    current_y = min_strip_height

    while current_y < h - (min_strip_height // 2):
        # Tìm vùng ít chữ nhất (tổng điểm tối nhỏ nhất) quanh khoảng current_y +/- 150px
        search_start = max(0, current_y - 120)
        search_end = min(h, current_y + 120)
        valley = int(search_start + np.argmin(horizontal_proj[search_start:search_end]))
        split_points.append(valley)
        current_y = valley + min_strip_height

    split_points.append(h)

    strips_coords = []
    for i in range(len(split_points) - 1):
        y0 = max(0, split_points[i] - (overlap_px if i > 0 else 0))
        y1 = min(h, split_points[i + 1] + (overlap_px if i < len(split_points) - 2 else 0))
        strips_coords.append((y0, y1))

    return strips_coords


def merge_strip_texts(texts: List[str]) -> str:
    """
    Ghép kết quả text từ các dải ảnh liên tiếp và loại bỏ dòng trùng
    ở vùng overlap giữa 2 dải kề nhau.
    """
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0].strip()

    final_lines: List[str] = []

    for t_idx, text in enumerate(texts):
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not final_lines:
            final_lines.extend(lines)
            continue

        # Kiểm tra xem các dòng đầu của dải hiện tại có trùng với các dòng cuối của dải trước không
        overlap_found = False
        for k in range(min(5, len(lines)), 0, -1):
            candidate_prefix = lines[:k]
            candidate_suffix = final_lines[-k:]
            if candidate_prefix == candidate_suffix:
                # Trùng khớp hoàn toàn k dòng ở vùng nối
                final_lines.extend(lines[k:])
                overlap_found = True
                break

        if not overlap_found:
            final_lines.extend(lines)

    return "\n".join(final_lines)
