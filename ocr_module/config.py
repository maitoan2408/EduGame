"""
Cấu hình tập trung cho toàn bộ pipeline OCR.
Bao gồm:
- Ngưỡng chất lượng đầu vào (Quality Gate)
- Các cờ bật/tắt từng bước tiền xử lý (phục vụ ablation test)
- Tham số model Vintern-1B v3.5 & tham số sinh văn bản
- Tham số cắt dải ảnh cho tài liệu dày chữ
- Cổng chất lượng văn bản đầu ra
"""

from dataclasses import dataclass, field
from typing import Any


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_dtype() -> Any:
    try:
        import torch
    except ImportError:
        return "float32"  # Core tests do not perform inference.
    return torch.float16 if torch.cuda.is_available() else torch.float32


@dataclass
class QualityThresholds:
    """Ngưỡng đánh giá chất lượng ảnh đầu vào."""
    blur_threshold: float = 60.0          # Laplacian var < 60 -> cảnh báo/từ chối mờ
    min_brightness: float = 40.0         # Độ sáng trung bình < 40 -> quá tối
    max_brightness: float = 230.0        # Độ sáng trung bình > 230 -> quá cháy sáng
    min_contrast: float = 20.0           # Độ tương phản (std) < 20 -> mờ nhạt, độ tương phản kém
    min_width: int = 300                 # Độ phân giải tối thiểu chiều rộng
    min_height: int = 300                # Độ phân giải tối thiểu chiều cao


@dataclass
class PreprocessConfig:
    """Cấu hình các bước tiền xử lý ảnh (bật/tắt để đo ablation)."""
    enable_exif: bool = True             # Bước 1: Sửa góc quay theo EXIF
    enable_resize: bool = True           # Bước 2: Giới hạn kích thước ảnh
    max_dimension: int = 1600            # Cạnh dài tối đa (px) (Giảm từ 2200 xuống 1600 để tăng tốc)
    
    enable_quality_check: bool = True    # Bước 3: Đo và cảnh báo chất lượng
    
    enable_perspective_warp: bool = False # Bước 5: Nắn phối cảnh (mặc định tắt, chỉ bật khi tìm thấy viền giấy chuẩn)
    perspective_min_area_ratio: float = 0.45
    
    enable_deskew: bool = True           # Bước 6: Chỉnh nghiêng nhỏ
    max_deskew_angle: float = 15.0       # Chỉ nắn trong phạm vi [-15°, 15°]
    
    enable_illumination_fix: bool = True # Bước 7: Cân bằng sáng (CLAHE + bóng đổ)
    illumination_clip_limit: float = 2.0
    
    enable_denoise: bool = False         # Bước 8: Khử nhiễu nhẹ (chỉ khi đo thấy nhiễu hạt)
    enable_sharpen: bool = False         # Bước 9: Unsharp mask nhẹ khi ảnh hơi mờ
    enable_page_regions: bool = False    # Experimental: split long source before resize; opt-in pending real CER.


@dataclass
class ModelConfig:
    """Cấu hình nạp model và sinh văn bản Vintern-1B v3.5."""
    model_name_or_path: str = "5CD-AI/Vintern-1B-v3_5"
    # Ghim commit/tag bất biến khi cần tái lập benchmark hoặc phát hành.
    # None giữ hành vi hiện tại của Hugging Face (revision mặc định).
    revision: str | None = None
    
    # Thiết bị và kiểu dữ liệu (tự chọn phù hợp với phần cứng)
    device: str = field(default_factory=_default_device)
    # Trên RTX 2050 (4GB VRAM) hoặc T4, dùng float16; trên A100/H100 dùng bfloat16
    torch_dtype: Any = field(default_factory=_default_dtype)
    
    # Tham số thị giác InternVL
    input_size: int = 448
    max_num: int = 2                     # Số ô ảnh (tile) tối đa: giảm xuống 2 (từ 4) để tăng gấp đôi tốc độ suy luận
    min_num: int = 1
    
    # Tham số sinh
    max_new_tokens: int = 1024
    do_sample: bool = False
    num_beams: int = 1                   # beam=1 tiết kiệm VRAM đáng kể trên GPU 4GB; có thể thử 3 khi đánh giá
    repetition_penalty: float = 1.2      # 1.2 tránh lặp từ mà không bóp méo chữ lặp hợp lệ
    
    # Prompt mặc định tối ưu cho chép văn bản
    prompt: str = (
        "<image>\n"
        "Chép lại chính xác toàn bộ văn bản trong ảnh, giữ nguyên xuống dòng. "
        "Không giải thích, không thêm nội dung."
    )


@dataclass
class StripSplittingConfig:
    """Cấu hình cắt dải ảnh đối với tài liệu dày chữ vượt ngưỡng token."""
    enable_strip_splitting: bool = True
    split_height_threshold: int = 2400    # Tăng ngưỡng cắt dải lên 2400px để hạn chế chia nhỏ nhiều lần gây chậm
    strip_target_lines: int = 15
    overlap_margin_px: int = 60          # Vùng đè nhau giữa 2 dải liên tiếp để tránh đứt dòng


@dataclass
class QualityGateConfig:
    """Cổng chất lượng hậu xử lý văn bản trích xuất."""
    min_char_count: int = 10             # Ít hơn 10 ký tự -> coi như không đọc được
    min_valid_char_ratio: float = 0.65   # Tỉ lệ ký tự hợp lệ (chữ cái, chữ số, dấu câu thông dụng)
    max_ngram_repetition_ratio: float = 0.35 # Tỉ lệ lặp 4-gram cảnh báo kẹt vòng lặp
    min_vietnamese_diacritic_ratio: float = 0.04 # Tỉ lệ ký tự có dấu nếu là tiếng Việt


@dataclass
class ResourceLimits:
    """Operational guards, independent of OCR quality thresholds."""
    max_file_bytes: int = 20 * 1024 * 1024
    max_page_pixels: int = 20_000_000
    max_pages: int = 50
    max_batch_bytes: int = 128 * 1024 * 1024
    max_session_image_bytes: int = 96 * 1024 * 1024


@dataclass
class OCRModuleConfig:
    """Tập hợp cấu hình tổng thể của module."""
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strip: StripSplittingConfig = field(default_factory=StripSplittingConfig)
    gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    debug: bool = False                  # Ghi log và lưu ảnh trung gian nếu True
    limits: ResourceLimits = field(default_factory=ResourceLimits)
