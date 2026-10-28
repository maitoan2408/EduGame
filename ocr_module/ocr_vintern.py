"""
Module tích hợp mô hình Vintern-1B v3.5 (ocr_vintern.py).
Bao gồm:
- Hàm tiền xử lý ảnh động (dynamic_preprocess, load_image_tensor) sao chép chuẩn từ model card Vintern-1B v3.5
- Quản lý nạp model & tokenizer (Singleton cache)
- Cơ chế cắt dải ảnh (Strip Splitting) dựa trên histogram chiếu ngang cho trang dày chữ
- Xử lý tràn VRAM (OOM Fallback) tự động giảm số ô ảnh (max_num)
- Hàm ocr_image_vintern(img_pil, cfg) trả về text trích xuất
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from functools import wraps
from threading import RLock
import gc
import re
import warnings

from .config import ModelConfig, StripSplittingConfig
from .diagnostics import timed, event, make_streamer

# Các hằng số chuẩn ImageNet theo model card
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# -------------------------------------------------------------
# 1. Các hàm xử lý ảnh từ Model Card 5CD-AI/Vintern-1B-v3_5
# -------------------------------------------------------------

def build_transform(input_size: int = 448):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
        
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_tensor(image: Image.Image, input_size: int = 448, max_num: int = 4) -> torch.Tensor:
    """Chuyển PIL.Image thành tensor pixel_values sẵn sàng cho Vintern."""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# -------------------------------------------------------------
# 2. Quản lý tải mô hình (Model Loader Cache)
# -------------------------------------------------------------

_GLOBAL_MODEL = None
_GLOBAL_TOKENIZER = None
_GLOBAL_MODEL_KEY = None
_MODEL_LOCK = RLock()


def _serialized(function):
    @wraps(function)
    def locked(*args, **kwargs):
        with _MODEL_LOCK:
            return function(*args, **kwargs)
    return locked


@_serialized
def clear_model_cache():
    """Explicit release, also used before loading a different model identity."""
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_MODEL_KEY
    _GLOBAL_MODEL = _GLOBAL_TOKENIZER = _GLOBAL_MODEL_KEY = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@_serialized
def get_vintern_model(cfg: ModelConfig):
    """
    Nạp model và tokenizer một lần duy nhất vào bộ nhớ.
    Hỗ trợ cả GPU (CUDA) và fallback CPU.
    """
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_MODEL_KEY
    model_key = (cfg.model_name_or_path, cfg.revision, cfg.device, str(cfg.torch_dtype))
    if (
        _GLOBAL_MODEL is not None
        and _GLOBAL_TOKENIZER is not None
        and _GLOBAL_MODEL_KEY == model_key
    ):
        event('model_cache', hit=True, device=cfg.device, dtype=str(cfg.torch_dtype))
        return _GLOBAL_MODEL, _GLOBAL_TOKENIZER

    event('model_cache', hit=False, device=cfg.device, dtype=str(cfg.torch_dtype))

    if _GLOBAL_MODEL is not None:
        clear_model_cache()
    if cfg.revision is None or not re.fullmatch(r"[0-9a-fA-F]{40}", cfg.revision):
        warnings.warn("Vintern revision chưa ghim SHA bất biến; chưa đủ điều kiện benchmark/phát hành tái lập.", RuntimeWarning)

    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel
    
    # Patch for transformers >= 5.x or 4.40+ with custom models
    if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
        PreTrainedModel.all_tied_weights_keys = {}

    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = cfg.torch_dtype

    print(f"[Vintern] Dang tai mo hinh {cfg.model_name_or_path} tren thiet bi {device} ({dtype})...")
    
    load_kwargs = {"trust_remote_code": True}
    if cfg.revision is not None:
        load_kwargs["revision"] = cfg.revision
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=False, **load_kwargs)
    
    model = AutoModel.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        **load_kwargs,
    ).eval()
    
    if use_cuda:
        model = model.cuda()
    else:
        model = model.to("cpu")

    _GLOBAL_MODEL = model
    _GLOBAL_TOKENIZER = tokenizer
    _GLOBAL_MODEL_KEY = model_key
    print("[Vintern] Tai mo hinh thanh cong.")
    return _GLOBAL_MODEL, _GLOBAL_TOKENIZER


# -------------------------------------------------------------
# 3. Kỹ thuật cắt dải ảnh cho tài liệu dày chữ (Strip Splitting)
# -------------------------------------------------------------

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


def split_image_into_strips(img_pil: Image.Image, strip_cfg: StripSplittingConfig) -> List[Image.Image]:
    """Cắt ảnh thành các dải ngang nếu kích thước vượt ngưỡng cho phép."""
    if strip_cfg.strip_target_lines != 15:
        raise ValueError("strip_target_lines là trường dự phòng chưa hỗ trợ thay đổi.")
    w, h = img_pil.size
    if not strip_cfg.enable_strip_splitting or h < strip_cfg.split_height_threshold:
        return [img_pil]

    coords = find_horizontal_split_points(img_pil, min_strip_height=800, overlap_px=strip_cfg.overlap_margin_px)
    strips = [img_pil.crop((0, y0, w, y1)) for (y0, y1) in coords]
    return strips


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


# -------------------------------------------------------------
# 4. Hàm thực thi suy luận OCR trên Vintern-1B v3.5
# -------------------------------------------------------------

def _chat_strip(model, tokenizer, strip, cfg, gen_config, max_num, use_cuda):
    """Local tensor scope releases references before the caller starts an OOM retry."""
    with timed('tensor_transfer'):
        pixel_values = load_image_tensor(strip, input_size=cfg.input_size, max_num=max_num).to(cfg.torch_dtype)
        if use_cuda:
            pixel_values = pixel_values.cuda()
    generation = dict(gen_config)
    streamer = make_streamer(tokenizer, cfg.num_beams)
    if streamer is not None:
        generation['streamer'] = streamer
    with torch.inference_mode():
        with timed('generation'):
            response, _ = model.chat(tokenizer, pixel_values, cfg.prompt, generation,
                                     history=None, return_history=True)
    return response


@_serialized
def ocr_image_vintern(
    img_pil: Image.Image, 
    cfg: ModelConfig, 
    strip_cfg: StripSplittingConfig
) -> str:
    """
    Thực thi OCR trên ảnh bằng Vintern-1B v3.5.
    Tự động chia dải nếu trang dài, và có cơ chế xử lý chống tràn VRAM (OOM fallback).
    """
    if cfg.max_num < 1 or cfg.min_num != 1:
        raise ValueError("max_num phải >= 1; min_num hiện chỉ hỗ trợ 1.")
    strips = split_image_into_strips(img_pil, strip_cfg)
    with timed('model_load_or_cache'):
        model, tokenizer = get_vintern_model(cfg)
    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    dtype = cfg.torch_dtype

    strip_outputs: List[str] = []

    gen_config = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        num_beams=cfg.num_beams,
        repetition_penalty=cfg.repetition_penalty
    )

    for strip_index, strip in enumerate(strips):
        event('strip', index=strip_index + 1, total=len(strips))
        try:
            response = _chat_strip(model, tokenizer, strip, cfg, gen_config, cfg.max_num, use_cuda)
        except torch.cuda.OutOfMemoryError:
            if cfg.max_num <= 1:
                raise
            print(f"[Vintern] OOM: retry max_num={cfg.max_num} -> {max(1, cfg.max_num // 2)}")
            event('oom_retry', from_tiles=cfg.max_num, to_tiles=max(1, cfg.max_num // 2))
        else:
            strip_outputs.append(response)
            continue
        # Outside the except block: exception traceback and failed tensors are gone.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        response = _chat_strip(model, tokenizer, strip, cfg, gen_config,
                               max(1, cfg.max_num // 2), use_cuda)
        strip_outputs.append(response)

    return merge_strip_texts(strip_outputs)
