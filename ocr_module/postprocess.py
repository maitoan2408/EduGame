"""
Module hậu xử lý và Cổng chất lượng văn bản (postprocess.py).
Bao gồm:
- Chuẩn hóa Unicode NFC (chống lệch dấu tổ hợp tiếng Việt)
- Làm sạch định dạng markdown thừa từ model sinh ra
- Chuẩn hóa ngắt dòng, khoảng trắng (giữ cấu trúc 'Thuật ngữ: Định nghĩa')
- Giữ dòng lặp của tài liệu, phát hiện kẹt vòng lặp n-gram bằng quality gate
- Cổng chất lượng (Quality Gate): đánh giá tính hợp lệ của văn bản trích xuất
"""

import re
import unicodedata
from typing import Tuple, List
from collections import Counter
from .config import QualityGateConfig


VIETNAMESE_DIACRITICS = set(
    "àáảãạăằắẳẵặâầấẩẫậ"
    "èéẻẽẹêềếểễệ"
    "ìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữự"
    "ỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
    "ÈÉẺẼẸÊỀẾỂỄỆ"
    "ÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰ"
    "ỲÝỶỸỴĐ"
)


def normalize_unicode_nfc(text: str) -> str:
    """Chuẩn hóa chuỗi văn bản về chuẩn Unicode NFC dựng sẵn."""
    return unicodedata.normalize("NFC", text)


def clean_markdown_and_artifacts(text: str) -> str:
    """Conservative markup cleanup; preserve multiplication, powers and code."""
    cleaned = re.sub(r"^[ \t]*#{1,6}[ \t]+", "", text, flags=re.MULTILINE)
    # Only a whole-line heading: inline ** can be Python/math exponentiation.
    cleaned = re.sub(r"^([ \t]*)\*\*([^\W\d_][^*\n]*[^\W\d_])\*\*[ \t]*$", r"\1\2", cleaned, flags=re.MULTILINE)
    # Only strip model tokens at the boundaries, not literal tags inside a lesson.
    token = r"</?(?:s|image|assistant|user)>"
    cleaned = re.sub(rf"^(?:{token})+|(?:{token})+$", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def deduplicate_consecutive_lines(text: str) -> str:
    """Compatibility API: preserve repeated source lines; the gate detects loops.

    Identical adjacent lines can be valid answers or table cells. Without spatial
    provenance it is unsafe to delete them. Keep this callable for existing callers.
    """
    return text


def normalize_whitespace_and_paragraphs(text: str) -> str:
    """Normalize line endings and trailing whitespace without guessing layout."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def calculate_repetition_rate(text: str, n: int = 4) -> float:
    """
    Tính tỉ lệ lặp n-gram từ (mặc định 4 từ liên tiếp) để phát hiện kẹt vòng lặp.
    """
    words = [w.lower() for w in re.findall(r"\w+", text)]
    if len(words) < n * 2:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated_count = sum(c - 1 for c in counts.values() if c > 1)
    return float(repeated_count) / float(len(ngrams))


def validate_extracted_text(
    text: str, 
    cfg: QualityGateConfig
) -> Tuple[bool, str, List[str]]:
    """
    Kiểm tra chất lượng văn bản sau trích xuất:
    - Độ dài tối thiểu
    - Tỉ lệ ký tự hợp lệ
    - Tỉ lệ lặp từ n-gram
    - Tỉ lệ dấu tiếng Việt (nếu phát hiện là văn bản tiếng Việt)
    """
    warnings: List[str] = []
    cleaned_len = len(text.strip())
    
    # 1. Kiểm tra độ dài tối thiểu
    if cleaned_len < cfg.min_char_count:
        return (
            False,
            f"Văn bản trích xuất quá ngắn ({cleaned_len} ký tự). Model có thể không đọc được hoặc trang rỗng.",
            warnings
        )
        
    # 2. Tỉ lệ ký tự hợp lệ (chữ cái, chữ số, dấu câu)
    content = ''.join(c for c in normalize_unicode_nfc(text) if not c.isspace())
    if not any(c.isalnum() for c in content):
        return False, 'Văn bản không có chữ hoặc số có thể sử dụng.', warnings
    valid_chars = sum(1 for c in content if c.isalnum() or c in ".,;:?!-_'\"()[]{}/@#%&+=*<>≤≥≠×÷√∑∫∞")
    valid_ratio = valid_chars / max(len(content), 1)
    if valid_ratio < cfg.min_valid_char_ratio:
        return (
            False,
            f"Chất lượng chữ kém, xuất hiện nhiều ký tự rác (tỉ lệ hợp lệ chỉ đạt {valid_ratio * 100:.1f}%).",
            warnings
        )
        
    # 3. Phát hiện kẹt vòng lặp (Repetition Loop)
    rep_ratio = calculate_repetition_rate(text, n=4)
    if rep_ratio > cfg.max_ngram_repetition_ratio:
        return (
            False,
            f"Phát hiện văn bản bị lặp lại bất thường (tỉ lệ lặp {rep_ratio * 100:.1f}%). Model bị kẹt vòng lặp suy luận.",
            warnings
        )
    elif rep_ratio > 0.15:
        warnings.append("Văn bản có một số cụm từ lặp lại nhiều lần.")
        
    # 4. Kiểm tra dấu tiếng Việt nếu có dấu hiệu tiếng Việt
    total_letters = sum(1 for c in text if c.isalpha())
    if total_letters > 20:
        diacritic_chars = sum(1 for c in text if c in VIETNAMESE_DIACRITICS)
        diacritic_ratio = diacritic_chars / total_letters
        # Nếu có chữ cái tiếng Việt nhưng tỉ lệ dấu quá thấp (ví dụ < 1%)
        if 0 < diacritic_ratio < cfg.min_vietnamese_diacritic_ratio:
            warnings.append(
                f"Tỉ lệ dấu tiếng Việt thấp ({diacritic_ratio * 100:.1f}%), có thể văn bản bị mất dấu hoặc là tiếng Anh/song ngữ."
            )

    return True, "", warnings


def clean_and_gate_text(
    raw_text: str, 
    cfg: QualityGateConfig
) -> Tuple[str, bool, str, List[str]]:
    """
    Quy trình hậu xử lý hoàn chỉnh:
    Làm sạch -> Chuẩn hóa -> Cổng chất lượng.
    """
    # Bước 1: Chuẩn hóa NFC
    text = normalize_unicode_nfc(raw_text)
    
    # Bước 2: Dọn markdown và artifacts
    text = clean_markdown_and_artifacts(text)
    
    # Bước 3: Giữ dòng lặp hợp lệ (không thể suy provenance chỉ từ chuỗi)
    text = deduplicate_consecutive_lines(text)
    
    # Bước 4: Chuẩn hóa khoảng trắng & ngắt đoạn
    text = normalize_whitespace_and_paragraphs(text)
    
    # Bước 5: Cổng kiểm tra chất lượng
    ok, reason, warnings = validate_extracted_text(text, cfg)
    
    return text, ok, reason, warnings
