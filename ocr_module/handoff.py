"""OCR → consumer boundary. Consumers receive only a validated plain string."""
from typing import Callable

from .config import QualityGateConfig
from .pipeline import OCRResult
from .postprocess import validate_extracted_text

TextConsumer = Callable[[str], None]


def deliver_text(*, result: OCRResult, text: str, consumer: TextConsumer | None,
                 gate: QualityGateConfig, result_context: str,
                 current_context: str, reviewed: bool = False) -> None:
    """Validate provenance/review/gates, then synchronously invoke the consumer.

    A returning callback acknowledges receipt, not question generation or game success.
    Exceptions propagate; the caller must not display success in that case.
    """
    if not current_context or result_context != current_context:
        raise ValueError("Kết quả đã cũ. Hãy trích xuất lại với file và cấu hình hiện tại.")
    if not callable(consumer):
        raise ValueError("Chưa kết nối nơi tiếp nhận văn bản của game.")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Không thể chuyển văn bản rỗng.")
    if (not result.ok or text != result.text or result.metrics.get('requires_review', False)) and not reviewed:
        raise ValueError("Hãy kiểm tra và xác nhận thủ công bản văn trước khi chuyển.")
    ok, reason, _ = validate_extracted_text(text, gate)
    if not ok:
        raise ValueError(reason)
    consumer(text)
