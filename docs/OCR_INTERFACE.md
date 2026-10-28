# Bàn giao văn bản OCR cho nhóm game

## Interface

`main(text_consumer: Callable[[str], None] | None = None)` trong `ocr_module.ui_upload` là điểm nhúng UI.
Consumer là hàm đồng bộ nhận duy nhất một `str`, trả về sau khi nhận dữ liệu hoặc raise exception nếu nhận thất bại.
Không cần import Vintern, torch, PIL hay `OCRResult` trong consumer của game.

```python
from ocr_module.ui_upload import main

def accept_learning_text(text: str) -> None:
    # Nhóm game nối hàm thật của mình ở đây; raise nếu không nhận được.
    receive_text_in_game_module(text)

main(text_consumer=accept_learning_text)
```

`receive_text_in_game_module` là điểm nối do nhóm game cung cấp, không được triển khai trong phân hệ OCR.
Khi chạy `streamlit run app.py` độc lập, nút chuyển bị vô hiệu hóa vì chưa có consumer.
UI chỉ báo nhận thành công sau khi callback trả về. Điều này chưa chứng minh sinh câu hỏi hoặc game đã chạy.
`game_raw_text` được cập nhật để tương thích sau khi callback thành công, không dùng làm bằng chứng giao nhận.

## Điều kiện chuyển

- `OCRResult` giữ nguyên bảy trường, kiểu và default. `ok` biểu thị kiểm tra OCR; xác nhận tay không sửa `ok` thành True.
- `metrics` giữ các khóa trang cuối như trước, bổ sung `per_page`: danh sách `{page, ok, reason, metrics}`. `pages` vẫn để chuỗi rỗng cho trang lỗi; `reason` tổng hợp mọi lỗi. Với PDF chọn trang 2–3, số trang trong chẩn đoán là 2 và 3.
- `deliver_text(...)` kiểm tra context file/config/góc xoay/phạm vi PDF khớp kết quả hiện tại.
- Chuỗi phải không rỗng và qua `validate_extracted_text`; không tự dọn hoặc sửa bản người dùng đã xác nhận.
- Bất kỳ sửa tay hoặc `result.ok=False` đều cần checkbox xác nhận đúng nội dung hiện tại.
- Xác nhận không bỏ qua gate văn bản. Ví dụ PDF có trang lỗi: người dùng phải kiểm tra bản thiếu trang trước khi xác nhận; nội dung gốc không được coi là đầy đủ.
- Đổi file, cấu hình, góc xoay, phạm vi trang hoặc chạy OCR lại làm mất hiệu lực kết quả cũ. Đổi nội dung làm mất hiệu lực xác nhận cũ.
- Ngoại lệ consumer được hiển thị; không lưu trạng thái giao thành công.

## Kiểm chứng

`tests/test_handoff.py`: contract string, empty/rejected/stale/review/consumer exception.
`tests/test_handoff_integration.py`: PDFium thật → pipeline với OCR mock → receiver nhận string.
`tests/test_ui_handoff.py`: Streamlit AppTest cho ảnh/PDF, sửa tay, context và thay kết quả.
Các test dùng receiver giả; nhóm game cần nối và kiểm thử consumer thật trong nhánh tích hợp của nhóm.

## Giới hạn tài nguyên

Cập nhật logic: upload ID ổn định khi thêm/sắp xếp file; đổi config giữ sửa tay thành draft chỉ để tải về, không tự bàn giao. PDF lỗi từng trang giữ pages rỗng tại trang lỗi và vẫn xử lý trang sau nếu document còn đọc được.
`PreparedImage` là dữ liệu chuẩn bị nội bộ cho preview/inference, được kiểm tra chữ ký cấu hình và góc; không phải contract mà nhóm game cần nhận. Game tiếp tục chỉ nhận str.
Thêm 128 MiB tổng upload/lượt, 96 MiB raster cache/session; cache ảnh bị loại không làm mất text/draft. Các giới hạn này không bao gồm toàn bộ RAM process/model.
Đầu ra nghi chạm giới hạn token hoặc chia vùng thử nghiệm có `requires_review=True`; xác nhận tay không biến chúng thành bằng chứng tài liệu đầy đủ.

Đường đọc lớp chữ PDF opt-in dùng `read_pdf_text` và cùng OCRResult. `metrics.requires_review=True` bắt buộc xác nhận tay trong adapter, kể cả khi text gate pass; lớp chữ có thể thiếu nội dung nằm trong ảnh. Không dùng đường này âm thầm cho scan.
`metrics.timings`/`execution` là chẩn đoán bổ sung; callback/preview streaming không được coi là dữ liệu bàn giao. Handoff chỉ nhận kết quả cuối đã qua gate/review.

`OCRModuleConfig.limits`: 20 MiB/file, 20 triệu pixel/ảnh hoặc trang PDF sau render, tối đa 50 trang/lượt. Vượt giới hạn trả kết quả thất bại rõ ràng, không âm thầm cắt nội dung.
UI mặc định chọn một trang PDF. Pipeline render và giải phóng lần lượt từng trang; không giữ toàn bộ raster PDF trong RAM. `load_images_from_pdf` giữ API list cho caller cũ, không nên dùng với PDF dài.
Các giới hạn vận hành không thay ngưỡng chất lượng. Không lưu upload ra đĩa.
`enable_perspective_warp=True` hiện trả lỗi rõ ràng vì thuật toán chưa triển khai; không bật transform mới trong vòng này.
