# Thiết kế giảm thời gian chờ OCR — RTX 2050

## Vấn đề và bằng chứng

Người dùng Windows, RTX 2050, Python 3.12.10, torch 2.14.1+cu126 báo tổng xử lý một ảnh PDF gần 2 phút. Chưa có PDF/ground truth, chưa biết đây là cold start hay warm inference. Cloud hiện chỉ có torch CPU, không có Vintern weights; không thể tái hiện số 120 giây hoặc đo mức cải thiện GPU thật.

Baseline 0ac2e43: 49 tests pass. Code đã cache model giữa các lượt, nhưng UI vẫn có thể chạy inference lại cùng kết quả và chạy preprocess preview khi widget rerun. Chưa có thời gian từng giai đoạn hoặc văn bản sớm trước khi model.chat kết thúc.

## Thiết kế đã triển khai

| Loại đầu vào/thao tác | Đường xử lý | Chất lượng và tính an toàn |
|---|---|---|
| PDF scan/ảnh — mặc định | Vintern cùng model/prompt/dtype/tiles/token cap như baseline | Stream chữ theo từ, tối đa khoảng 7 lần cập nhật UI/giây; preview dải hiện tại chưa qua gate, không giao cho game. Kết quả cuối vẫn lấy nguyên response của model.chat rồi postprocess/gate |
| PDF có lớp chữ — người dùng chọn | PDFium text layer, không render và không gọi model | Luôn cảnh báo có thể thiếu chữ trong ảnh; yêu cầu review tay qua adapter. Không tự chuyển PDF scan sang đường này |
| Bấm lại cùng kết quả thành công/context | Tái sử dụng kết quả hiện tại | Giữ bản người dùng đã sửa. Nút Nhận dạng lại từ đầu mới ép chạy model lại. Kết quả fail không được cache như thành công |
| Sửa text/xác nhận/chuyển game | Reuse preview theo file/góc/config preprocess/quality/limits | Không chạy lại CPU preprocess khi chỉ thao tác editor. Không cache toàn cục hoặc ghi thêm upload ra đĩa |

Streaming là giảm thời gian đến **chữ đầu tiên**, không hứa giảm thời gian hoàn tất. Beam 3 không hỗ trợ streamer của Transformers, nên đường này giữ kết quả cuối và có thông báo. OOM vẫn retry một lần với nửa tiles; streamer được tạo mới khi retry để xóa preview không hoàn chỉnh của lần lỗi.

`diagnostics.capture` dùng ContextVar cho từng request, không trộn hai session. Callback progress lỗi không làm hỏng inference. Các trường timing nằm trong metrics chẩn đoán, không đổi bảy trường OCRResult. Có sự kiện model_cache hit/miss, device/dtype, page, strip và OOM retry.

## Cách kiểm tra trên máy Windows

1. Dừng Streamlit bằng Ctrl+C, `git pull --ff-only origin toanbranch`, chạy lại Streamlit trong cùng venv.
2. Chọn một trang PDF scan, giữ 2 tile/beam 1/1024 token mặc định. Bật Hiện chữ trong lúc nhận dạng. Đừng dùng chế độ lớp chữ cho PDF scan.
3. Bấm Trích xuất; xem các trạng thái tải model/nhận dạng và preview. Sau hoàn tất mở Chi tiết thời gian xử lý.
4. Bấm **Nhận dạng lại từ đầu** trên cùng tài liệu/cấu hình để đo warm run. Bấm Trích xuất lại sẽ reuse kết quả, không phải phép đo inference thứ hai.
5. Gửi `total_seconds`, `stages`, `execution` của cả hai lượt. Không cần gửi nội dung tài liệu nếu riêng tư. Chỉ cần thêm một mẫu được phép dùng và truth khi đánh giá CER.

Các stage là wall time trên CPU, GPU transfer có thể bất đồng bộ. `first_visible_text` là mốc thời gian từ đầu request, không cộng vào tổng. Tổng còn bao gồm IO, chờ lock và xử lý UI. `model_load_or_cache` không tách download khỏi load weights. Không cộng cơ học các mốc/stage để suy GPU kernel time.

Đo bằng CLI với model SHA đã review:

```powershell
.\.venv\Scripts\python.exe tools/profile_ocr.py "C:\path\sample.pdf" --revision APPROVED_40_CHAR_SHA --runs 2
```

Thay placeholder bằng SHA thật. Profiler không có UI result cache và chạy hai lần trong cùng process, báo thời gian và VRAM peak. UI thời gian đo chỉ bắt đầu lúc xử lý request, không bao gồm thời gian bạn upload hoặc tải trang ứng dụng ban đầu.

## Kiểm thử và phép đo hiện có

- Test actual Transformers generate trên decoder nhỏ khởi tạo local: token IDs trước/sau gắn streamer phải giống hệt. Đây là contract của streaming, không phải CER Vintern.
- Test model giả: không phát prompt ra preview, giữ response cuối, generation/dtype giữ nguyên; OOM/cache/regression cũ tiếp tục được kiểm tra.
- PDF có lớp chữ tổng hợp, 20 lượt Linux: median 0.000287 giây, max 0.006161 giây; chuỗi trích xuất khớp chính xác fixture. Chỉ đo một fixture ngắn; không được quảng cáo là tốc độ PDF thật hay thay thế phép đo scan.
- Test UI: reuse không gọi inference và không làm mất sửa tay; đổi cách đọc vô hiệu context; lớp chữ cần xác nhận; fallback đường scan là lựa chọn rõ ràng của người dùng.

## Bước tối ưu tiếp theo dựa vào số đo

| Nút thắt được đo | Hướng xử lý | Điều kiện chấp nhận |
|---|---|---|
| Load model chiếm phần lớn | Giữ process mở, dùng cache, chuẩn bị model trước buổi demo | So cold/warm, không lặp tải weights mỗi lượt |
| Generation warm vẫn gần 120s | Kiểm tra Windows Task Manager GPU memory/shared memory, device/dtype, OOM, số token; kiểm tra KV cache/attention của revision thật | Paired inference cùng model SHA và truth; CER không xấu đi ngoài ngưỡng chủ sản phẩm duyệt |
| Decode/preprocess chiếm phần lớn | Benchmark từng transform, loại thao tác thừa có cùng raster output | So pixel và CER; không tự tắt transform làm mất chữ |
| Cần giảm mạnh tổng thời gian với scan | Đánh giá engine OCR chuyên dụng nhanh, chỉ dùng VLM cho case khó | Cần xác nhận dependency/model mới; so CER theo Việt/Anh/song ngữ và latency trên RTX 2050 trước khi thay default |

Chưa thêm engine/model/runtime dependency; chưa bật quantization, đổi attention, giảm tiles/token cap hoặc thay ngưỡng. Không có cơ sở hứa 120s thành một số giây cụ thể cho scan trên RTX 2050 trong vòng này.
