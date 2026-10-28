# Sửa logic theo ưu tiên — 28/10/2026

## 1. Bảo toàn văn bản và quality gate

Baseline efcab5b: 58 tests pass nhưng regression mới tái hiện 11 failure/subcase: mất phép nhân/lũy thừa/code, dòng lặp/đáp án/cột; dấu câu hoặc whitespace che emoji qua gate.
Sau sửa: 61 tests pass. Không đổi ngưỡng số hiện có; tỷ lệ hợp lệ tính trên cùng tập ký tự không-whitespace và yêu cầu có ít nhất chữ/số. Không dùng whitespace để bù ký tự rác. Dấu tiếng Việt vẫn warning, không reject ngôn ngữ.

So sánh hậu xử lý cùng chuỗi/ground truth tổng hợp, dùng calculate_cer của repo:

| Fixture | CER trước | CER sau | Giữ nguyên chuỗi trước/sau |
|---|---:|---:|---|
| T1 phép nhân | 0.1538 | 0 | Không / Có |
| T2 lũy thừa và __init__ | 0.2963 | 0 | Không / Có |
| T3 #include | 0.0233 | 0 | Không / Có |
| T4 A/B/C | 0 | 0 | Không / Có |
| T5 cột/tab | 0 | 0 | Không / Có |
| T6 đáp án lặp cách đoạn | 0.5263 | 0 | Không / Có |
| T7 dòng lặp liền nhau | 0.5152 | 0 | Không / Có |

CER chuẩn hóa whitespace không phát hiện mất layout ở T4/T5, nên test thêm exact string. Đây là regression hậu xử lý, không phải CER inference trên tài liệu thật hoặc bằng chứng accuracy production. Gate có 4 nhóm negative và 3 positive bổ sung; cần tiếp tục hiệu chuẩn trên dữ liệu thật.

## 2. Trạng thái và lỗi từng trang

ID file không còn phụ thuộc vị trí trong danh sách upload. Thêm/sắp xếp file giữ kết quả và bản sửa của file cũ; chỉ file bị bỏ mới giải phóng state/ảnh. Khi đổi config/góc/phạm vi, kết quả không còn được bàn giao, nhưng bản sửa được giữ thành một draft trong session để tải về (không tự khôi phục vào kết quả mới).

PDF renderer tạo record thành công/lỗi theo trang; pipeline giữ chuỗi rỗng tại vị trí lỗi và tiếp tục trang sau, ok tổng vẫn False. Iterator ảnh/list API cũ giữ cơ chế raise khi gặp lỗi để không âm thầm bỏ trang. Đường đọc lớp chữ có cùng hành vi hồi phục. Không cố tiếp tục nếu không mở được cả document hoặc phạm vi không hợp lệ.

AppTest và fixture 3 trang xác nhận lỗi trang 2 không ngăn trang 3; tất cả handles được đóng. Regression toàn bộ pass sau sửa; không có thay đổi thuật toán inference trong vòng này.

## 3. Bỏ tính toán trùng và đo tài nguyên

`PreparedImage` mang ảnh/gate/metrics cùng chữ ký preprocess, quality, resource limits và góc xoay. UI và pipeline dùng chung ảnh này; nếu chữ ký/góc thay đổi, ảnh cũ không được inference. Bảy trường OCRResult giữ nguyên. Bản hiển thị gốc được thumbnail sau khi chuẩn bị ảnh OCR; không hạ độ phân giải ảnh đưa vào model vì nhu cầu hiển thị.

Giới hạn bổ sung: 128 MiB tổng upload/lượt và 96 MiB raster cache/session. Eviction bỏ ảnh, giữ text/draft; không phải giới hạn RAM toàn process hoặc model/VRAM. Không ghi upload ra đĩa. Windows profiler dùng Win32 API chuẩn, không thêm runtime dependency.

Đo tuần tự trên Linux/Python 3.12.14, mỗi mẫu 5 lượt; baseline 853a205 và candidate hiện tại, model giả trả cùng một chuỗi. Đo đường preview + pipeline, không gồm tải model/inference thật:

| Mẫu synthetic có sẵn | Median trước (s) | Median sau (s) | Số lần illumination trước/sau |
|---|---:|---:|---|
| clean | 0.287731 | 0.143589 | 2 / 1 |
| shadow | 0.288227 | 0.143306 | 2 / 1 |
| skewed | 0.302528 | 0.153925 | 2 / 1 |

Digest pixel OCR và text model giả trùng nhau trước/sau cho cả ba mẫu. Đây là giảm công việc tiền xử lý, không phải bằng chứng 120 giây Vintern trên RTX2050 đã giảm một nửa; không kết luận accuracy từ ba ảnh tổng hợp.

## 4. Token budget, revision và chia vùng

- Ghi actual_tiles (kể cả thumbnail), model_lock_wait, output_tokens/tokens_per_second và model revision resolved nếu thư viện cung cấp. Khi streamer chạy beam 1, đếm token từ callback; khi không stream thì retokenize response và ghi estimated=True. Đây không phải finish_reason chính thức của model.chat.
- Nếu có khả năng chạm token cap mà không quan sát EOS, cảnh báo và đặt requires_review=True. Gate vẫn kiểm tra nội dung; không giảm token cap hoặc bỏ chữ để lấy tốc độ.
- Có ô revision SHA tùy chọn. Chưa tự chọn/pin model SHA khi chưa review được weights/mã remote. `constraints-tested.txt` pin dependency trực tiếp đã chạy CI, không phải lock mọi dependency bắc cầu hoặc driver CUDA.
- Chia dải trước resize là opt-in `enable_page_regions=False` mặc định. Dùng cùng thuật toán histogram/overlap hiện có, trên nguồn đã xoay đúng; từng vùng được preprocess và qua gate riêng. Dòng ghép vẫn cần xác nhận tay, lỗi vùng không được báo thành trang hoàn chỉnh. Default cũ giữ nguyên và cảnh báo khi resize khiến ngưỡng 2400 không thể đạt.
- Test source 1200×3000 xác nhận nhiều vùng thực sự được xử lý; ảnh preview toàn trang trước/sau bằng pixel; xoay tay chỉ áp dụng một lần; cache eviction đóng cả ảnh con. Đây là kiểm chứng logic/cờ, chưa có CER/latency thật để bật mặc định.

## Chưa thể chốt bằng Cloud CPU

CER và latency Vintern trên RTX2050; ngưỡng quality/repetition theo dữ liệu thật; default chia vùng tối ưu; KV cache/attention/quantization; engine OCR chuyên dụng thay thế. Chưa thay model, prompt, dtype, tile count, generation defaults hoặc thêm engine/dependency mới. Consumer game thật vẫn do nhóm game nối vào callback.
