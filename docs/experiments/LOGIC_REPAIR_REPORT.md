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
