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
