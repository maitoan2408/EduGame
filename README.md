# Module OCR & Tiền xử lý ảnh tài liệu - Dự án EduGame

> **Phân hệ**: Tiền xử lý ảnh tài liệu học tập → Trích xuất văn bản qua Vintern-1B v3.5 → Giao diện Upload & Bàn giao dữ liệu cho bước sinh câu hỏi ôn tập bằng game.  
> **Thành viên phụ trách**: `maitoan2408`  
> **Nhánh phát triển**: `toanbranch`  
> **Dự án**: [EduGame](https://github.com/TAnh05/EduGame) (Nhóm 5 người, Python + Streamlit).

---

## 1. Tổng quan Kiến trúc

Hệ thống được thiết kế theo 4 nguyên tắc cốt lõi:
1. **Đo trước, xử lý sau**: Mọi bước tiền xử lý đều có cờ bật/tắt độc lập và đo lường CER (Character Error Rate) qua Ablation Study.
2. **Không làm hỏng ảnh**: Giữ ảnh màu/xám tự nhiên, tuyệt đối không nhị phân hóa cứng (Otsu/Binarization) gây mất nét chữ với Vision-Language Model.
3. **Cổng kiểm tra chất lượng 2 đầu (Quality Gates)**:
   - *Cổng đầu vào (Ảnh)*: Từ chối ảnh quá nhỏ hoặc quá mờ; độ sáng và tương phản hiện là cảnh báo, chưa hiệu chuẩn thành luật từ chối.
   - *Cổng đầu ra (Văn bản)*: Kiểm tra độ dài tối thiểu, tỷ lệ ký tự hợp lệ, phát hiện model loop (n-gram repetition) và tỷ lệ dấu tiếng Việt.
4. **Giao diện dữ liệu chuẩn (`OCRResult`)**: Đóng gói đầu ra độc lập với backend, giúp module sinh câu hỏi ôn tập dễ dàng tích hợp.

```
Đọc ảnh / PDF / HEIC 
   ↓
Sửa góc EXIF → Resize an toàn → Đo chất lượng ảnh
   ↓
Chỉnh nghiêng dòng (Deskew) → Cân bằng sáng (CLAHE + Background Division)
   ↓
Cắt dải ngang (Horizontal Histogram) nếu tài liệu dày chữ
   ↓
Mô hình Vintern-1B v3.5 (Tự động hạ max_num nếu OOM trên GPU 4GB)
   ↓
Hậu xử lý (Chuẩn hóa NFC, dọn markdown, khử lặp) → Cổng chất lượng văn bản
   ↓
Đầu ra chuẩn OCRResult → Giao diện Streamlit / Module tạo câu hỏi Game
```

---

## 2. Cấu trúc thư mục

```
ocr_module/
├── __init__.py
├── config.py          # Ngưỡng chất lượng, kích thước tối đa, tham số sinh & cờ ablation
├── io_utils.py        # Đọc ảnh, EXIF transpose, hỗ trợ HEIC (iPhone) & PDF scan đa trang
├── quality.py         # Đo độ mờ (Laplacian var), độ sáng, tương phản, độ phân giải
├── preprocess.py      # Chuỗi tiền xử lý: Resize, xoay, deskew, nắn sáng CLAHE, khử nhiễu
├── ocr_vintern.py     # Nạp Vintern-1B v3.5, load_image chuẩn, cắt dải ảnh, xử lý OOM fallback
├── postprocess.py     # Chuẩn hóa Unicode NFC, lọc markdown, khử lặp dòng/n-gram, cổng chất lượng
├── pipeline.py        # Điều phối toàn bộ luồng: run_pipeline(source) -> OCRResult
├── ui_upload.py       # Giao diện Web Streamlit hoàn chỉnh (so sánh ảnh, xoay 90°, sửa text)
└── eval/
    ├── images/        # Bộ ảnh kiểm thử mẫu (sạch, nghiêng, bóng đổ)
    ├── truth/         # Văn bản đáp án chuẩn (ground truth)
    └── run_eval.py    # Đo CER (Levenshtein distance) và chạy Ablation Study 5 cấu hình (A -> E)

run_single_test.py     # Script kiểm tra nhanh 1 ảnh từ dòng lệnh
test_stage0_env.py     # Script kiểm tra môi trường, GPU CUDA và logic module
requirements.txt       # Danh sách thư viện phụ thuộc
.gitignore             # Chặn file rác, file tạm, cache và model weights
```

---

## 3. Định dạng dữ liệu đầu ra (`OCRResult`)

```python
@dataclass
class OCRResult:
    text: str                          # Toàn bộ văn bản đã được làm sạch và chuẩn hóa NFC
    ok: bool                           # Vượt qua cả 2 cổng chất lượng (Ảnh & Văn bản) chưa
    reason: str                        # Lý do cụ thể nếu bị từ chối (để UI hướng dẫn người dùng chụp lại)
    warnings: list[str]                # Danh sách cảnh báo ("Ảnh hơi mờ", "Đã xoay 90°", v.v.)
    pages: list[str]                   # Danh sách văn bản theo từng trang (khi upload file PDF đa trang)
    metrics: dict                      # Các chỉ số đo đạc vật lý của ảnh (độ nét, sáng, tương phản)
    processed_preview: Optional[Image] # Ảnh sau khi qua chuỗi tiền xử lý để hiển thị so sánh
```

---

## 4. Hướng dẫn cài đặt & Chạy

### 4.1 Cài đặt môi trường

Python 3.12 được hỗ trợ. Kiểm thử CPU không cần torch/GPU và không tải model:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate ; Windows: .venv\Scripts\activate
python -m pip install -r requirements-test.txt
python test_stage0_env.py --core
python -m coverage run -m unittest discover -s tests -v
python -m coverage report
```

Để chạy OCR thực, cài thêm PyTorch/torchvision phù hợp phần cứng và các gói model bên dưới.
Chỉ cài requirements-core/test không đủ để inference.

Máy CPU: `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, sau đó cài `requirements.txt`.

1. **Cài PyTorch với GPU CUDA** (nếu máy có GPU NVIDIA):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
2. **Cài đặt các gói phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   ```

### 4.2 Chạy kiểm tra môi trường
```bash
python test_stage0_env.py
```

### 4.3 Chạy thử nghiệm trích xuất 1 ảnh (CLI)
```bash
# Thử nghiệm trên ảnh mẫu:
python run_single_test.py

# Hoặc thử nghiệm ảnh bất kỳ của bạn:
python run_single_test.py "duong_dan_anh.jpg"
```

### 4.4 Khởi động Giao diện Web Streamlit
```bash
streamlit run app.py
# hoặc:
streamlit run ocr_module/ui_upload.py
```
- Truy cập `http://localhost:8501`.
- Kéo thả ảnh hoặc PDF scan, so sánh ảnh gốc và ảnh đã xử lý trực quan.
- Tùy chỉnh xoay 90° nếu ảnh ngược, xem các chỉ số chất lượng ảnh.
- Nhận diện, chỉnh sửa và tải TXT. Để chuyển game, nhóm tích hợp callback nhận string theo [OCR_INTERFACE.md](docs/OCR_INTERFACE.md); standalone không giả lập giao nhận.

### 4.5 Đánh giá độ chính xác & Ablation Study
```bash
python -m ocr_module.eval.run_eval
```
Script sẽ so sánh 5 cấu hình tiền xử lý:
- **A**: Ảnh gốc (Baseline)
- **B**: A + EXIF + Resize
- **C**: B + Chỉnh nghiêng (Deskew)
- **D**: C + Cân bằng sáng & Khử bóng (CLAHE)
- **E**: D + Khử nhiễu & Làm nét
In ra bảng tổng hợp gồm CER trung bình, thời gian xử lý và tỷ lệ từ chối.

Tùy chọn `--images-dir`, `--truth-dir`, `--revision` và `--output` cho tập dữ liệu được phép dùng. Dataset rỗng/thiếu truth báo lỗi. Chưa tải weights hoặc xác minh CER thực trong đợt hoàn thiện này.

## 5. Hồ sơ kỹ thuật OCR

### Giảm thời gian chờ

Giữ beam 1 để xem chữ trong lúc Vintern nhận dạng. Preview chưa được kiểm tra; chỉ kết quả cuối mới có thể bàn giao.
PDF có thể copy chữ: chọn cách đọc **Đọc lớp chữ có sẵn**, kiểm tra các hình có chữ và xác nhận thủ công trước khi chuyển. PDF scan giữ **Nhận dạng ảnh bằng Vintern**.
Bấm lại Trích xuất dùng kết quả hiện tại; **Nhận dạng lại từ đầu** ép chạy inference mới. Mở **Chi tiết thời gian xử lý** để tách tải model, render, preprocess và generation.
Xem [thiết kế và cách đo trên RTX 2050](docs/experiments/OCR_LATENCY_DESIGN.md). Chưa có benchmark giảm tổng thời gian Vintern thực trên RTX 2050.

- [Đặc tả và nghiệm thu](docs/OCR_PRD.md)
- [Kiến trúc, Mermaid và quyết định](docs/OCR_ARCHITECTURE.md)
- [Interface tích hợp](docs/OCR_INTERFACE.md)
- [Defect log](docs/OCR_DEFECT_LOG.md)
- [Báo cáo kiểm thử](docs/OCR_TEST_REPORT.md)
- [Benchmark tài nguyên](docs/experiments/PDF_RESOURCE_BENCHMARK.md)
- [Các vòng triển khai](docs/ITERATION_REPORTS.md)
- [Demo và bảo vệ](docs/DEFENSE_GUIDE.md)
