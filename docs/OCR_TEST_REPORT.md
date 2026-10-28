# Test report — 28/10/2026

## Cập nhật sửa logic theo ưu tiên

**74 tests pass, không skip** trong profile đầy đủ CPU; coverage toàn OCR khoảng **88%**, hậu xử lý/quality gate/session cache/adapter 100%. Profile core không torch skip 15 tests ML/profiling, được chạy trong job inference-contract. Smoke đầy đủ, compileall, pip check và diff check đều đạt; CI xác minh đúng SHA sau push.

Regression mới bảo vệ phép nhân/lũy thừa/code, đáp án A/B/C, tab/cột và dòng lặp; chặn dấu câu-only và whitespace che emoji. AppTest giữ bản sửa khi thêm/sắp xếp file và giữ draft khi thay config. PDF 3 trang lỗi trang 2 vẫn thử trang 3 ở cả renderer và text-layer.

Ảnh chuẩn bị dùng chung preview/inference có test một lần preprocess và exact pixel; chữ ký bảo vệ cả góc, preprocess, quality, limits và strip settings. Cache eviction giữ text/draft và đóng cả ảnh vùng. Windows memory có test API thật trên CI Windows và test counter >4 GiB bằng API giả. Token budget/EOS và review gate có regression.

`constraints-tested.txt` chốt phiên bản dependency trực tiếp trong CI. Đây chưa phải lock transitive/driver hoặc benchmark inference GPU. Chia vùng trước resize là opt-in mặc định tắt, chỉ có test logic/pixel/assembly bằng OCR mock; chưa có CER thật để chọn default mới. Xem `experiments/LOGIC_REPAIR_REPORT.md` cho paired regression/benchmark và giới hạn.

Các phần dưới đây là lịch sử baseline trước đợt sửa logic.

## Cập nhật giảm thời gian chờ

Bản latency: **58 tests pass, không skip** trong môi trường đầy đủ; coverage toàn OCR **87%** (1184 statements, 368 branches), diagnostics/session reuse/adapter 100%. Core job không torch skip 13 tests ML/profiling, được chạy bắt buộc trong inference-contract.
Thêm kiểm chứng actual Transformers generate trên decoder nhỏ local: token IDs streaming/non-streaming bằng nhau; đây không phải chất lượng Vintern thật. PDFium text layer thật có fixture chuỗi khớp chính xác và không render/infer. AppTest kiểm tra cache/sửa tay/cách đọc/context. Chi tiết phạm vi và hạn chế ở `experiments/OCR_LATENCY_DESIGN.md`.

Các con số 49 tests dưới đây là baseline trước đợt latency, được giữ để truy vết.

## Kết quả xác minh cục bộ

- Baseline e3eeaf8 trong Python mặc định: 3 module discovery lỗi vì thiếu torch, chưa chạy được 6 test. Sau cài đầy đủ dependency, cả 6 test gốc pass trên cùng môi trường; phân biệt lỗi môi trường với lỗi logic.
- Bản hoàn thiện: **49 tests pass, không skip** trên Python 3.12.14 Linux, torch 2.14.1+cpu, torchvision 0.29.1+cpu, Transformers 5.21.0; không tải model weights và không gọi Hugging Face trong tests.
- Core Linux/Windows đã được GitHub Actions kiểm chứng; inference-contract CPU cũng xanh tại 9636fd9: [run 40552010114](https://github.com/maitoan2408/EduGame/actions/runs/40552010114). Commit cuối cần đối chiếu đúng SHA ở Actions, không coi CI của commit trước là bằng chứng cho code mới.
- Smoke full dependency, compileall và pip check pass. CUDA unavailable; OOM là mô phỏng trên CPU.
- Coverage branch+statement toàn `ocr_module`, gồm evaluator: **87%** (1006 statements, 320 branches); adapter 100%, image quality gate 100%, pipeline 89%, engine 90%, UI 89%. Con số 90% ở vòng trước chưa bao gồm evaluator; phạm vi coverage được mở rộng, không loại code để tăng điểm.
- Ba sơ đồ trong OCR_ARCHITECTURE.md đã parse thành công bằng Mermaid 11 (DOM giả lập phục vụ parser, không thêm dependency vào dự án). Sửa một dấu chấm phẩy trong message sequence để tránh Mermaid tách thành câu lệnh.

## Phạm vi tests

| Nhóm | Bằng chứng |
|---|---|
| Unit/regression | Quality ảnh rỗng/nhỏ/mờ/tối/cháy sáng/contrast, Unicode Việt/Anh, gate text, resize/rotation/flags, invalid inputs |
| Contract | Bảy trường OCRResult/defaults; text string; reject/rỗng/stale không giao; review tay; consumer exception |
| Integration | PDFium thật → pipeline OCR mock → receiver; PDF render/inference tuần tự và đóng handles |
| UI | Streamlit AppTest ảnh/PDF, editor, review, đổi cấu hình/góc/phạm vi, extraction mới, bỏ file, consumer error |
| Model mock | PyTorch CPU thật, model giả, tensor lifetime trước OOM retry, bounded retry, cache/revision, concurrent loader, generation/dtype |
| Evaluation/profiling | Dataset rỗng/thiếu truth báo lỗi; CER NFC/whitespace; per-image details; metadata không lộ văn bản; rejected output không được chấm như thành công |

## Tái lập

```bash
python -m venv .venv
# Kích hoạt venv theo hệ điều hành
python -m pip install -r requirements-test.txt
python test_stage0_env.py --core
python -m coverage run -m unittest discover -s tests -v
python -m coverage report
```

Profile core không cài torch sẽ skip 10 tests thuộc inference/profiling; các test này là bắt buộc trong job inference-contract riêng. Để chạy đủ 49 tests:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python test_stage0_env.py
python -m coverage run -m unittest discover -s tests -v
python -m coverage report --fail-under=75
python -m coverage xml
```

CI đặt HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE cho job model mock và xuất coverage XML artifact. Artifact không được commit. Dependency runtime không được thêm mới; coverage chỉ là công cụ test. Các phiên bản trên ghi lại môi trường đã đo; requirements vẫn có khoảng phiên bản, chưa phải lock đa nền tảng.

## Chưa chứng minh

Inference Vintern thật/cold start/warm latency; GPU VRAM/OOM thực; CER/false accept/false reject trên dữ liệu thật; tải nhiều người dùng; consumer game thật và demo toàn bộ EduGame; video bảo vệ. Cần weights/revision được review, GPU và dữ liệu có consent/ground truth. Các tham số inference/thuật toán OCR không được retune trong đợt này.

Không có DB/HTTP API trong phạm vi OCR nên không tạo OpenAPI/ERD giả. Bảo mật hiện được kiểm tra bằng review các đường upload/model/dependency và test giới hạn tài nguyên; chưa thực hiện audit OWASP toàn hệ thống hay quét lỗ hổng dependency đầy đủ.
