# Các vòng hoàn thiện OCR

## A — Core CPU và CI (28/10/2026, Asia/Bangkok)

| Trường | Nội dung |
|---|---|
| Task | Tách import engine và môi trường test khỏi ML/GPU |
| Nguyên nhân | Baseline tại e3eeaf8: discovery lỗi cả 3 module test do config import torch; workflow không cài torch |
| Files changed | config.py, pipeline.py, requirements*.txt, ci.yml, test_stage0_env.py, test_imports.py, README, coverage config |
| Tests | PASS: 7 tests, core smoke, compileall, pip check trên venv Python 3.12.14 Linux không torch |
| Regression | 6 test cũ + 1 import-isolation test pass |
| Performance | Chưa benchmark; inference chưa chạy |
| Coverage | 29% branch+statement toàn bộ OCR; bao gồm engine/UI chưa kiểm thử |
| Commit | Xem Git log với message fix(ci): isolate CPU contracts from inference dependencies |
| Push | Xác minh bằng remote SHA sau commit; không dùng báo cáo này làm bằng chứng push |
| Rủi ro còn lại | Chưa xác minh CI hosted; inference dependencies/model weights chưa cài |
| Tiếp theo | B: adapter nhận string, kiểm soát chất lượng và trạng thái UI/PDF |

## B — OCR → consumer

| Trường | Nội dung |
|---|---|
| Task | Adapter string và UI chung cho ảnh/PDF |
| Nguyên nhân | PDF kết thúc trước các nút xuất/chuyển; nút ảnh chỉ đặt session_state; editor Streamlit có thể khôi phục nội dung cũ |
| Files changed | handoff.py, ui_upload.py, test_handoff*.py, test_ui_handoff.py, OCR_INTERFACE.md |
| Tests | PASS: 18 tests; adapter 100% branch+statement; UI 92%; toàn bộ OCR 58% |
| Regression | 7 tests vòng A pass; AppTest ảnh/PDF gọi receiver thật nhận đúng string |
| Performance | Chưa đo inference; không thay thuật toán/model |
| Commit | Xem Git log: fix(ocr): deliver reviewed image and PDF text through consumer adapter |
| Push | Xác minh SHA từ remote sau commit |
| Rủi ro còn lại | Receiver nhóm game chưa có trong nhánh OCR; standalone vô hiệu nút chuyển |
| Tiếp theo | C/D: PDF streaming, lỗi từng trang, tài nguyên, OOM và kiểm thử sâu |

## C1 — IO và tài nguyên

| Trường | Nội dung |
|---|---|
| Task | Render PDF tuần tự; giữ chẩn đoán trang; input an toàn |
| Nguyên nhân | 5 regression mới tái hiện 4 failure + 1 exception: list bytes hỏng crash, item không hỗ trợ bị bỏ qua, mất lỗi trang đầu, cảnh báo EXIF sai, flag perspective không có tác dụng |
| Files changed | config.py, io_utils.py, pipeline.py, preprocess.py, ui_upload.py, test_resource_safety.py, test_pdf_streaming.py, OCR_INTERFACE.md |
| Tests | PASS: 28 tests; kiểm chứng render/inference xen kẽ, đóng PDFium handles, guard trước render, partial failure |
| Regression | Contract OCRResult giữ nguyên; 18 tests A/B pass |
| Performance | Bằng chứng cấu trúc: một raster/trang tại thời điểm inference; benchmark số liệu được ghi riêng ở vòng D |
| Commit | Xem Git log: fix(ocr): stream bounded PDF pages and retain page diagnostics |
| Push | Xác minh SHA từ remote sau commit |
| Rủi ro còn lại | Chưa có inference thực, chưa thay default preprocessing/gate/model; partial document cần review tay |
| Tiếp theo | Kiểm thử OOM/cache và báo cáo tài nguyên/coverage |

## C2/D — OOM, cache, CPU inference contract và coverage

| Trường | Nội dung |
|---|---|
| Task | Giải phóng tensor trước OOM retry; khóa cache/inference giữa sessions; bổ sung test sâu |
| Nguyên nhân | Regression weakref tái hiện tensor lỗi còn sống trước fallback; strip_target_lines thay đổi bị bỏ qua |
| Files changed | ocr_vintern.py, test_inference_mock.py, test_core_edges.py, ci.yml, tools/benchmark_pdf_resources.py |
| Tests | PASS: 42 tests, không skip, với torch CPU và model giả; full dependency smoke + pip check pass |
| Regression | 6 test gốc cũng pass trên cùng venv đầy đủ; 28 test A/B/C1 pass; normal generation/dtype không đổi |
| Performance | PDF 10 trang tổng hợp, render-only: eager 131200 KiB RSS/0.1065s, stream 68688 KiB/0.1323s; raster digest trùng; chưa chứng minh inference nhanh hơn |
| Coverage | 90% toàn OCR; engine 90%, adapter và image quality gate 100%; ngưỡng CI CPU mock >=75% |
| Commit | Xem Git log: fix(ocr): release OOM tensors and serialize model cache access |
| Push | Xác minh SHA từ remote sau commit |
| Rủi ro còn lại | Không GPU, không tải weights; OOM là mô phỏng, chưa có CER thật; revision mặc định chưa pin, có cảnh báo rõ |
| Tiếp theo | E: PRD, Mermaid, báo cáo kỹ thuật và demo trong phạm vi OCR |

CI hosted vòng C1 đã xác minh Success, 2 jobs: https://github.com/maitoan2408/EduGame/actions/runs/40551418917 (commit 76b8ba7). Không suy rộng trạng thái này cho commit mới.

## D/E — Evaluator, hồ sơ OCR và đo kiểm

| Trường | Nội dung |
|---|---|
| Task | Sửa lỗi dataset rỗng/thiếu truth; giữ chi tiết eval; thêm profiler và hồ sơ OCR |
| Nguyên nhân | test_eval tái hiện CER=0 trên dataset rỗng và thiếu chẩn đoán rejection; tài liệu PRD/Mermaid/interface chưa đầy đủ |
| Files changed | eval/run_eval.py, test_eval.py, test_profile.py, test_ui_handoff.py, tools/profile_ocr.py, README và docs OCR |
| Tests | PASS: 49 tests không skip, full smoke, compileall, pip check, diff check |
| Regression | Tất cả A–D pass; defaults/gates/normalization CER không đổi |
| Performance | Ba lần PDF render độc lập: median RSS 130064 → 68584 KiB, giảm 47.3%; digest giống nhau. Xem benchmark report |
| Coverage | 87% với phạm vi mở rộng gồm evaluator; adapter 100% |
| Commit | Xem Git log: feat(ocr): add auditable evaluation and OCR defense documentation |
| Push | Xác minh bằng remote SHA và Actions của commit này sau push |
| Rủi ro còn lại | Model thật/GPU/CER/consumer game thật chưa được cung cấp; model SHA chưa pin |
| Tiếp theo | Chủ phân hệ xác nhận model SHA, dữ liệu và máy demo; nhóm game nối callback và test tích hợp thật |

CI hosted C2/D: cả 3 jobs Success tại https://github.com/maitoan2408/EduGame/actions/runs/40552010114 (9636fd9).

## Latency — đo nút thắt, xem chữ sớm và bỏ tính toán lặp

| Trường | Nội dung |
|---|---|
| Task | Streaming preview; timing cold/warm; session reuse; PDF text layer opt-in |
| Nguyên nhân | Người dùng báo tổng gần 120s trên RTX2050, chưa có stage timings. Code UI chạy lại model/preprocess dù kết quả/context không đổi; model.chat chỉ trả khi sinh xong |
| Files changed | diagnostics.py, session_work.py, pdf_text.py, pipeline.py, ocr_vintern.py, ui_upload.py, handoff.py, profile_ocr.py; tests và docs liên quan |
| Tests | PASS 58 tests, smoke/compile; coverage 87%; exact token IDs của decoder Transformers local khi có/không streamer |
| Regression | Baseline 49 tests tiếp tục pass; giữ OCRResult, defaults model/prompt/tiles/dtype/token cap, quality gates |
| Performance | Chỉ đo được text-layer fixture ngắn: median 0.000287s/20 lượt; không có benchmark Vintern GPU hoặc CER thật |
| Commit | Xem Git log: perf(ocr): stream previews and reuse current results with stage diagnostics |
| Push | Xác nhận remote SHA/Actions sau commit, không suy từ bảng này |
| Rủi ro còn lại | Streaming không giảm tổng inference; chưa tái hiện 120s trên GPU thật. PDF text layer có thể thiếu chữ trong hình nên luôn review tay |
| Tiếp theo | Đo first/cold và warm bằng Nhận dạng lại từ đầu trên máy người dùng; kiểm tra generation/VRAM trước khi đổi thuật toán |
