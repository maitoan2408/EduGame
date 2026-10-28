"""Streamlit OCR UI. Embedders inject a consumer accepting a plain string."""
import hashlib
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from ocr_module.config import OCRModuleConfig
from ocr_module.handoff import TextConsumer, deliver_text
from ocr_module.io_utils import (
    PDF_AVAILABLE, HEIF_AVAILABLE, load_image_from_bytes, pil_to_cv2, cv2_to_pil,
)
from ocr_module.pipeline import OCRResult, run_pipeline
from ocr_module.preprocess import run_preprocessing_pipeline
from ocr_module.diagnostics import capture
from ocr_module.pdf_text import read_pdf_text
from ocr_module.session_work import get_or_compute, upload_identities, prune_removed_files, release_images


def get_file_md5(data: bytes) -> str:
    """Content identity only; this digest is not used for security."""
    return hashlib.md5(data).hexdigest()


def invalidate_result(state: dict) -> None:
    result = state.get('result')
    if result is not None and state.get('user_text') and state['user_text'] != result.text:
        state['draft_text'] = state['user_text']
        state['draft_context'] = state.get('result_context', '')
    state.update(result=None, user_text="", result_context="")
    state["generation"] = state.get("generation", 0) + 1
    state.pop("processing_time", None)
    st.session_state.pop("game_raw_text", None)


def preview_processed_image(file_bytes, rotation_deg, cfg):
    original, _ = load_image_from_bytes(file_bytes, apply_exif=cfg.preprocess.enable_exif,
                                       max_pixels=cfg.limits.max_page_pixels)
    processed, ok, reason, warnings, metrics = run_preprocessing_pipeline(
        pil_to_cv2(original), cfg.preprocess, cfg.quality, rotation_deg,
    )
    return original, cv2_to_pil(processed), warnings, metrics, ok, reason


def render_sidebar_configs() -> OCRModuleConfig:
    cfg = OCRModuleConfig()
    st.sidebar.header("Cấu hình OCR")
    with st.sidebar.expander("Tiền xử lý ảnh"):
        for flag, label in [
            ("exif", "Sửa hướng EXIF"), ("resize", "Giới hạn kích thước"),
            ("deskew", "Chỉnh nghiêng"), ("illumination_fix", "Cân bằng sáng"),
            ("sharpen", "Làm nét nhẹ"), ("denoise", "Khử nhiễu nhẹ"),
        ]:
            name = "enable_" + flag
            setattr(cfg.preprocess, name, st.checkbox(label, value=getattr(cfg.preprocess, name)))
        cfg.preprocess.max_dimension = st.slider("Cạnh tối đa (px)", 1200, 3000, cfg.preprocess.max_dimension, step=100)
    with st.sidebar.expander("Vintern"):
        cfg.model.max_num = st.slider("Số ô ảnh tối đa", 1, 8, cfg.model.max_num)
        cfg.model.num_beams = st.selectbox("Số beam", [1, 3])
        cfg.model.repetition_penalty = st.slider("Phạt lặp từ", 1.0, 2.5, cfg.model.repetition_penalty, step=0.1)
        cfg.model.max_new_tokens = st.slider("Token tối đa", 256, 2048, cfg.model.max_new_tokens, step=128)
    with st.sidebar.expander("Ngưỡng chất lượng"):
        cfg.quality.blur_threshold = st.number_input("Ngưỡng mờ", min_value=0.0, value=cfg.quality.blur_threshold)
        cfg.quality.min_brightness = st.number_input("Độ sáng tối thiểu", min_value=0.0, max_value=255.0, value=cfg.quality.min_brightness)
        cfg.quality.max_brightness = st.number_input("Độ sáng tối đa", min_value=cfg.quality.min_brightness, max_value=255.0, value=max(cfg.quality.max_brightness, cfg.quality.min_brightness))
    return cfg


def render_result(state: dict, file_key: str, filename: str, cfg: OCRModuleConfig,
                  context: str, consumer: TextConsumer | None) -> None:
    """Shared by image and PDF; changing generation resets editor widget state."""
    result = state.get("result")
    if result is None:
        return
    if result.ok:
        st.success("Trích xuất hoàn tất. Hãy kiểm tra văn bản trước khi sử dụng.")
    else:
        st.error(result.reason)
    for warning in result.warnings:
        st.warning(warning)
    editor_key = f"text_{file_key}_{state.get('generation', 0)}"
    state["user_text"] = st.text_area(
        "Văn bản bàn giao (có thể chỉnh sửa)", value=state.get("user_text", result.text),
        key=editor_key, height=300,
        on_change=lambda: st.session_state.pop("game_raw_text", None),
    )
    text = state["user_text"]
    review_needed = not result.ok or text != result.text or result.metrics.get('requires_review', False)
    reviewed = False
    if review_needed:
        # An acknowledgement belongs to the exact editor contents, not a previous edit.
        digest = hashlib.sha256(text.encode()).hexdigest()
        reviewed = st.checkbox("Tôi đã kiểm tra và xác nhận nội dung này", key=f"review_{editor_key}_{digest}")
    st.download_button("Tải văn bản .txt", text, file_name=f"ocr_{filename.rsplit('.', 1)[0]}.txt", mime="text/plain", key=f"download_{file_key}")
    if consumer is None:
        st.info("Chưa kết nối game. Bạn có thể tải văn bản hoặc nhờ nhóm tích hợp nơi tiếp nhận.")
    if st.button("Chuyển văn bản sang game", key=f"send_{file_key}", disabled=consumer is None):
        st.session_state.pop("game_raw_text", None)
        try:
            deliver_text(result=result, text=text, consumer=consumer, gate=cfg.gate,
                         reviewed=reviewed, result_context=state.get("result_context", ""), current_context=context)
        except Exception as exc:
            st.error(f"Chưa chuyển văn bản: {exc}")
        else:
            st.session_state["game_raw_text"] = text  # Compatibility, only AFTER receipt.
            st.success("Nơi tiếp nhận đã nhận văn bản.")


def main(text_consumer: TextConsumer | None = None):
    st.set_page_config(page_title="OCR Tài liệu Học tập", page_icon="📖", layout="wide")
    st.title("OCR tài liệu ôn tập")
    cfg = render_sidebar_configs()
    show_partial = st.sidebar.checkbox('Hiện chữ trong lúc nhận dạng', value=True,
                                       help='Bản xem trước chưa qua kiểm tra, chỉ có với beam 1. Không giảm tổng thời gian suy luận.')
    if show_partial and cfg.model.num_beams != 1:
        st.sidebar.info('Beam 3 chỉ hiển thị kết quả khi hoàn tất. Chọn beam 1 để xem chữ sớm.')
    allowed = ["jpg", "jpeg", "png", "webp"]
    if PDF_AVAILABLE:
        allowed.append("pdf")
    if HEIF_AVAILABLE:
        allowed.extend(["heic", "heif"])
    uploaded = st.file_uploader("Tải ảnh hoặc PDF", type=allowed, accept_multiple_files=True)
    identities = upload_identities(uploaded or [])
    upload_context = (identities, repr(cfg))
    if st.session_state.get("upload_context") != upload_context:
        st.session_state.pop("game_raw_text", None)
        st.session_state["upload_context"] = upload_context
    if 'ocr_states' not in st.session_state:
        st.session_state['ocr_states'] = {}
    active_keys = {hashlib.sha256(identity.encode()).hexdigest() for identity in identities}
    prune_removed_files(st.session_state.ocr_states, active_keys)
    if not uploaded:
        st.info("Tải tài liệu để bắt đầu.")
        return
    if len(uploaded) > cfg.limits.max_pages:
        st.error(f"Chọn tối đa {cfg.limits.max_pages} tập tin mỗi lượt.")
        return
    for identity, file in zip(identities, uploaded):
        file_key = hashlib.sha256(identity.encode()).hexdigest()
        data = file.getvalue()
        state = st.session_state.ocr_states.setdefault(file_key, {"rot": 0, "generation": 0})
        st.subheader(file.name)
        if len(data) > cfg.limits.max_file_bytes:
            invalidate_result(state)
            st.error("Tập tin vượt giới hạn 20 MiB. Hãy chia nhỏ tài liệu.")
            continue
        is_pdf = file.name.lower().endswith(".pdf")
        start, count, method = 0, None, 'vintern'
        if is_pdf:
            start = int(st.number_input("Trang bắt đầu", min_value=1, value=1, step=1, key=f"start_{file_key}")) - 1
            count = int(st.number_input("Số trang", min_value=1, max_value=50, value=1, step=1, key=f"count_{file_key}"))
            method = st.selectbox('Cách đọc PDF', ['vintern', 'pdf_text'],
                                  format_func=lambda v: 'Nhận dạng ảnh bằng Vintern' if v == 'vintern' else 'Đọc lớp chữ có sẵn (nhanh, cần kiểm tra)',
                                  key=f'method_{file_key}')
        else:
            for column, (label, delta) in zip(st.columns(3), [("Xoay trái", -90), ("Xoay phải", 90), ("Đặt lại", 0)]):
                if column.button(label, key=f"{label}_{file_key}"):
                    state["rot"] = (state["rot"] + delta) % 360 if delta else 0
                    invalidate_result(state)
            try:
                preview_key = (identity, state['rot'], repr(cfg.preprocess), repr(cfg.quality), repr(cfg.limits))
                if state.get('preview_key') != preview_key:
                    release_images(state)
                    state['preview'] = preview_processed_image(data, state['rot'], cfg)
                    state['preview_key'] = preview_key
                original, preview, warnings, metrics, ok, reason = state['preview']
            except (ValueError, OSError, RuntimeError) as exc:
                invalidate_result(state)
                st.error(f"Không thể đọc ảnh: {exc}")
                continue
            left, right = st.columns(2)
            left.image(original, caption="Ảnh gốc")
            right.image(preview, caption="Sau tiền xử lý")
            for warning in warnings:
                st.warning(warning)
            if not ok:
                st.error(reason)
            st.caption(f"Độ nét: {metrics['blur']:.1f}; độ sáng: {metrics['brightness']:.1f}")
        context = repr((identity, repr(cfg), state["rot"], start, count, method))
        if state.get("context") != context:
            invalidate_result(state)
            state["context"] = context
        if state.get('draft_text'):
            st.info('Bản sửa trước đã được giữ làm bản nháp. Bản nháp không được tự động chuyển sang game.')
            st.download_button('Tải bản nháp trước', state['draft_text'],
                               file_name='ocr_draft.txt', mime='text/plain', key=f'draft_{file_key}')
        requested = st.button("Trích xuất văn bản", key=f"ocr_{file_key}")
        force = st.button('Nhận dạng lại từ đầu', key=f'retry_{file_key}') if state.get('result') is not None else False
        if requested or force:
            stage_box, partial_box = st.empty(), st.empty()
            def progress(entry):
                labels = dict(preprocess='Đang tiền xử lý ảnh', model_load_or_cache='Đang chuẩn bị model (lần đầu có thể lâu)',
                              generation='Đang nhận dạng chữ', postprocess='Đang kiểm tra văn bản',
                              pdf_read_render='Đang đọc trang PDF', tensor_transfer='Đang chuẩn bị dữ liệu GPU',
                              oom_retry='Thiếu VRAM, đang thử lại với ít ô ảnh hơn')
                if entry['stage'] in labels:
                    stage_box.info(labels[entry['stage']])
                if entry['stage'] == 'model_cache' and not entry['hit']:
                    stage_box.info('Đang nạp model. Giữ ứng dụng mở để các lần sau dùng lại model.')
            def partial(text):
                partial_box.code('Bản xem trước dải ảnh hiện tại — chưa kiểm tra:\n' + text, language=None)
            before = time.perf_counter()
            def compute():
                with capture(progress, partial if show_partial else None) as diagnostic:
                    if method == 'pdf_text':
                        result = read_pdf_text(data, cfg, start, count)
                    else:
                        result = run_pipeline(data, cfg, manual_rotation_deg=state['rot'], pdf_start_page=start, pdf_max_pages=count)
                    result.metrics['timings'] = dict(diagnostic.seconds)
                    result.metrics['execution'] = diagnostic.events
                    return result
            with st.spinner('Đang xử lý tài liệu...'):
                result, reused = get_or_compute(state, context, compute, force)
            if not reused:
                invalidate_result(state)
                state.update(result=result, user_text=result.text, result_context=context,
                             processing_time=time.perf_counter() - before)
            else:
                st.info('Đã dùng lại kết quả hiện tại và giữ bản bạn sửa. Chọn Nhận dạng lại từ đầu nếu cần chạy model lại.')
            state['last_request_seconds'] = time.perf_counter() - before
            state['reused'] = reused
            stage_box.empty()
            partial_box.empty()
        if "processing_time" in state:
            if state.get('reused'):
                st.caption(f"Lấy lại kết quả: {state['last_request_seconds']:.3f}s; lần nhận dạng gốc: {state['processing_time']:.2f}s")
            else:
                st.caption(f"Thời gian xử lý: {state['processing_time']:.2f}s")
            with st.expander('Chi tiết thời gian xử lý'):
                metrics = state.get('result').metrics
                st.json(dict(total_seconds=state['processing_time'], stages=metrics.get('timings', {}),
                             execution=metrics.get('execution', [])))
        render_result(state, file_key, file.name, cfg, context, text_consumer)


if __name__ == "__main__":
    main()
