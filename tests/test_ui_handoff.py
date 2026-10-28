"""Streamlit widget interactions with a fake game receiver and OCR engine."""
import io
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from ocr_module.pipeline import OCRResult
from test_pdf_io import make_one_page_pdf

TEXT = "This is valid extracted learning material for a game."
APP = '''
import io
from unittest.mock import patch
import streamlit as st
from ocr_module.ui_upload import main
def receive(text):
    if st.session_state.get('consumer_error'):
        raise RuntimeError('receiver unavailable')
    st.session_state['received'] = st.session_state.get('received', []) + [text]
files = []
for name, data in st.session_state.get('files', []):
    f = io.BytesIO(data)
    f.name = name
    files.append(f)
with patch('ocr_module.ui_upload.st.file_uploader', return_value=files):
    main(receive)
'''


class UIHandoffTests(unittest.TestCase):
    def app(self, pdf=False):
        if pdf:
            name, data = "study.pdf", make_one_page_pdf()
        else:
            stream = io.BytesIO()
            Image.new("RGB", (400, 400), "white").save(stream, format="PNG")
            name, data = "study.png", stream.getvalue()
        app = AppTest.from_string(APP, default_timeout=10)
        app.session_state['files'] = [(name, data)]
        app.run()
        self.assertEqual(len(app.exception), 0)
        return app

    def button(self, app, label):
        return next(button for button in app.button if button.label == label)

    def extract(self, app, result=None):
        with patch('ocr_module.ui_upload.run_pipeline', return_value=result or OCRResult(text=TEXT, ok=True)):
            label = 'Nhận dạng lại từ đầu' if result is not None and any(b.label == 'Nhận dạng lại từ đầu' for b in app.button) else 'Trích xuất văn bản'
            self.button(app, label).click().run()
        self.assertEqual(len(app.exception), 0)

    def test_image_and_pdf_deliver_to_real_callback(self):
        for pdf in (False, True):
            with self.subTest(pdf=pdf):
                app = self.app(pdf)
                self.extract(app)
                self.button(app, "Chuyển văn bản sang game").click().run()
                self.assertEqual(app.session_state['received'], [TEXT])
                self.assertEqual(app.session_state['game_raw_text'], TEXT)

    def test_edit_needs_acknowledgement_and_ack_resets_for_new_edit(self):
        app = self.app()
        self.extract(app)
        app.text_area[0].set_value(TEXT + " Corrected.").run()
        self.button(app, "Chuyển văn bản sang game").click().run()
        self.assertNotIn('received', app.session_state)
        next(c for c in app.checkbox if 'xác nhận' in c.label).check().run()
        app.text_area[0].set_value(TEXT + " Another edit.").run()
        self.button(app, "Chuyển văn bản sang game").click().run()
        self.assertNotIn('received', app.session_state)
        next(c for c in app.checkbox if 'xác nhận' in c.label).check().run()
        self.button(app, "Chuyển văn bản sang game").click().run()
        self.assertEqual(app.session_state['received'], [TEXT + " Another edit."])

    def test_page_range_rotation_and_config_invalidate_result(self):
        for pdf, change in [(True, 'range'), (False, 'rotation'), (False, 'config')]:
            with self.subTest(change=change):
                app = self.app(pdf)
                self.extract(app)
                self.button(app, "Chuyển văn bản sang game").click().run()
                if change == 'range':
                    next(n for n in app.number_input if n.label == 'Trang bắt đầu').set_value(2).run()
                elif change == 'rotation':
                    self.button(app, "Xoay phải").click().run()
                else:
                    app.slider[0].set_value(1700).run()
                self.assertEqual(len(app.text_area), 0)
                self.assertNotIn('game_raw_text', app.session_state)
                self.assertEqual(len(app.exception), 0)

    def test_new_extraction_replaces_old_editor_and_removal_clears_payload(self):
        app = self.app()
        self.extract(app)
        app.text_area[0].set_value(TEXT + " old edit").run()
        self.extract(app, OCRResult(text=TEXT + " NEW", ok=True))
        self.assertEqual(app.text_area[0].value, TEXT + " NEW")
        self.button(app, "Chuyển văn bản sang game").click().run()
        app.session_state['files'] = []
        app.run()
        self.assertNotIn('game_raw_text', app.session_state)
        self.assertEqual(len(app.text_area), 0)

    def test_empty_rejected_and_consumer_error(self):
        app = self.app(True)
        self.extract(app, OCRResult(text="", ok=False, reason="Rejected"))
        self.button(app, "Chuyển văn bản sang game").click().run()
        self.assertNotIn('received', app.session_state)
        self.assertNotIn('game_raw_text', app.session_state)

    def test_consumer_exception_is_shown_without_success_payload(self):
        app = self.app()
        self.extract(app)
        app.session_state['consumer_error'] = True
        self.button(app, "Chuyển văn bản sang game").click().run()
        self.assertNotIn('game_raw_text', app.session_state)
        self.assertNotIn('received', app.session_state)
        self.assertTrue(any('receiver unavailable' in error.value for error in app.error))

    def test_repeat_click_preserves_edits_and_skips_inference(self):
        app = self.app()
        self.extract(app)
        app.text_area[0].set_value(TEXT + ' edited').run()
        with patch('ocr_module.ui_upload.run_pipeline') as infer:
            self.button(app, 'Trích xuất văn bản').click().run()
            infer.assert_not_called()
        self.assertEqual(app.text_area[0].value, TEXT + ' edited')

    def test_pdf_text_mode_has_separate_context_and_requires_review(self):
        app = self.app(True)
        self.extract(app)
        next(s for s in app.selectbox if s.label == 'Cách đọc PDF').set_value('pdf_text').run()
        self.assertEqual(len(app.text_area), 0)
        with patch('ocr_module.ui_upload.read_pdf_text', return_value=OCRResult(text=TEXT, ok=True, metrics={'requires_review': True})), patch('ocr_module.ui_upload.run_pipeline') as infer:
            self.button(app, 'Trích xuất văn bản').click().run()
            infer.assert_not_called()
        self.button(app, 'Chuyển văn bản sang game').click().run()
        self.assertNotIn('received', app.session_state)
        next(c for c in app.checkbox if 'xác nhận' in c.label).check().run()
        self.button(app, 'Chuyển văn bản sang game').click().run()
        self.assertEqual(app.session_state['received'], [TEXT])

    def test_adding_or_reordering_files_preserves_existing_result_and_edit(self):
        app = self.app(True)
        self.extract(app)
        app.text_area[0].set_value(TEXT + ' saved edit').run()
        original = app.session_state['files']
        app.session_state['files'] = original + [('other.pdf', make_one_page_pdf())]
        app.run()
        self.assertEqual(app.text_area[0].value, TEXT + ' saved edit')
        app.session_state['files'] = list(reversed(app.session_state['files']))
        app.run()
        self.assertEqual(app.text_area[0].value, TEXT + ' saved edit')
        self.assertEqual(len(app.exception), 0)

    def test_config_change_keeps_edit_as_nontransferable_draft(self):
        app = self.app(True)
        self.extract(app)
        app.text_area[0].set_value(TEXT + ' important edit').run()
        app.slider[0].set_value(1700).run()
        drafts = [s.get('draft_text') for s in app.session_state['ocr_states'].values()]
        self.assertIn(TEXT + ' important edit', drafts)
        self.assertFalse(any(b.label == 'Chuyển văn bản sang game' for b in app.button))
