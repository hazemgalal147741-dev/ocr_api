"""
Arabic OCR Studio — رفع ← OCR ← تصحيح ← نتيجة
"""

import streamlit as st
import io
from PIL import Image

from ocr_processor import OCRProcessor, PDF_IMAGE_SUPPORT, PYPDF_SUPPORT, DOCX_READ_SUPPORT
from ollama_corrector import check_ollama_status, correct_text_with_ollama

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Arabic OCR Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }

/* Header */
.ocr-header {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a3e 50%, #0d1b2a 100%);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 16px;
    padding: 28px 40px;
    margin-bottom: 28px;
    text-align: center;
}
.ocr-title { font-size: 2.2rem; font-weight: 900; color: #e2e8f0; margin: 0; }
.ocr-title span { color: #63b3ed; }
.ocr-sub { color: #718096; font-size: 0.9rem; margin-top: 6px; direction: rtl; }

/* Step cards */
.step-card {
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.step-label {
    font-size: 0.78rem;
    color: #63b3ed;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* Result box */
.result-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 22px 26px;
    font-family: 'Cairo', sans-serif;
    font-size: 1.1rem;
    line-height: 2.1;
    color: #c9d1d9;
    direction: rtl;
    text-align: right;
    white-space: pre-wrap;
    word-break: break-word;
    min-height: 160px;
    max-height: 600px;
    overflow-y: auto;
}
.result-box-ltr {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 22px 26px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    line-height: 1.9;
    color: #c9d1d9;
    white-space: pre-wrap;
    word-break: break-word;
    min-height: 160px;
    max-height: 600px;
    overflow-y: auto;
}

/* Stats */
.stat-row { display: flex; gap: 10px; margin-top: 12px; }
.stat-box {
    flex: 1;
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 10px 8px;
    text-align: center;
}
.stat-num  { font-size: 1.4rem; font-weight: 900; color: #63b3ed; font-family: 'JetBrains Mono', monospace; }
.stat-lbl  { color: #718096; font-size: 0.74rem; margin-top: 2px; }

/* Badges */
.badge-ok   { background:rgba(72,187,120,.15); color:#68d391; border:1px solid rgba(72,187,120,.3); border-radius:20px; padding:3px 12px; font-size:.82rem; font-weight:600; }
.badge-off  { background:rgba(252,129,74,.15);  color:#fc814a; border:1px solid rgba(252,129,74,.3); border-radius:20px; padding:3px 12px; font-size:.82rem; font-weight:600; }
.badge-lang { background:rgba(99,179,237,.15);  color:#63b3ed; border:1px solid rgba(99,179,237,.4); border-radius:20px; padding:2px 12px; font-size:.80rem; font-weight:700; }

/* Pipeline arrows */
.pipeline {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin: 18px 0;
    flex-wrap: wrap;
}
.pipe-step {
    background: #1a202c;
    border: 1px solid #4a5568;
    border-radius: 8px;
    padding: 7px 16px;
    font-size: 0.85rem;
    color: #a0aec0;
    font-weight: 600;
}
.pipe-step.active { border-color: #63b3ed; color: #63b3ed; }
.pipe-step.done   { border-color: #68d391; color: #68d391; }
.pipe-arrow { color: #4a5568; font-size: 1.1rem; }

section[data-testid="stSidebar"] { background:#111827; border-right:1px solid #1f2937; }
.stButton>button { font-family:'Cairo',sans-serif !important; font-weight:700 !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "raw_text": "",
    "corrected_text": "",
    "detected_lang": "",
    "stage": "idle",          # idle | ocr_done | correcting | done
    "file_name": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

processor = OCRProcessor()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ الإعدادات")
    st.markdown("---")

    st.markdown("**🌐 اللغة**")
    lang_choice = st.selectbox(
        "اللغة", ["🤖 كشف تلقائي", "🇸🇦 عربي", "🇬🇧 إنجليزي", "🇸🇦🇬🇧 عربي + إنجليزي"],
        index=0, label_visibility="collapsed"
    )
    lang_map = {
        "🤖 كشف تلقائي": None,
        "🇸🇦 عربي": "ara",
        "🇬🇧 إنجليزي": "eng",
        "🇸🇦🇬🇧 عربي + إنجليزي": "ara+eng",
    }
    forced_lang = lang_map[lang_choice]

    st.markdown("---")
    st.markdown("**🦙 Ollama**")
    ollama_ok, ollama_models = check_ollama_status()
    if ollama_ok:
        st.markdown(f'<span class="badge-ok">✓ متصل — {len(ollama_models)} نموذج</span>', unsafe_allow_html=True)
        selected_model = st.selectbox("النموذج", ollama_models, label_visibility="collapsed")
    else:
        st.markdown('<span class="badge-off">✗ Ollama غير مشغّل</span>', unsafe_allow_html=True)
        st.caption("شغّل Ollama:\n```\nollama serve\n```")
        selected_model = None

    st.markdown("---")
    st.markdown("**📦 المكتبات**")
    for name, ok in [
        ("pdf2image", PDF_IMAGE_SUPPORT),
        ("pypdf", PYPDF_SUPPORT),
        ("python-docx", DOCX_READ_SUPPORT),
    ]:
        icon = "✅" if ok else "❌"
        st.markdown(f"{icon} {name}")

    st.markdown("---")
    if st.button("🔄 بدء من جديد", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ocr-header">
  <div class="ocr-title">🔍 Arabic <span>OCR</span> Studio</div>
  <div class="ocr-sub">ارفع الملف ← استخرج النص ← تصحيح تلقائي ← النتيجة</div>
</div>
""", unsafe_allow_html=True)

# ─── Pipeline Status Bar ──────────────────────────────────────────────────────
stage = st.session_state.stage
s1 = "done" if stage in ("ocr_done","correcting","done") else ("active" if stage=="idle" else "")
s2 = "done" if stage in ("correcting","done") else ("active" if stage=="ocr_done" else "")
s3 = "done" if stage == "done" else ("active" if stage=="correcting" else "")

st.markdown(f"""
<div class="pipeline">
  <div class="pipe-step {s1}">📤 رفع الملف</div>
  <div class="pipe-arrow">›</div>
  <div class="pipe-step {s1}">🔍 استخراج OCR</div>
  <div class="pipe-arrow">›</div>
  <div class="pipe-step {s2}">✨ تصحيح Ollama</div>
  <div class="pipe-arrow">›</div>
  <div class="pipe-step {s3}">✅ النتيجة</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — رفع الملف
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("#### 📤 الخطوة الأولى — ارفع الملف")

uploaded = st.file_uploader(
    "اختر ملف",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "webp", "pdf", "docx"],
    label_visibility="collapsed",
    help="الصور: PNG, JPG, BMP, TIFF, WEBP | المستندات: PDF, DOCX",
)

if uploaded:
    ext = uploaded.name.split(".")[-1].lower()
    col_prev, col_info = st.columns([1, 1], gap="large")

    with col_prev:
        if ext in ("png", "jpg", "jpeg", "bmp", "tiff", "webp"):
            pil = Image.open(uploaded)
            st.image(pil, caption=uploaded.name, use_container_width=True)
        else:
            icon = "📄" if ext == "pdf" else "📝"
            size_kb = len(uploaded.getvalue()) // 1024
            st.markdown(f"""
            <div style="background:#1a202c;border:1px solid #2d3748;border-radius:12px;
                        padding:30px;text-align:center;margin-top:8px">
              <div style="font-size:3rem">{icon}</div>
              <div style="color:#63b3ed;font-weight:700;margin-top:8px">{uploaded.name}</div>
              <div style="color:#718096;font-size:0.85rem;margin-top:4px">{size_kb} KB</div>
            </div>""", unsafe_allow_html=True)

    with col_info:
        st.markdown("**معلومات الملف:**")
        st.write(f"- الاسم: `{uploaded.name}`")
        st.write(f"- النوع: `{ext.upper()}`")
        st.write(f"- الحجم: `{len(uploaded.getvalue())//1024} KB`")
        if forced_lang:
            st.write(f"- اللغة: `{forced_lang}`")
        else:
            st.write("- اللغة: كشف تلقائي 🤖")

    st.markdown("---")

    # ── OCR تلقائي فور رفع الملف ─────────────────────────────────────────────
    if st.session_state.stage == "idle" or st.session_state.file_name != uploaded.name:
        file_bytes = uploaded.getvalue()
        with st.spinner("⏳ جاري استخراج النص..."):
            try:
                if ext in ("png", "jpg", "jpeg", "bmp", "tiff", "webp"):
                    pil = Image.open(io.BytesIO(file_bytes))
                    text, lang = processor.extract_from_pil(pil, forced_lang)

                elif ext == "pdf":
                    prog_bar = st.progress(0)
                    prog_txt = st.empty()
                    def pdf_cb(c, t):
                        prog_bar.progress(int(c/t*100))
                        prog_txt.text(f"صفحة {c} من {t}...")
                    text, lang = processor.extract_from_pdf(file_bytes, forced_lang, pdf_cb)
                    prog_bar.empty(); prog_txt.empty()

                elif ext == "docx":
                    text, lang = processor.extract_from_docx(file_bytes)

                st.session_state.raw_text      = text
                st.session_state.corrected_text = text
                st.session_state.detected_lang  = lang
                st.session_state.stage          = "ocr_done"
                st.session_state.file_name      = uploaded.name
                st.rerun()

            except Exception as e:
                st.error(f"❌ خطأ في الاستخراج: {e}")

    # ── عرض نص الـ OCR لو اتعمل ─────────────────────────────────────────────
    if st.session_state.stage in ("ocr_done", "correcting", "done"):
        lang = st.session_state.detected_lang
        is_ara = "ara" in lang
        css = "result-box" if is_ara else "result-box-ltr"

        lang_labels = {"ara": "🇸🇦 عربي", "eng": "🇬🇧 إنجليزي", "ara+eng": "🇸🇦🇬🇧 عربي + إنجليزي"}
        st.markdown(
            f'<span class="badge-lang">{lang_labels.get(lang, lang)}</span> '
            f'<span style="color:#718096;font-size:.82rem">— نص OCR خام</span>',
            unsafe_allow_html=True,
        )
        txt_safe = st.session_state.raw_text.replace("<","&lt;").replace(">","&gt;")
        st.markdown(f'<div class="{css}">{txt_safe}</div>', unsafe_allow_html=True)

        # Stats
        w = len(st.session_state.raw_text.split())
        c = len(st.session_state.raw_text)
        l = st.session_state.raw_text.count("\n") + 1
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-box"><div class="stat-num">{w}</div><div class="stat-lbl">كلمة</div></div>
          <div class="stat-box"><div class="stat-num">{c}</div><div class="stat-lbl">حرف</div></div>
          <div class="stat-box"><div class="stat-num">{l}</div><div class="stat-lbl">سطر</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3 — تصحيح Ollama تلقائي
        # ═══════════════════════════════════════════════════════════════════════
        if st.session_state.stage == "ocr_done":
            if not ollama_ok:
                st.info("ℹ️ Ollama مش شغّال — تم عرض نص الـ OCR مباشرة.")
                st.session_state.stage = "done"
                st.rerun()
            else:
                with st.spinner("✨ Ollama يصحح النص تلقائياً..."):
                    st.session_state.stage = "correcting"
                    bar  = st.progress(0)
                    info = st.empty()
                    def cb(cur, total):
                        bar.progress(int(cur/total*100))
                        info.text(f"جزء {cur} من {total}...")
                    result = correct_text_with_ollama(
                        st.session_state.raw_text,
                        model_name=selected_model,
                        progress_callback=cb,
                    )
                    bar.empty(); info.empty()
                if result:
                    st.session_state.corrected_text = result
                    st.session_state.stage = "done"
                    st.rerun()
                else:
                    st.error("❌ فشل التصحيح — تحقق من Ollama")
                    st.session_state.corrected_text = st.session_state.raw_text
                    st.session_state.stage = "done"
                    st.rerun()

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4 — النتيجة النهائية
        # ═══════════════════════════════════════════════════════════════════════
        if st.session_state.stage == "done":
            st.markdown("---")
            st.markdown("#### ✅ النتيجة النهائية")

            final = st.session_state.corrected_text
            lang  = st.session_state.detected_lang
            is_ara = "ara" in lang
            css   = "result-box" if is_ara else "result-box-ltr"

            was_corrected = final != st.session_state.raw_text
            label = "✨ نص مصحح بـ Ollama" if was_corrected else "📄 نص OCR (بدون تصحيح)"
            st.markdown(f'<span class="badge-ok">{label}</span>', unsafe_allow_html=True)

            final_safe = final.replace("<","&lt;").replace(">","&gt;")
            st.markdown(f'<div class="{css}">{final_safe}</div>', unsafe_allow_html=True)

            st.markdown("")
            with st.expander("✏️ تعديل يدوي قبل التحميل"):
                edited = st.text_area(
                    "تعديل النص", value=final, height=220, label_visibility="collapsed",
                    key="manual_edit"
                )
                if st.button("💾 حفظ التعديل"):
                    st.session_state.corrected_text = edited
                    st.rerun()

            st.markdown("")
            dl_col1, dl_col2 = st.columns(2, gap="medium")
            base_name = st.session_state.file_name.rsplit(".", 1)[0]

            with dl_col1:
                st.download_button(
                    "⬇️ تحميل كـ TXT",
                    data=st.session_state.corrected_text.encode("utf-8"),
                    file_name=f"{base_name}_corrected.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with dl_col2:
                if st.button("📝 تحميل كـ DOCX", use_container_width=True):
                    try:
                        from docx import Document
                        from docx.shared import Pt
                        from docx.enum.text import WD_ALIGN_PARAGRAPH
                        doc = Document()
                        doc.styles["Normal"].font.name = "Arial"
                        doc.styles["Normal"].font.size = Pt(12)
                        align = WD_ALIGN_PARAGRAPH.RIGHT if is_ara else WD_ALIGN_PARAGRAPH.LEFT
                        for line in st.session_state.corrected_text.split("\n"):
                            p = doc.add_paragraph(line)
                            p.alignment = align
                        buf = io.BytesIO()
                        doc.save(buf); buf.seek(0)
                        st.download_button(
                            "⬇️ اضغط للتحميل",
                            data=buf,
                            file_name=f"{base_name}_corrected.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )
                    except ImportError:
                        st.error("ثبّت: pip install python-docx")

else:
    st.markdown("""
    <div style="background:#1a202c;border:2px dashed #2d3748;border-radius:16px;
                padding:50px;text-align:center;margin-top:10px">
      <div style="font-size:3rem;margin-bottom:12px">📂</div>
      <div style="color:#718096;font-size:1.05rem;direction:rtl">
        ارفع صورة أو PDF أو ملف Word من القائمة بالأعلى للبدء
      </div>
      <div style="color:#4a5568;font-size:0.85rem;margin-top:8px">
        PNG · JPG · BMP · TIFF · WEBP · PDF · DOCX
      </div>
    </div>
    """, unsafe_allow_html=True)
