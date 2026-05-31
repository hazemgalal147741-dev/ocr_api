"""
Arabic OCR Studio — رفع ← OCR ← تصحيح ← تحليل NAQAAE ← نتيجة
"""

import streamlit as st
import io
from PIL import Image

from ocr_processor import OCRProcessor, PDF_IMAGE_SUPPORT, PYPDF_SUPPORT, DOCX_READ_SUPPORT
from ollama_corrector import check_ollama_status, correct_text_with_ollama
from naqaae_client import analyze_with_naqaae, check_space_status

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

.naqaae-section {
    background: #0d1117;
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 24px;
}
.naqaae-title {
    font-size: 1.3rem;
    font-weight: 900;
    color: #e2e8f0;
    direction: rtl;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(99,179,237,0.2);
}
.naqaae-title span { color: #63b3ed; }

.status-card-ok {
    background: rgba(72,187,120,0.12);
    border: 1.5px solid rgba(72,187,120,0.4);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.status-card-fail {
    background: rgba(252,129,74,0.12);
    border: 1.5px solid rgba(252,129,74,0.4);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.status-label { font-size: 1.7rem; font-weight: 900; margin: 0; }
.status-label-ok   { color: #68d391; }
.status-label-fail { color: #fc814a; }
.status-sub { color: #718096; font-size: 0.85rem; margin-top: 4px; }

.score-num {
    font-size: 2.2rem;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    text-align: center;
    margin-bottom: 8px;
}
.score-bar-bg {
    background: #1a202c;
    border-radius: 999px;
    height: 14px;
    overflow: hidden;
    border: 1px solid #2d3748;
}
.score-bar-fill { height: 100%; border-radius: 999px; }
.score-green  { color: #68d391; background: linear-gradient(90deg,#276749,#68d391); }
.score-orange { color: #fc814a; background: linear-gradient(90deg,#7b341e,#fc814a); }
.score-red    { color: #fc8181; background: linear-gradient(90deg,#63171b,#fc8181); }

.recs-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px 24px;
    font-family: 'Cairo', sans-serif;
    font-size: 1rem;
    line-height: 1.9;
    color: #c9d1d9;
    direction: rtl;
    text-align: right;
    white-space: pre-wrap;
    word-break: break-word;
    min-height: 80px;
    max-height: 400px;
    overflow-y: auto;
    margin-top: 16px;
}

.stat-row { display: flex; gap: 10px; margin-top: 12px; }
.stat-box {
    flex: 1; background: #1a202c; border: 1px solid #2d3748;
    border-radius: 8px; padding: 10px 8px; text-align: center;
}
.stat-num { font-size: 1.4rem; font-weight: 900; color: #63b3ed; font-family: 'JetBrains Mono', monospace; }
.stat-lbl { color: #718096; font-size: 0.74rem; margin-top: 2px; }

.badge-ok   { background:rgba(72,187,120,.15); color:#68d391; border:1px solid rgba(72,187,120,.3); border-radius:20px; padding:3px 12px; font-size:.82rem; font-weight:600; }
.badge-off  { background:rgba(252,129,74,.15);  color:#fc814a; border:1px solid rgba(252,129,74,.3); border-radius:20px; padding:3px 12px; font-size:.82rem; font-weight:600; }
.badge-lang { background:rgba(99,179,237,.15);  color:#63b3ed; border:1px solid rgba(99,179,237,.4); border-radius:20px; padding:2px 12px; font-size:.80rem; font-weight:700; }

.pipeline { display:flex; align-items:center; justify-content:center; gap:8px; margin:18px 0; flex-wrap:wrap; }
.pipe-step { background:#1a202c; border:1px solid #4a5568; border-radius:8px; padding:7px 16px; font-size:0.85rem; color:#a0aec0; font-weight:600; }
.pipe-step.active { border-color:#63b3ed; color:#63b3ed; }
.pipe-step.done   { border-color:#68d391; color:#68d391; }
.pipe-arrow { color:#4a5568; font-size:1.1rem; }

section[data-testid="stSidebar"] { background:#111827; border-right:1px solid #1f2937; }
.stButton>button { font-family:'Cairo',sans-serif !important; font-weight:700 !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "raw_text": "",
    "corrected_text": "",
    "detected_lang": "",
    "stage": "idle",
    "file_name": "",
    "naqaae_result": None,
    "naqaae_done": False,
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
    st.markdown("**🎓 NAQAAE Space**")
    space_ok = check_space_status()
    if space_ok:
        st.markdown('<span class="badge-ok">✓ Space شغّال</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-off">⚠️ Space مش متاح</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🦙 Groq**")
    ollama_ok, ollama_models = check_ollama_status()
    if ollama_ok:
        st.markdown(f'<span class="badge-ok">✓ متصل — {len(ollama_models)} نموذج</span>', unsafe_allow_html=True)
        selected_model = st.selectbox("النموذج", ollama_models, label_visibility="collapsed")
    else:
        st.markdown('<span class="badge-off">✗ Groq غير متصل</span>', unsafe_allow_html=True)
        selected_model = None

    st.markdown("---")
    st.markdown("**📦 المكتبات**")
    for name, ok in [
        ("pdf2image", PDF_IMAGE_SUPPORT),
        ("pypdf", PYPDF_SUPPORT),
        ("python-docx", DOCX_READ_SUPPORT),
    ]:
        st.markdown(f"{'✅' if ok else '❌'} {name}")

    st.markdown("---")
    if st.button("🔄 بدء من جديد", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ocr-header">
  <div class="ocr-title">🔍 Arabic <span>OCR</span> Studio × <span>NAQAAE</span></div>
  <div class="ocr-sub">ارفع الملف ← استخرج النص ← تصحيح ← تحليل الاعتماد المؤسسي</div>
</div>
""", unsafe_allow_html=True)

# ─── Pipeline ─────────────────────────────────────────────────────────────────
stage = st.session_state.stage
s1 = "done" if stage in ("ocr_done","correcting","done") else ("active" if stage=="idle" else "")
s2 = "done" if stage in ("correcting","done") else ("active" if stage=="ocr_done" else "")
s3 = "done" if stage == "done" else ("active" if stage=="correcting" else "")
s4 = "done" if st.session_state.naqaae_done else ""

st.markdown(f"""
<div class="pipeline">
  <div class="pipe-step {s1}">📤 رفع</div><div class="pipe-arrow">›</div>
  <div class="pipe-step {s1}">🔍 OCR</div><div class="pipe-arrow">›</div>
  <div class="pipe-step {s2}">✨ تصحيح</div><div class="pipe-arrow">›</div>
  <div class="pipe-step {s4}">🎓 NAQAAE</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — رفع الملف
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("#### 📤 ارفع الملف")

uploaded = st.file_uploader(
    "اختر ملف",
    type=["png","jpg","jpeg","bmp","tiff","webp","pdf","docx"],
    label_visibility="collapsed",
)

if uploaded:
    ext = uploaded.name.split(".")[-1].lower()
    col_prev, col_info = st.columns([1,1], gap="large")

    with col_prev:
        if ext in ("png","jpg","jpeg","bmp","tiff","webp"):
            st.image(Image.open(uploaded), caption=uploaded.name, use_container_width=True)
        else:
            icon = "📄" if ext=="pdf" else "📝"
            size_kb = len(uploaded.getvalue())//1024
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
        st.write(f"- اللغة: `{forced_lang}`" if forced_lang else "- اللغة: كشف تلقائي 🤖")

    st.markdown("---")

    # ── OCR ───────────────────────────────────────────────────────────────────
    if st.session_state.stage == "idle" or st.session_state.file_name != uploaded.name:
        file_bytes = uploaded.getvalue()
        with st.spinner("⏳ جاري استخراج النص..."):
            try:
                if ext in ("png","jpg","jpeg","bmp","tiff","webp"):
                    pil = Image.open(io.BytesIO(file_bytes))
                    text, lang = processor.extract_from_pil(pil, forced_lang)
                elif ext == "pdf":
                    prog_bar = st.progress(0)
                    prog_txt = st.empty()
                    def pdf_cb(c,t):
                        prog_bar.progress(int(c/t*100))
                        prog_txt.text(f"صفحة {c} من {t}...")
                    text, lang = processor.extract_from_pdf(file_bytes, forced_lang, pdf_cb)
                    prog_bar.empty(); prog_txt.empty()
                elif ext == "docx":
                    text, lang = processor.extract_from_docx(file_bytes)

                st.session_state.raw_text       = text
                st.session_state.corrected_text  = text
                st.session_state.detected_lang   = lang
                st.session_state.stage           = "ocr_done"
                st.session_state.file_name       = uploaded.name
                st.session_state.naqaae_result   = None
                st.session_state.naqaae_done     = False
                st.rerun()
            except Exception as e:
                st.error(f"❌ خطأ في الاستخراج: {e}")

    # ── عرض نص OCR ────────────────────────────────────────────────────────────
    if st.session_state.stage in ("ocr_done","correcting","done"):
        lang   = st.session_state.detected_lang
        is_ara = "ara" in lang
        css    = "result-box" if is_ara else "result-box-ltr"
        lang_labels = {"ara":"🇸🇦 عربي","eng":"🇬🇧 إنجليزي","ara+eng":"🇸🇦🇬🇧 عربي + إنجليزي"}

        st.markdown(
            f'<span class="badge-lang">{lang_labels.get(lang,lang)}</span> '
            f'<span style="color:#718096;font-size:.82rem">— نص OCR خام</span>',
            unsafe_allow_html=True,
        )
        txt_safe = st.session_state.raw_text.replace("<","&lt;").replace(">","&gt;")
        st.markdown(f'<div class="{css}">{txt_safe}</div>', unsafe_allow_html=True)

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

        # ── Groq correction ───────────────────────────────────────────────────
        if st.session_state.stage == "ocr_done":
            if not ollama_ok:
                st.info("ℹ️ Groq مش متصل — تم استخدام نص الـ OCR مباشرة.")
                st.session_state.stage = "done"
                st.rerun()
            else:
                with st.spinner("✨ جاري تصحيح النص..."):
                    st.session_state.stage = "correcting"
                    bar  = st.progress(0)
                    info = st.empty()
                    def cb(cur,total):
                        bar.progress(int(cur/total*100))
                        info.text(f"جزء {cur} من {total}...")
                    result = correct_text_with_ollama(
                        st.session_state.raw_text,
                        model_name=selected_model,
                        progress_callback=cb,
                    )
                    bar.empty(); info.empty()
                st.session_state.corrected_text = result if result else st.session_state.raw_text
                st.session_state.stage = "done"
                st.rerun()

        # ── النتيجة النهائية ──────────────────────────────────────────────────
        if st.session_state.stage == "done":
            st.markdown("---")
            st.markdown("#### ✅ النص النهائي")

            final  = st.session_state.corrected_text
            is_ara = "ara" in st.session_state.detected_lang
            css    = "result-box" if is_ara else "result-box-ltr"
            was_corrected = final != st.session_state.raw_text

            st.markdown(
                f'<span class="badge-ok">{"✨ نص مصحح" if was_corrected else "📄 نص OCR"}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="{css}">{final.replace("<","&lt;").replace(">","&gt;")}</div>',
                        unsafe_allow_html=True)

            with st.expander("✏️ تعديل يدوي"):
                edited = st.text_area(
                    "تعديل النص", value=final, height=220,
                    label_visibility="collapsed", key="manual_edit"
                )
                if st.button("💾 حفظ التعديل"):
                    st.session_state.corrected_text = edited
                    st.session_state.naqaae_result  = None
                    st.session_state.naqaae_done    = False
                    st.rerun()

            dl1, dl2 = st.columns(2, gap="medium")
            base_name = st.session_state.file_name.rsplit(".",1)[0]

            with dl1:
                st.download_button(
                    "⬇️ تحميل TXT",
                    data=st.session_state.corrected_text.encode("utf-8"),
                    file_name=f"{base_name}_corrected.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dl2:
                if st.button("📝 تحميل DOCX", use_container_width=True):
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
                            "⬇️ اضغط للتحميل", data=buf,
                            file_name=f"{base_name}_corrected.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )
                    except ImportError:
                        st.error("ثبّت: pip install python-docx")

            # ═══════════════════════════════════════════════════════════════════
            # NAQAAE
            # ═══════════════════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("#### 🎓 تحليل الاعتماد المؤسسي — NAQAAE")

            if st.button(
                "🚀 تحليل الوثيقة بـ NAQAAE",
                use_container_width=True,
                type="primary",
                disabled=st.session_state.naqaae_done,
            ):
                with st.spinner("⏳ جاري الإرسال لـ NAQAAE... (قد يستغرق دقيقة أو أكثر)"):
                    res = analyze_with_naqaae(st.session_state.corrected_text)
                    st.session_state.naqaae_result = res
                    st.session_state.naqaae_done   = True
                    st.rerun()

            if st.session_state.naqaae_done and st.session_state.naqaae_result:
                res = st.session_state.naqaae_result

                if res.get("error"):
                    st.error(res["error"])
                    if st.button("🔄 إعادة المحاولة"):
                        st.session_state.naqaae_done   = False
                        st.session_state.naqaae_result = None
                        st.rerun()
                else:
                    status = res["status"]
                    score  = res["score"] or 0.0
                    recs   = res["recs"]  or ""

                    is_ok       = "معتمد" in status and "غير" not in status
                    card_class  = "status-card-ok"    if is_ok else "status-card-fail"
                    label_class = "status-label-ok"   if is_ok else "status-label-fail"
                    score_class = "score-green" if score>=70 else ("score-orange" if score>=50 else "score-red")

                    st.markdown(f"""
                    <div class="naqaae-section">
                      <div class="naqaae-title">🎓 نتيجة <span>NAQAAE</span></div>
                      <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start">

                        <div style="flex:1;min-width:200px">
                          <div class="{card_class}">
                            <div class="status-label {label_class}">{status}</div>
                            <div class="status-sub">حالة الاعتماد المؤسسي</div>
                          </div>
                          <div style="margin-top:20px">
                            <div class="score-num {score_class}">{score:.1f}<span style="font-size:1rem;opacity:.6">/100</span></div>
                            <div class="score-bar-bg">
                              <div class="score-bar-fill {score_class}" style="width:{min(score,100):.0f}%"></div>
                            </div>
                            <div style="color:#718096;font-size:0.8rem;text-align:center;margin-top:6px;direction:rtl">درجة الجودة</div>
                          </div>
                        </div>

                        <div style="flex:2;min-width:280px">
                          <div style="color:#63b3ed;font-size:0.82rem;font-weight:700;margin-bottom:8px;direction:rtl">
                            {'✅ لا توصيات مطلوبة' if score>60 else '📋 توصيات التحسين'}
                          </div>
                          <div class="recs-box">{recs.replace('<','&lt;').replace('>','&gt;')}</div>
                        </div>

                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("")
                    if st.button("🔄 إعادة تحليل NAQAAE"):
                        st.session_state.naqaae_done   = False
                        st.session_state.naqaae_result = None
                        st.rerun()

else:
    st.markdown("""
    <div style="background:#1a202c;border:2px dashed #2d3748;border-radius:16px;
                padding:50px;text-align:center;margin-top:10px">
      <div style="font-size:3rem;margin-bottom:12px">📂</div>
      <div style="color:#718096;font-size:1.05rem;direction:rtl">
        ارفع صورة أو PDF أو ملف Word للبدء
      </div>
      <div style="color:#4a5568;font-size:0.85rem;margin-top:8px">
        PNG · JPG · BMP · TIFF · WEBP · PDF · DOCX
      </div>
    </div>
    """, unsafe_allow_html=True)