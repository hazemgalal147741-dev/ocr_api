"""
Arabic OCR API — FastAPI
POST /ocr      ← ملف → نص
POST /analyze  ← نص → تحليل NAQAAE + تقرير Groq
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import io, json
from PIL import Image

from ocr_processor import OCRProcessor
from ollama_corrector import (
    check_ollama_status,
    correct_text_with_ollama,
    pre_analyze_text,
    synthesize_report,
)
from naqaae_evaluator import analyze_with_naqaae, extract_document_meta

app = FastAPI(title="Arabic OCR + NAQAAE API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = OCRProcessor()


# ─── Models ───────────────────────────────────────────────────────────────────

class OCRResponse(BaseModel):
    text: str
    language: str

class AnalyzeRequest(BaseModel):
    text: str
    model: Optional[str] = "llama-3.3-70b-versatile"

class AnalyzeResponse(BaseModel):
    document_id: Optional[str] = None
    document_date: Optional[str] = None
    naqaae_status: str
    naqaae_score: float
    domain_scores: dict = {}
    final_report: str
    error: Optional[str] = None


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/ocr", "/analyze"]}


# ─── Endpoint 1: OCR ─────────────────────────────────────────────────────────

@app.post("/ocr", response_model=OCRResponse)
async def ocr(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="ara | eng | ara+eng"),
):
    """
    ارفع ملف (صورة / PDF / DOCX) → يرجع النص المستخرج
    """
    ext     = file.filename.split(".")[-1].lower()
    content = await file.read()

    try:
        if ext in ("png", "jpg", "jpeg", "bmp", "tiff", "webp"):
            text, lang = processor.extract_from_pil(Image.open(io.BytesIO(content)), language)
        elif ext == "pdf":
            text, lang = processor.extract_from_pdf(content, language)
        elif ext == "docx":
            text, lang = processor.extract_from_docx(content)
        elif ext == "txt":
            text = content.decode("utf-8", errors="ignore")
            from ocr_processor import detect_language_from_text
            lang = language or detect_language_from_text(text)
        else:
            raise HTTPException(400, f"نوع الملف غير مدعوم: {ext}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

    return OCRResponse(text=text, language=lang)


# ─── Endpoint 2: Analyze ─────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest):
    """
    ابعت النص → يرجع:
    - naqaae_status : معتمد / غير معتمد
    - naqaae_score  : الدرجة من 100
    - domain_scores : درجات كل محور
    - final_report  : تقرير Groq النهائي بالتوصيات
    """
    if not body.text.strip():
        raise HTTPException(400, "النص فارغ")

    model = body.model or "llama-3.3-70b-versatile"

    # 0. استخراج رقم/كود الوثيقة وتاريخها — ولو الوثيقة قديمة (قبل 2025) نرفضها
    #    فوراً بدون أي استدعاءات Groq زيادة (تحليل أولي / NAQAAE / تقرير)
    meta       = extract_document_meta(body.text, model=model)
    doc_id     = meta.get("document_id")
    doc_date   = meta.get("document_date")
    doc_year   = meta.get("document_year")

    if doc_year is not None and doc_year < 2025:
        return AnalyzeResponse(
            document_id=doc_id,
            document_date=doc_date,
            naqaae_status="مرفوض - ملف قديم",
            naqaae_score=0.0,
            domain_scores={},
            final_report=(
                f"## ⚠️ هذا الملف قديم\n\n"
                f"تاريخ الوثيقة المُكتشف: **{doc_date or doc_year}**\n\n"
                "لا يمكن قبول وثائق يسبق تاريخها عام 2025 للتحليل والتقييم. "
                "برجاء رفع نسخة محدثة من الوثيقة بتاريخ 2025 أو أحدث."
            ),
        )

    # 1. Groq: تحليل أولي
    pre_raw = pre_analyze_text(body.text, model=model)
    pre_str = ""
    if pre_raw:
        try:
            clean   = pre_raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            pre_str = json.dumps(json.loads(clean), ensure_ascii=False)
        except Exception:
            pre_str = pre_raw

    # 2. NAQAAE: تصنيف
    naqaae          = analyze_with_naqaae(body.text)
    n_error         = naqaae.get("error")
    n_status        = naqaae.get("status") or "غير معروف"
    n_score         = float(naqaae.get("score") or 0.0)
    n_recs          = naqaae.get("recs") or ""
    n_domain_scores = naqaae.get("domain_scores", {})
    n_is_relevant   = naqaae.get("is_relevant", True)

    if n_error:
        return AnalyzeResponse(
            document_id=doc_id,
            document_date=doc_date,
            naqaae_status=n_status,
            naqaae_score=n_score,
            domain_scores=n_domain_scores,
            final_report="",
            error=n_error,
        )

    # لو النص مالوش علاقة بمعايير الاعتماد من الأساس، رسالة واضحة
    # بدون استدعاء Groq تاني لعمل تقرير على درجات غير حقيقية
    if not n_is_relevant:
        return AnalyzeResponse(
            document_id=doc_id,
            document_date=doc_date,
            naqaae_status=n_status,
            naqaae_score=n_score,
            domain_scores=n_domain_scores,
            final_report=(
                "## ⚠️ المحتوى غير مناسب للتحليل\n\n"
                "النص أو الملف المُرفق لا يحتوي على محتوى مؤسسي أو تعليمي "
                "يمكن تقييمه وفق معايير الاعتماد.\n\n"
                "برجاء التأكد من رفع وثيقة مؤسسية مناسبة مثل: خطة استراتيجية، "
                "تقرير سنوي، لائحة داخلية، أو مرفقات اعتماد، وإعادة المحاولة."
            ),
        )

    # 3. Groq: تقرير نهائي
    final_report = synthesize_report(
        text=body.text,
        pre_analysis=pre_str,
        naqaae_status=n_status,
        naqaae_score=n_score,
        naqaae_recs=n_recs,
        domain_scores=n_domain_scores,
        model=model,
    ) or ""

    return AnalyzeResponse(
        document_id=doc_id,
        document_date=doc_date,
        naqaae_status=n_status,
        naqaae_score=n_score,
        domain_scores=n_domain_scores,
        final_report=final_report,
    )


# ─── Endpoint 3: Process (OCR + Analyze في خطوة واحدة) ──────────────────────

class ProcessResponse(BaseModel):
    # OCR
    text: str
    language: str
    # Document Meta
    document_id: Optional[str] = None
    document_date: Optional[str] = None
    # NAQAAE + Groq
    naqaae_status: str
    naqaae_score: float
    domain_scores: dict = {}
    final_report: str
    error: Optional[str] = None


@app.post("/process", response_model=ProcessResponse)
async def process(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="ara | eng | ara+eng"),
    model: Optional[str] = Query("llama-3.3-70b-versatile"),
):
    """
    ارفع ملف → OCR → تصحيح Groq → NAQAAE → تقرير رفع الـ score
    كل حاجة في خطوة واحدة
    """
    ext     = file.filename.split(".")[-1].lower()
    content = await file.read()

    # ── OCR ──────────────────────────────────────────────────────────────────
    try:
        if ext in ("png", "jpg", "jpeg", "bmp", "tiff", "webp"):
            text, lang = processor.extract_from_pil(Image.open(io.BytesIO(content)), language)
        elif ext == "pdf":
            text, lang = processor.extract_from_pdf(content, language)
        elif ext == "docx":
            text, lang = processor.extract_from_docx(content)
        elif ext == "txt":
            text = content.decode("utf-8", errors="ignore")
            from ocr_processor import detect_language_from_text
            lang = language or detect_language_from_text(text)
        else:
            raise HTTPException(400, f"نوع الملف غير مدعوم: {ext}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"OCR error: {e}")

    # ── Groq (1): تصحيح إملائي ───────────────────────────────────────────────
    groq_ok, _ = check_ollama_status()
    corrected  = text
    if groq_ok:
        result    = correct_text_with_ollama(text, model_name=model)
        corrected = result if result else text

    # ── استخراج رقم/كود الوثيقة وتاريخها — ولو الوثيقة قديمة (قبل 2025)
    #    نرفضها فوراً بدون أي استدعاءات Groq زيادة ───────────────────────────
    meta     = extract_document_meta(corrected, model=model)
    doc_id   = meta.get("document_id")
    doc_date = meta.get("document_date")
    doc_year = meta.get("document_year")

    if doc_year is not None and doc_year < 2025:
        return ProcessResponse(
            text=corrected,
            language=lang,
            document_id=doc_id,
            document_date=doc_date,
            naqaae_status="مرفوض - ملف قديم",
            naqaae_score=0.0,
            domain_scores={},
            final_report=(
                f"## ⚠️ هذا الملف قديم\n\n"
                f"تاريخ الوثيقة المُكتشف: **{doc_date or doc_year}**\n\n"
                "لا يمكن قبول وثائق يسبق تاريخها عام 2025 للتحليل والتقييم. "
                "برجاء رفع نسخة محدثة من الوثيقة بتاريخ 2025 أو أحدث."
            ),
        )

    # ── Groq (2) pre-analysis ─────────────────────────────────────────────────
    pre_raw = pre_analyze_text(corrected, model=model)
    pre_str = ""
    if pre_raw:
        try:
            clean   = pre_raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            pre_str = json.dumps(json.loads(clean), ensure_ascii=False)
        except Exception:
            pre_str = pre_raw

    # ── NAQAAE ────────────────────────────────────────────────────────────────
    naqaae          = analyze_with_naqaae(corrected)
    n_error         = naqaae.get("error")
    n_status        = naqaae.get("status") or "غير معروف"
    n_score         = float(naqaae.get("score") or 0.0)
    n_recs          = naqaae.get("recs") or ""
    n_domain_scores = naqaae.get("domain_scores", {})
    n_is_relevant   = naqaae.get("is_relevant", True)

    if n_error:
        return ProcessResponse(
            text=corrected, language=lang,
            document_id=doc_id, document_date=doc_date,
            naqaae_status=n_status, naqaae_score=n_score,
            domain_scores=n_domain_scores,
            final_report="", error=n_error,
        )

    # لو النص مالوش علاقة بمعايير الاعتماد من الأساس، رسالة واضحة
    # بدون استدعاء Groq تاني لعمل تقرير على درجات غير حقيقية
    if not n_is_relevant:
        return ProcessResponse(
            text=corrected,
            language=lang,
            document_id=doc_id,
            document_date=doc_date,
            naqaae_status=n_status,
            naqaae_score=n_score,
            domain_scores=n_domain_scores,
            final_report=(
                "## ⚠️ المحتوى غير مناسب للتحليل\n\n"
                "الملف المُرفق لا يحتوي على محتوى مؤسسي أو تعليمي "
                "يمكن تقييمه وفق معايير الاعتماد.\n\n"
                "برجاء التأكد من رفع وثيقة مؤسسية مناسبة مثل: خطة استراتيجية، "
                "تقرير سنوي، لائحة داخلية، أو مرفقات اعتماد، وإعادة المحاولة."
            ),
        )

    # ── Groq (3): تقرير رفع الـ score ────────────────────────────────────────
    final_report = synthesize_report(
        text=corrected,
        pre_analysis=pre_str,
        naqaae_status=n_status,
        naqaae_score=n_score,
        naqaae_recs=n_recs,
        domain_scores=n_domain_scores,
        model=model,
    ) or ""

    return ProcessResponse(
        text=corrected,
        language=lang,
        document_id=doc_id,
        document_date=doc_date,
        naqaae_status=n_status,
        naqaae_score=n_score,
        domain_scores=n_domain_scores,
        final_report=final_report,
    )