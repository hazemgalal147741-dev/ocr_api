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
from naqaae_client import analyze_with_naqaae

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
    naqaae_status: str
    naqaae_score: float
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
    - final_report  : تقرير Groq النهائي بالتوصيات
    """
    if not body.text.strip():
        raise HTTPException(400, "النص فارغ")

    model = body.model or "llama-3.3-70b-versatile"

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
    naqaae   = analyze_with_naqaae(body.text)
    n_error  = naqaae.get("error")
    n_status = naqaae.get("status") or "غير معروف"
    n_score  = float(naqaae.get("score") or 0.0)
    n_recs   = naqaae.get("recs") or ""

    if n_error:
        return AnalyzeResponse(
            naqaae_status=n_status,
            naqaae_score=n_score,
            final_report="",
            error=n_error,
        )

    # 3. Groq: تقرير نهائي
    final_report = synthesize_report(
        text=body.text,
        pre_analysis=pre_str,
        naqaae_status=n_status,
        naqaae_score=n_score,
        naqaae_recs=n_recs,
        model=model,
    ) or ""

    return AnalyzeResponse(
        naqaae_status=n_status,
        naqaae_score=n_score,
        final_report=final_report,
    )