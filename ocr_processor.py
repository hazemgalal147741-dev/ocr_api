"""
OCR Processor v3 - Arabic & English
Auto language detection + advanced preprocessing + PDF/DOCX support
"""

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import os
import re
import io
import unicodedata
from typing import Optional, List, Dict, Tuple

# ─── Tesseract Auto-Detect ────────────────────────────────────────────────────
common_paths = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    '/usr/bin/tesseract',
    '/usr/local/bin/tesseract',
]
for path in common_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break

# ─── Arabic Support ───────────────────────────────────────────────────────────
# arabic_reshaper و bidi مش محتاجينهم
# Streamlit والـ browser بيتعاملوا مع Unicode العربي القياسي تلقائياً
ARABIC_SUPPORT = False  # عمداً معطّل

# ─── PDF Support ──────────────────────────────────────────────────────────────
try:
    from pdf2image import convert_from_bytes
    PDF_IMAGE_SUPPORT = True
except ImportError:
    PDF_IMAGE_SUPPORT = False

try:
    from pypdf import PdfReader as _PdfReader
    def _make_pdf_reader(b): return _PdfReader(io.BytesIO(b))
    PYPDF_SUPPORT = True
except ImportError:
    try:
        import PyPDF2
        def _make_pdf_reader(b): return PyPDF2.PdfReader(io.BytesIO(b))
        PYPDF_SUPPORT = True
    except ImportError:
        PYPDF_SUPPORT = False

# ─── DOCX Support ─────────────────────────────────────────────────────────────
try:
    from docx import Document as DocxDocument
    DOCX_READ_SUPPORT = True
except ImportError:
    DOCX_READ_SUPPORT = False


# ─── Visual Arabic Fix ────────────────────────────────────────────────────────
_PRESENTATION_FORMS_RE = re.compile(r'[\uFB50-\uFDFF\uFE70-\uFEFF]')
_CONTROL_CHARS_RE = re.compile(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]')


def is_visual_arabic(text: str) -> bool:
    """هل النص يحتوي على Presentation Forms (Visual Arabic من OCR)؟"""
    return bool(_PRESENTATION_FORMS_RE.search(text))


def fix_visual_arabic(text: str) -> str:
    """
    تصليح النص العربي المرئي (Visual Arabic / Presentation Forms).

    الخوارزمية لكل سطر:
      1. NFKC  → تحويل FB50-FDFF إلى Unicode عربي قياسي (بيصلح الحروف نفسها)
      2. حذف invisible bidi control chars
      3. لو السطر كان أصلاً Presentation Forms → عكس ترتيب الـ tokens فقط
         (بدون عكس حروف الكلمات — NFKC كفيل بده)
      الأرقام والتواريخ والـ IDs لا تُمس.
    """
    lines = text.split('\n')
    fixed_lines = []

    for line in lines:
        if not line.strip():
            fixed_lines.append(line)
            continue

        was_presentation = is_visual_arabic(line)

        norm = unicodedata.normalize('NFKC', line)
        norm = _CONTROL_CHARS_RE.sub('', norm).strip()

        arabic_count = sum(1 for c in norm if '\u0600' <= c <= '\u06FF')
        if arabic_count <= 1:
            fixed_lines.append(norm)
            continue

        if was_presentation:
            tokens = norm.split()
            tokens = tokens[::-1]
            fixed_lines.append(' '.join(tokens))
        else:
            fixed_lines.append(norm)

    return '\n'.join(fixed_lines)


def detect_language(pil_image: Image.Image) -> str:
    try:
        small = pil_image.copy()
        small.thumbnail((600, 600))
        sample = pytesseract.image_to_string(
            small.convert('L'), lang='ara+eng', config='--oem 3 --psm 6')
        arabic = sum(1 for c in sample if '\u0600' <= c <= '\u06FF')
        latin  = sum(1 for c in sample if c.isalpha() and ord(c) < 256)
        total  = arabic + latin
        if total == 0:
            return 'ara+eng'
        ratio = arabic / total
        if ratio > 0.6:   return 'ara'
        if ratio < 0.15:  return 'eng'
        return 'ara+eng'
    except:
        return 'ara+eng'


def detect_language_from_text(text: str) -> str:
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin  = sum(1 for c in text if c.isalpha() and ord(c) < 256)
    total  = arabic + latin
    if total == 0: return 'ara+eng'
    ratio = arabic / total
    if ratio > 0.6:  return 'ara'
    if ratio < 0.15: return 'eng'
    return 'ara+eng'


class ImagePreprocessor:
    """Four preprocessing strategies — processor tries all and picks best"""

    @staticmethod
    def otsu(img: Image.Image) -> Image.Image:
        arr = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.equalizeHist(gray)
        _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(t)

    @staticmethod
    def adaptive(img: Image.Image) -> Image.Image:
        arr = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        t = cv2.adaptiveThreshold(gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return Image.fromarray(t)

    @staticmethod
    def clahe(img: Image.Image) -> Image.Image:
        arr = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = cl.apply(gray)
        _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(t)

    @staticmethod
    def sharpen(img: Image.Image) -> Image.Image:
        arr = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        k = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, k)
        _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pil = Image.fromarray(t)
        return ImageEnhance.Contrast(pil).enhance(1.5)

    @staticmethod
    def upscale(img: Image.Image, min_dim: int = 1400) -> Image.Image:
        w, h = img.size
        if min(w, h) < min_dim:
            scale = min_dim / min(w, h)
            return img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        return img


class OCRProcessor:
    STRATEGIES = [
        ImagePreprocessor.otsu,
        ImagePreprocessor.adaptive,
        ImagePreprocessor.clahe,
        ImagePreprocessor.sharpen,
    ]
    PSMS = ['--oem 3 --psm 6', '--oem 3 --psm 3', '--oem 3 --psm 4']

    def __init__(self, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        try:
            self.supported_languages = pytesseract.get_languages()
        except:
            self.supported_languages = ['eng']

    def _best_ocr(self, pil_image: Image.Image, language: str) -> str:
        pil_image = ImagePreprocessor.upscale(pil_image)
        best = ""
        for strategy in self.STRATEGIES:
            try:
                processed = strategy(pil_image)
            except Exception:
                continue
            for psm in self.PSMS:
                try:
                    raw = pytesseract.image_to_string(
                        processed, config=f'{psm} -l {language}')
                    if len(raw.strip()) > len(best.strip()):
                        best = raw
                except Exception:
                    continue
        if not best.strip():
            try:
                best = pytesseract.image_to_string(
                    pil_image, lang=language, config='--oem 3 --psm 3')
            except:
                pass
        return best

    def extract_from_pil(self, pil_image: Image.Image,
                          language: Optional[str] = None) -> Tuple[str, str]:
        if not language:
            language = detect_language(pil_image)
        raw = self._best_ocr(pil_image, language)
        if 'ara' in language:
            raw = self.process_arabic_text(raw)
            raw = self.clean_ocr_errors(raw)
        return raw.strip(), language

    def extract_from_pdf(self, pdf_bytes: bytes,
                          language: Optional[str] = None,
                          progress_callback=None) -> Tuple[str, str]:
        native_text = ""
        if PYPDF_SUPPORT:
            try:
                reader = _make_pdf_reader(pdf_bytes)
                parts = []
                for page in reader.pages:
                    t = page.extract_text() or ""
                    parts.append(t)
                native_text = "\n".join(parts).strip()
            except Exception:
                native_text = ""

        if len(native_text) > 50:
            lang = language or detect_language_from_text(native_text)
            if 'ara' in lang:
                native_text = self.process_arabic_text(native_text)
                native_text = self.clean_ocr_errors(native_text)
            return native_text, lang

        if not PDF_IMAGE_SUPPORT:
            return ("⚠️ هذا PDF ممسوح ضوئيًا ويحتاج pdf2image.\n"
                    "شغّل: pip install pdf2image\n"
                    "وثبّت Poppler: https://github.com/oschwartz10612/poppler-windows/releases"), 'ara+eng'

        try:
            pages = convert_from_bytes(pdf_bytes, dpi=250)
        except Exception as e:
            return f"⚠️ فشل تحويل PDF: {e}", 'ara+eng'

        detected_lang = language
        if not detected_lang and pages:
            detected_lang = detect_language(pages[0])
        detected_lang = detected_lang or 'ara+eng'

        page_texts = []
        for i, page_img in enumerate(pages):
            if progress_callback:
                progress_callback(i + 1, len(pages))
            text, _ = self.extract_from_pil(page_img, detected_lang)
            page_texts.append(f"── صفحة {i+1} ──\n{text}")

        return "\n\n".join(page_texts), detected_lang

    def extract_from_docx(self, docx_bytes: bytes) -> Tuple[str, str]:
        if not DOCX_READ_SUPPORT:
            return ("⚠️ python-docx غير مثبت.\nشغّل: pip install python-docx"), 'ara+eng'
        try:
            doc = DocxDocument(io.BytesIO(docx_bytes))
            raw = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            lang = detect_language_from_text(raw)
            if 'ara' in lang:
                raw = self.process_arabic_text(raw)
                raw = self.clean_ocr_errors(raw)
            return raw.strip(), lang
        except Exception as e:
            return f"⚠️ فشل قراءة DOCX: {e}", 'ara+eng'

    def extract_receipt_data(self, pil_image: Image.Image) -> Dict:
        text, _ = self.extract_from_pil(pil_image, 'ara+eng')
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        result = {'raw_text': text, 'lines': lines, 'total_amount': None, 'date': None}
        for line in lines:
            if any(w in line.lower() for w in ['مجموع', 'total', 'المبلغ', 'الإجمالي']):
                nums = re.findall(r'\d+', line)
                if nums:
                    result['total_amount'] = nums[-1]
            if any(c.isdigit() for c in line) and ('/' in line or '-' in line):
                result['date'] = line
        return result

    def process_arabic_text(self, text: str) -> str:
        """
        معالجة النص العربي — بدون reshaper/bidi
        الـ browser بيتعامل مع Unicode تلقائياً
        """
        if is_visual_arabic(text):
            text = fix_visual_arabic(text)
        return text

    def clean_ocr_errors(self, text: str) -> str:
        for char in ['ا', 'أ', 'إ', 'آ']:
            text = re.sub(f'{re.escape(char)}{{3,}}', char, text)
        text = re.sub(r'([\u0600-\u06FF])\1{2,}', r'\1', text)
        fixes = {
            r'اال(\w)': r'الا\1',
            r'التادريبية': 'التدريبية',
            r'رسا+لة': 'رسالة',
            r'المؤسا+سة': 'المؤسسة',
        }
        for err, fix in fixes.items():
            text = re.sub(err, fix, text)
        text = re.sub(r'\bاا([بتثجحخدذرزسشصضطظعغفقكلمنهوي])', r'ال\1', text)
        return text
