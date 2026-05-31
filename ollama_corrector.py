"""
Groq Integration for Arabic Text Correction
يدعم Groq Cloud API عبر متغير البيئة GROQ_API_KEY
"""

import os
import requests
import re
import unicodedata
from typing import Optional, Tuple, List

# ─── Groq API ─────────────────────────────────────────────────────────────────
GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
DEFAULT_MODEL = "llama-3.3-70b-versatile"  # أفضل مودل مجاني على Groq

# ─── Visual Arabic Detection ──────────────────────────────────────────────────
_PRESENTATION_FORMS_RE = re.compile(r'[\uFB50-\uFDFF\uFE70-\uFEFF]')
_CONTROL_CHARS_RE = re.compile(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]')


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }


def is_visual_arabic(text: str) -> bool:
    return bool(_PRESENTATION_FORMS_RE.search(text))


def fix_visual_arabic(text: str) -> str:
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


def check_ollama_status() -> Tuple[bool, List[str]]:
    """
    بدل Ollama — بتتحقق من Groq API وترجع المودلز المتاحة
    الدالة اتركت باسمها عشان app.py مش محتاج تعديل
    """
    if not GROQ_API_KEY:
        return False, []
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers=_headers(),
            timeout=8,
        )
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            names = [m["id"] for m in models if "id" in m]
            # فلتر المودلز المفيدة بس
            preferred = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "gemma2-9b-it",
                "mixtral-8x7b-32768",
            ]
            ordered = [n for n in preferred if n in names]
            rest = [n for n in names if n not in preferred]
            return True, ordered + rest
        return False, []
    except Exception:
        return False, []


def send_to_ollama(text: str, model_name: str) -> Optional[str]:
    """
    الدالة اتركت باسمها عشان app.py مش محتاج تعديل
    بترسل النص لـ Groq وترجع النص المصحح
    """
    if is_visual_arabic(text):
        text = fix_visual_arabic(text)

    system_prompt = """أنت مساعد متخصص في تصحيح الأخطاء الإملائية في النصوص العربية.
مهمتك: تصحيح الأخطاء الإملائية فقط دون تغيير المعنى أو المحتوى أو التنسيق.

قواعد مهمة:
- صحح الأخطاء الإملائية فقط
- لا تغير المعنى أو المحتوى
- احتفظ بالتنسيق كما هو
- لا تضف أي تعليقات أو شروحات
- إذا وجدت تاريخاً أو رقم ID أو رقم ملف في النص، استخرجهم وضعهم في أعلى النص داخل مربع بهذا الشكل:
  ┌─────────────────────────────┐
  │ رقم الملف: [القيمة]         │
  │ التاريخ: [القيمة]           │
  └─────────────────────────────┘
- أعد النص المصحح فقط بدون تعليقات"""

    try:
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"صحح النص التالي:\n\n{text}"},
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 4096,
        }
        resp = requests.post(GROQ_API_URL, json=data, headers=_headers(), timeout=120)
        if resp.status_code != 200:
            return None

        corrected = resp.json()["choices"][0]["message"]["content"].strip()

        skip_kw = ['النص المصحح', 'التصحيح', 'إليك', 'هنا', 'الإجابة', 'النتيجة']
        lines = [
            l for l in corrected.split('\n')
            if not any(kw in l and len(l) < 50 for kw in skip_kw)
        ]
        corrected = '\n'.join(lines).strip()

        if len(corrected) < len(text) * 0.3:
            return text
        return corrected

    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def correct_text_with_ollama(text: str, model_name: str = DEFAULT_MODEL,
                              max_chunk: int = 2000,
                              progress_callback=None) -> Optional[str]:
    """
    الدالة اتركت باسمها عشان app.py مش محتاج تعديل
    بتصحح النص العربي باستخدام Groq
    """
    if not text.strip():
        return text

    if is_visual_arabic(text):
        text = fix_visual_arabic(text)

    if len(text) <= max_chunk:
        return send_to_ollama(text, model_name)

    lines = text.split('\n')
    parts, current = [], ""
    for line in lines:
        if len(current) + len(line) + 1 > max_chunk and current:
            parts.append(current)
            current = line + '\n'
        else:
            current += line + '\n'
    if current:
        parts.append(current)

    corrected_parts = []
    for i, part in enumerate(parts):
        if progress_callback:
            progress_callback(i + 1, len(parts))
        result = send_to_ollama(part, model_name)
        corrected_parts.append(result if result else part)

    return '\n'.join(corrected_parts)