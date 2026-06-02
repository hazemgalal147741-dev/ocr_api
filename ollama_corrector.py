"""
Groq Integration for Arabic Text Correction
يدعم Groq Cloud API عبر متغير البيئة GROQ_API_KEY
"""

import os
import requests
import re
import unicodedata
from typing import Optional, Tuple, List

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
DEFAULT_MODEL = "llama-3.3-70b-versatile"

_PRESENTATION_FORMS_RE = re.compile(r'[\uFB50-\uFDFF\uFE70-\uFEFF]')
_CONTROL_CHARS_RE      = re.compile(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]')


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }


def is_visual_arabic(text: str) -> bool:
    return bool(_PRESENTATION_FORMS_RE.search(text))


def fix_visual_arabic(text: str) -> str:
    lines, fixed_lines = text.split('\n'), []
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
            fixed_lines.append(' '.join(tokens[::-1]))
        else:
            fixed_lines.append(norm)
    return '\n'.join(fixed_lines)


def check_ollama_status() -> Tuple[bool, List[str]]:
    if not GROQ_API_KEY:
        return False, []
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers=_headers(), timeout=8,
        )
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            names  = [m["id"] for m in models if "id" in m]
            preferred = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "gemma2-9b-it",
                "mixtral-8x7b-32768",
            ]
            ordered = [n for n in preferred if n in names]
            rest    = [n for n in names if n not in preferred]
            return True, ordered + rest
        return False, []
    except Exception:
        return False, []


def _call_groq(messages: list, model: str, temperature: float = 0.1) -> Optional[str]:
    """helper عام لاستدعاء Groq"""
    try:
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        resp = requests.post(GROQ_API_URL, json=data, headers=_headers(), timeout=120)
        if resp.status_code != 200:
            return None
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def pre_analyze_text(text: str, model: str = DEFAULT_MODEL) -> Optional[str]:
    """
    المرحلة الأولى: Groq يحلل النص العربي ويستخرج:
    - ملخص الوثيقة
    - نقاط القوة
    - نقاط الضعف المحتملة
    ويرجع تحليله كـ JSON string
    """
    if is_visual_arabic(text):
        text = fix_visual_arabic(text)

    system = """أنت خبير في تقييم جودة الوثائق والتقارير المؤسسية العربية.
مهمتك: تحليل النص وتقديم تقرير أولي منظم بصيغة JSON فقط.

أعد JSON بهذا الشكل بالضبط (بدون أي نص خارج الـ JSON):
{
  "document_summary": "ملخص الوثيقة في جملتين",
  "document_type": "نوع الوثيقة (تقرير / خطة / سياسة / إجراء / أخرى)",
  "strengths": ["نقطة قوة 1", "نقطة قوة 2"],
  "weaknesses": ["نقطة ضعف 1", "نقطة ضعف 2"],
  "completeness_estimate": 75
}"""

    words = text.split()
    if len(words) > 1000:
        text = " ".join(words[:1000])

    result = _call_groq(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"حلل هذه الوثيقة:\n\n{text}"},
        ],
        model=model,
        temperature=0.2,
    )
    return result


def synthesize_report(
    text: str,
    pre_analysis: str,
    naqaae_status: str,
    naqaae_score: float,
    naqaae_recs: str,
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """
    المرحلة الثانية: Groq يلخص كل حاجة ويطلع تقرير نهائي منظم
    بيجمع: تحليله الأولي + نتيجة NAQAAE + التوصيات الخام
    """
    system = """أنت خبير اعتماد مؤسسي. بناءً على التحليل الأولي ونتيجة نظام NAQAAE،
اكتب تقريراً نهائياً واضحاً ومنظماً باللغة العربية.

التقرير يجب أن يحتوي على:
1. ملخص تنفيذي (جملتان)
2. نتيجة الاعتماد: معتمد ✅ أو غير معتمد ❌ مع الدرجة
3. أبرز نقاط القوة (3 نقاط كحد أقصى)
4. توصيات التحسين المحددة والقابلة للتنفيذ (مرتبة حسب الأولوية)
5. الخلاصة

اكتب بأسلوب مهني واضح. لا تذكر أسماء الأنظمة أو التقنيات المستخدمة."""

    user_content = f"""التحليل الأولي للوثيقة:
{pre_analysis}

نتيجة تقييم الجودة:
- الحالة: {naqaae_status}
- الدرجة: {naqaae_score:.1f} / 100
- ملاحظات النظام: {naqaae_recs or 'لا توجد'}

النص الأصلي (أول 500 كلمة):
{' '.join(text.split()[:500])}

اكتب التقرير النهائي:"""

    return _call_groq(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        model=model,
        temperature=0.3,
    )


def send_to_ollama(text: str, model_name: str) -> Optional[str]:
    """تصحيح إملائي فقط — محتفظ بالدالة للتوافق"""
    if is_visual_arabic(text):
        text = fix_visual_arabic(text)

    system = """أنت مساعد متخصص في تصحيح الأخطاء الإملائية في النصوص العربية.
- صحح الأخطاء الإملائية فقط
- لا تغير المعنى أو التنسيق
- أعد النص المصحح فقط بدون تعليقات"""

    result = _call_groq(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"صحح النص:\n\n{text}"},
        ],
        model=model_name,
        temperature=0.1,
    )

    if result and len(result) >= len(text) * 0.3:
        return result
    return text


def correct_text_with_ollama(text: str, model_name: str = DEFAULT_MODEL,
                              max_chunk: int = 2000,
                              progress_callback=None) -> Optional[str]:
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