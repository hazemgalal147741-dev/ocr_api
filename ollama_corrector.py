"""
Groq Integration
مرحلتين: تصحيح إملائي + تقرير رفع الـ score
"""

import os
import requests
import re
import time
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


def _parse_retry_wait(resp) -> float:
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    try:
        match = re.search(r"try again in ([\d.]+)s", resp.text)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return 5.0


def _call_groq(messages: list, model: str, temperature: float = 0.1,
                max_retries: int = 3) -> Optional[str]:
    for attempt in range(max_retries + 1):
        try:
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4096,
            }
            resp = requests.post(GROQ_API_URL, json=data, headers=_headers(), timeout=120)

            if resp.status_code == 429:
                wait = min(_parse_retry_wait(resp) + 0.5, 30)
                print(f"[Groq] Rate limited (attempt {attempt + 1}/{max_retries + 1}), waiting {wait}s")
                if attempt < max_retries:
                    time.sleep(wait)
                    continue
                return None

            if resp.status_code != 200:
                print(f"[Groq] HTTP {resp.status_code}: {resp.text[:500]}")
                return None

            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Groq] Exception: {e}")
            return None
    return None


# ─── المرحلة الأولى: تصحيح إملائي ───────────────────────────────────────────

def send_to_ollama(text: str, model_name: str) -> Optional[str]:
    """Groq (1): تصحيح الأخطاء الإملائية فقط"""
    if is_visual_arabic(text):
        text = fix_visual_arabic(text)

    system = """أنت مدقق لغوي متخصص في النصوص العربية المؤسسية.
مهمتك الوحيدة: تصحيح الأخطاء الإملائية والنحوية فقط.

قواعد صارمة:
- لا تغير المعنى أو المحتوى أو التنسيق
- لا تحذف أي جزء من النص
- لا تضف تعليقات أو شروحات
- أعد النص المصحح فقط"""

    result = _call_groq(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"صحح الأخطاء الإملائية في النص التالي:\n\n{text}"},
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


# ─── المرحلة الثانية: تقرير رفع الـ score ────────────────────────────────────

def synthesize_report(
    text: str,
    pre_analysis: str,
    naqaae_status: str,
    naqaae_score: float,
    naqaae_recs: str,
    domain_scores: Optional[dict] = None,
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """
    Groq (2): بعد ما NAQAAE دي الـ score
    يحلل ليه الـ score كده وإزاي نرفعه بخطوات محددة ومرتبة
    حسب أولوية المحاور الأضعف فعلياً — بدون Markdown أو رموز
    """

    domain_scores = domain_scores or {}

    sorted_domains = sorted(domain_scores.items(), key=lambda kv: kv[1])

    if sorted_domains:
        ranked_domains_str = "\n".join(
            f"{i+1}. {name} — {score:.0f}/100"
            for i, (name, score) in enumerate(sorted_domains)
        )
    else:
        ranked_domains_str = "لا توجد بيانات تفصيلية للمحاور"

    weakest_three = [name for name, _ in sorted_domains[:3]]

    if naqaae_score >= 90:
        score_level = "ممتاز — الوثيقة معتمدة بدرجة عالية جداً"
    elif naqaae_score >= 70:
        score_level = "جيد — الوثيقة معتمدة"
    elif naqaae_score >= 50:
        score_level = "متوسط — يحتاج تحسينات جوهرية"
    else:
        score_level = "منخفض — الوثيقة تحتاج إعادة هيكلة"

    gap_note = f"الفجوة المتبقية {100 - naqaae_score:.0f} نقطة للكمال"

    system = """أنت خبير اعتماد مؤسسي متخصص في معايير NAQAAE.
مهمتك: تحليل نتيجة تقييم الوثيقة وتقديم تقرير واضح ومرتب بدون أي رموز أو تنسيق Markdown.

قواعد التنسيق الصارمة:
- ممنوع تماماً استخدام: ## أو ** أو * أو - أو # أو أي رموز Markdown
- ممنوع استخدام أي إيموجي أو رموز خاصة (مثل 📊 🚀 ✅ 🪜 📈)
- اكتب نصاً عربياً واضحاً ومنظماً بعناوين مرقمة فقط
- افصل كل قسم عن التالي بسطر فاضي واحد
- افصل كل خطوة أو بند عن التالي بسطر فاضي واحد
- لا تلاصق خطوتين أو بندين في نفس الفقرة

اكتب التقرير بهذا الهيكل بالضبط:

ملخص النتيجة
[جملة أو جملتان عن الوضع الحالي والدرجة الإجمالية]

ترتيب المحاور من الأضعف للأقوى
[قائمة مرقمة، كل محور في سطر مستقل مع درجته]

خطة التحسين المرتبة بالأولوية

أولاً: [اسم المحور الأضعف] — الدرجة الحالية: [X]/100
الإجراء الأول: [إجراء محدد وقابل للتنفيذ]

الإجراء الثاني: [إجراء محدد وقابل للتنفيذ]

الأثر المتوقع على الدرجة: زيادة تقريبية [X] نقاط

ثانياً: [اسم المحور الثاني] — الدرجة الحالية: [X]/100
الإجراء الأول: [إجراء محدد وقابل للتنفيذ]

الإجراء الثاني: [إجراء محدد وقابل للتنفيذ]

الأثر المتوقع على الدرجة: زيادة تقريبية [X] نقاط

ثالثاً: [اسم المحور الثالث] — الدرجة الحالية: [X]/100
الإجراء الأول: [إجراء محدد وقابل للتنفيذ]

الإجراء الثاني: [إجراء محدد وقابل للتنفيذ]

الأثر المتوقع على الدرجة: زيادة تقريبية [X] نقاط

خطوات الحصول على الاعتماد رسمياً

الخطوة الأولى: [عنوان الخطوة]
[وصف الإجراء في جملة أو جملتين]

الخطوة الثانية: [عنوان الخطوة]
[وصف الإجراء في جملة أو جملتين]

[استمر حتى الخطوة السابعة أو الثامنة، آخرها تقديم طلب الاعتماد الرسمي]

الخلاصة
[تقدير الدرجة المتوقعة بعد تنفيذ التوصيات مع التأكيد على إمكانية الوصول إلى 100]"""

    user_content = f"""معلومات الوثيقة:
- الدرجة الإجمالية: {naqaae_score:.1f} / 100
- مستوى الدرجة: {score_level}
- الهدف: الوصول إلى 100/100
- {gap_note}
- حالة الاعتماد: {naqaae_status}
- ملاحظات نظام التقييم: {naqaae_recs or 'لا توجد'}

ترتيب المحاور من الأضعف للأقوى (اتبعه حرفياً):
{ranked_domains_str}

أضعف 3 محاور تتناولهم بالتفصيل في خطة التحسين بهذا الترتيب: {', '.join(weakest_three) if weakest_three else 'غير متوفر'}

محتوى الوثيقة (أول 800 كلمة):
{' '.join(text.split()[:800])}

اكتب التقرير بدون أي رموز أو Markdown، نصاً عربياً منظماً فقط:"""

    return _call_groq(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        model=model,
        temperature=0.3,
    )


# ─── pre_analyze (محتفظ بيه للتوافق) ────────────────────────────────────────

def pre_analyze_text(text: str, model: str = DEFAULT_MODEL) -> Optional[str]:
    """تحليل أولي سريع للوثيقة"""
    system = """أنت خبير وثائق مؤسسية. حلل النص وأعد JSON فقط بهذا الشكل:
{
  "document_type": "نوع الوثيقة",
  "strengths": ["نقطة قوة 1", "نقطة قوة 2"],
  "weaknesses": ["نقطة ضعف 1", "نقطة ضعف 2"]
}
بدون أي نص خارج الـ JSON."""

    words = text.split()
    if len(words) > 800:
        text = " ".join(words[:800])

    return _call_groq(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"حلل:\n\n{text}"},
        ],
        model=model,
        temperature=0.2,
    )