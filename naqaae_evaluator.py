"""
naqaae_evaluator.py
بيحل محل naqaae_client.py
بيقيّم الوثيقة على 12 معيار NAQAAE الرسمية عن طريق Groq

Prompt Engineering Techniques Applied:
  1. Role Prompting
  2. Scoring Rubric (explicit criteria)
  3. Chain of Thought (step-by-step reasoning before scoring)
  4. Few-Shot Examples (good doc vs bad doc)
  5. Structured Output (strict JSON schema)
  6. Negative Instructions (irrelevant content, edge cases)
  7. Output Constraints (no text outside JSON)
"""

import os
import json
import re
import time
import requests
from typing import Optional

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
DEFAULT_MODEL = "llama-3.3-70b-versatile"

DOMAINS = [
    "التخطيط الاستراتيجي",
    "القيادة والحوكمة",
    "إدارة الجودة والتطوير",
    "أعضاء هيئة التدريس والهيئة المعاونة",
    "الجهاز الإداري",
    "الموارد المالية والمادية",
    "المعايير الأكاديمية والبرامج التعليمية",
    "التدريس والتعلم",
    "الطالب والخريجون",
    "البحث العلمي والأنشطة العلمية",
    "الدراسات العليا",
    "المشاركة المجتمعية وتنمية البيئة",
]

# ─── Few-Shot Examples ────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = """
مثال 1 — وثيقة مؤسسية قوية (درجات متوقعة: 80-95)
النص:
تمتلك كلية الهندسة خطة استراتيجية خمسية (2022-2027) معتمدة من مجلس الكلية
بتاريخ 15/3/2022، تتضمن رؤية ورسالة واضحتين وأهدافاً استراتيجية مرتبطة
بمؤشرات أداء قابلة للقياس. يضم مجلس الكلية لجان متخصصة للجودة والبحث العلمي.
أعضاء هيئة التدريس حاصلون على الدكتوراه ولديهم خطط سنوية للتطوير المهني.
معدل توظيف الخريجين 85% خلال 6 أشهر. الميزانية تغطي 120% من الاحتياجات.

الـ JSON المتوقع:
{
  "thinking": "وثيقة مؤسسية قوية تحتوي على خطة استراتيجية واضحة، لجان جودة، بيانات خريجين.",
  "is_relevant": true,
  "domain_scores": {
    "التخطيط الاستراتيجي": 90,
    "القيادة والحوكمة": 85,
    "إدارة الجودة والتطوير": 70,
    "أعضاء هيئة التدريس والهيئة المعاونة": 82,
    "الجهاز الإداري": 60,
    "الموارد المالية والمادية": 87,
    "المعايير الأكاديمية والبرامج التعليمية": 80,
    "التدريس والتعلم": 65,
    "الطالب والخريجون": 88,
    "البحث العلمي والأنشطة العلمية": 55,
    "الدراسات العليا": 50,
    "المشاركة المجتمعية وتنمية البيئة": 58
  },
  "overall_score": 72.5,
  "strengths": "خطة استراتيجية واضحة مع مؤشرات قياس ومعدل توظيف خريجين مرتفع",
  "weaknesses": "البحث العلمي والمشاركة المجتمعية يحتاجان توثيقاً أوضح"
}

مثال 2 — وثيقة مؤسسية ضعيفة (درجات متوقعة: 10-30)
النص:
المعهد يضم عدداً من الأقسام ويقدم برامج متنوعة. يوجد مجلس إدارة.
المدرسون متخصصون في مجالاتهم. يتم قبول الطلاب وفق اللوائح العامة.

الـ JSON المتوقع:
{
  "thinking": "وثيقة شديدة الإيجاز، لا تحتوي على أدلة أو تفاصيل لأي معيار.",
  "is_relevant": true,
  "domain_scores": {
    "التخطيط الاستراتيجي": 15,
    "القيادة والحوكمة": 20,
    "إدارة الجودة والتطوير": 10,
    "أعضاء هيئة التدريس والهيئة المعاونة": 18,
    "الجهاز الإداري": 12,
    "الموارد المالية والمادية": 10,
    "المعايير الأكاديمية والبرامج التعليمية": 22,
    "التدريس والتعلم": 15,
    "الطالب والخريجون": 20,
    "البحث العلمي والأنشطة العلمية": 10,
    "الدراسات العليا": 50,
    "المشاركة المجتمعية وتنمية البيئة": 10
  },
  "overall_score": 17.7,
  "strengths": "الوثيقة تقر بوجود هيكل تنظيمي أساسي",
  "weaknesses": "غياب شبه كامل للتفاصيل والأدلة في جميع المعايير"
}
"""

# ─── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""أنت خبير تقييم مؤسسي متخصص في معايير الاعتماد المؤسسي الرسمية
للهيئة القومية لضمان جودة التعليم والاعتماد (NAQAAE) في مصر —
دليل اعتماد كليات ومعاهد التعليم العالي (إصدار 2015 المعدل)،
وهو يتضمن 12 معياراً رسمياً.

══════════════════════════════════════════
خطوات التفكير (Chain of Thought) — اتبعها بالترتيب:
══════════════════════════════════════════
الخطوة 1 — تحديد الصلة:
  هل النص وثيقة مؤسسية / تعليمية / اعتماد جودة؟
  أم نص لا علاقة له بالموضوع (محادثة، نص أدبي، كود، إلخ)؟

الخطوة 2 — تحليل كل معيار من الـ 12:
  استخرج الأدلة الموجودة في النص لكل معيار قبل إعطاء الدرجة.

الخطوة 3 — التقييم وفق الـ Rubric:
  • غير مذكور أو غير متعلق  → 0  - 30
  • مذكور بشكل سطحي          → 30 - 60
  • مفصّل مع أدلة             → 60 - 85
  • شامل ومتكامل مع مؤشرات   → 85 - 100

الخطوة 4 — حساب overall_score = متوسط الـ 12 درجة.

الخطوة 5 — كتابة الـ JSON فقط.

══════════════════════════════════════════
قواعد خاصة (Negative Instructions):
══════════════════════════════════════════
- لا تخترع معلومات غير موجودة في النص.
- لا تفترض وجود أدلة لم تُذكر صراحةً.
- لو النص لا علاقة له بالموضوع المؤسسي → is_relevant=false وأعط 0-10 لكل معيار.
- معيار "الدراسات العليا" قد لا ينطبق على مؤسسات بدون برامج عليا → أعطه 50.
- لا تضع أي نص خارج الـ JSON في الإجابة النهائية.
- حقل "thinking" مطلوب دائماً.

══════════════════════════════════════════
أمثلة توضيحية (Few-Shot Examples):
══════════════════════════════════════════
{FEW_SHOT_EXAMPLES}

══════════════════════════════════════════
صيغة الإجابة المطلوبة (Structured Output):
══════════════════════════════════════════
أعد JSON فقط بهذا الشكل بالضبط، بدون أي نص قبله أو بعده:
{{
  "thinking": "<شرح موجز لتفكيرك — 2-4 جمل>",
  "is_relevant": <true أو false>,
  "domain_scores": {{
    "التخطيط الاستراتيجي":                   <0-100>,
    "القيادة والحوكمة":                       <0-100>,
    "إدارة الجودة والتطوير":                  <0-100>,
    "أعضاء هيئة التدريس والهيئة المعاونة":   <0-100>,
    "الجهاز الإداري":                         <0-100>,
    "الموارد المالية والمادية":               <0-100>,
    "المعايير الأكاديمية والبرامج التعليمية": <0-100>,
    "التدريس والتعلم":                        <0-100>,
    "الطالب والخريجون":                       <0-100>,
    "البحث العلمي والأنشطة العلمية":          <0-100>,
    "الدراسات العليا":                        <0-100>,
    "المشاركة المجتمعية وتنمية البيئة":       <0-100>
  }},
  "overall_score": <متوسط الـ 12 درجة>,
  "strengths":  "<أهم نقطتين إيجابيتين، أو فارغة لو النص غير متعلق>",
  "weaknesses": "<أهم نقطتين سلبيتين، أو وصف كون النص غير مؤسسي>"
}}"""


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }


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


def _call_groq(messages: list, model: str = DEFAULT_MODEL, max_retries: int = 3) -> Optional[str]:
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                GROQ_API_URL,
                json={
                    "model":       model,
                    "messages":    messages,
                    "temperature": 0.1,
                    "max_tokens":  2048,
                },
                headers=_headers(),
                timeout=120,
            )

            if resp.status_code == 429:
                wait = min(_parse_retry_wait(resp) + 0.5, 30)
                print(f"[Groq/NAQAAE] Rate limited (attempt {attempt+1}/{max_retries+1}), waiting {wait}s")
                if attempt < max_retries:
                    time.sleep(wait)
                    continue
                return None

            if resp.status_code != 200:
                print(f"[Groq/NAQAAE] HTTP {resp.status_code}: {resp.text[:500]}")
                return None

            return resp.json()["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"[Groq/NAQAAE] Exception: {e}")
            return None
    return None


def analyze_with_naqaae(text: str, model: str = DEFAULT_MODEL) -> dict:
    """
    نفس الـ interface القديم بتاع naqaae_client.py
    بيرجع: {"status", "score", "recs", "error", "domain_scores", "thinking"}
    """
    if not text or not text.strip():
        return {"status": None, "score": None, "recs": None,
                "error": "⚠️ النص فاضي", "domain_scores": {}, "thinking": ""}

    words = text.split()
    if len(words) > 1200:
        text = " ".join(words[:1200])

    user_content = (
        "قيّم الوثيقة التالية وفق المعايير الـ 12 الرسمية لـ NAQAAE.\n"
        "اتبع خطوات التفكير المذكورة في التعليمات، ثم أعد JSON فقط:\n\n"
        f"{text}"
    )

    raw = _call_groq(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        model=model,
    )

    if not raw:
        return {"status": "غير معروف", "score": 0.0, "recs": "",
                "error": "⚠️ تعذر إكمال التحليل، برجاء المحاولة مرة أخرى بعد قليل",
                "domain_scores": {}, "thinking": ""}

    try:
        clean = raw.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean.strip())
    except Exception:
        return {"status": "غير معروف", "score": 0.0, "recs": raw,
                "error": None, "domain_scores": {}, "thinking": ""}

    domain_scores = data.get("domain_scores", {})
    is_relevant   = data.get("is_relevant", True)
    thinking      = data.get("thinking", "")

    raw_score = data.get("overall_score")
    if raw_score:
        score = float(raw_score)
    elif domain_scores:
        vals = [v for v in domain_scores.values() if isinstance(v, (int, float))]
        score = sum(vals) / len(vals) if vals else 0.0
    else:
        score = 0.0

    score = round(max(0.0, min(score, 100.0)), 1)

    if not is_relevant:
        status = "غير قابل للتقييم"
    elif score >= 70:
        status = "معتمد"
    elif score >= 50:
        status = "مؤجل"
    else:
        status = "غير معتمد"

    strengths  = data.get("strengths", "")
    weaknesses = data.get("weaknesses", "")
    if not is_relevant:
        recs = (weaknesses or
                "المحتوى لا يتعلق بمعايير الاعتماد المؤسسي، "
                "برجاء رفع وثيقة مؤسسية (خطة استراتيجية، تقرير، لائحة، أو مرفقات اعتماد).")
    else:
        recs = f"نقاط القوة: {strengths}\nنقاط الضعف: {weaknesses}" if strengths or weaknesses else ""

    return {
        "status":        status,
        "score":         score,
        "recs":          recs,
        "error":         None,
        "domain_scores": domain_scores,
        "is_relevant":   is_relevant,
        "thinking":      thinking,
    }


def check_space_status() -> bool:
    return bool(GROQ_API_KEY)


def extract_document_meta(text: str, model: str = DEFAULT_MODEL) -> dict:
    empty = {"document_id": None, "document_date": None, "document_year": None}

    if not text or not text.strip():
        return empty

    words = text.split()
    sample = " ".join(words[:600])

    system = """أنت مساعد متخصص في استخراج البيانات الوصفية من الوثائق المؤسسية العربية.
مهمتك: استخراج رقم/كود الوثيقة (لو موجود) وأحدث تاريخ مذكور في النص.

قواعد:
- لو مفيش رقم/كود وثيقة واضح → document_id = null
- لو مفيش تاريخ واضح → document_date = null و document_year = null
- لا تحوّل التواريخ الهجرية — استخرج السنة الميلادية فقط لو مذكورة صراحة
- التاريخ بصيغة ISO: YYYY-MM-DD أو YYYY-MM أو YYYY
- لا تضع أي نص خارج الـ JSON

أعد JSON فقط:
{
  "document_id":   "<رقم/كود الوثيقة أو null>",
  "document_date": "<تاريخ بصيغة ISO أو null>",
  "document_year": <السنة كرقم أو null>
}"""

    raw = _call_groq(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"استخرج البيانات من النص التالي:\n\n{sample}"},
        ],
        model=model,
    )

    if not raw:
        return empty

    try:
        clean = raw.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean.strip())
    except Exception:
        return empty

    doc_id   = data.get("document_id") or None
    doc_date = data.get("document_date") or None
    doc_year = data.get("document_year")

    try:
        doc_year = int(doc_year) if doc_year is not None else None
    except (ValueError, TypeError):
        doc_year = None

    if doc_year is None and doc_date:
        match = re.match(r"^(\d{4})", doc_date.strip())
        if match:
            doc_year = int(match.group(1))

    return {
        "document_id":   doc_id,
        "document_date": doc_date,
        "document_year": doc_year,
    }