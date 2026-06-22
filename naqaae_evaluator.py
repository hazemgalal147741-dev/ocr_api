"""
naqaae_evaluator.py
بيحل محل naqaae_client.py
بيقيّم الوثيقة على 9 محاور NAQAAE عن طريق Groq
"""

import os
import json
import requests
from typing import Optional

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
DEFAULT_MODEL = "llama-3.3-70b-versatile"

DOMAINS = [
    "الرؤية والرسالة والأهداف",
    "الحوكمة والقيادة المؤسسية",
    "البرامج الأكاديمية وضمان الجودة",
    "هيئة التدريس والبحث العلمي",
    "الطلاب وخدماتهم",
    "البنية التحتية والمرافق",
    "الموارد المالية والإدارية",
    "المجتمع وسوق العمل",
    "نظم المعلومات والتكنولوجيا",
]


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }


def _call_groq(messages: list, model: str = DEFAULT_MODEL) -> Optional[str]:
    try:
        resp = requests.post(
            GROQ_API_URL,
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            headers=_headers(),
            timeout=120,
        )
        if resp.status_code != 200:
            return None
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def analyze_with_naqaae(text: str, model: str = DEFAULT_MODEL) -> dict:
    """
    نفس الـ interface القديم بتاع naqaae_client.py
    بيرجع: {"status", "score", "recs", "error", "domain_scores"}
    """
    if not text or not text.strip():
        return {"status": None, "score": None, "recs": None,
                "error": "⚠️ النص فاضي", "domain_scores": {}}

    # اختصار النص لو كبير
    words = text.split()
    if len(words) > 1200:
        text = " ".join(words[:1200])

    domains_str = "\n".join(f'  "{d}": <رقم من 0 لـ 100>' for d in DOMAINS)

    system = """أنت خبير تقييم مؤسسي متخصص في معايير NAQAAE المصرية.
مهمتك: تقييم الوثيقة المؤسسية على 9 محاور وإعطاء درجة لكل محور.

قواعد التقييم:
- قيّم فقط ما هو موجود فعلاً في النص
- لو المحور مش مذكور في النص → أعطه درجة منخفضة (10-30)
- لو المحور مذكور بشكل سطحي → 30-60
- لو المحور مفصّل مع أدلة → 60-85
- لو المحور شامل ومتكامل → 85-100

أعد JSON فقط بهذا الشكل بالضبط، بدون أي نص خارجه:
{
  "domain_scores": {
""" + domains_str + """
  },
  "overall_score": <متوسط الـ 9 درجات>,
  "strengths": "<أهم نقطتين إيجابيتين في جملة واحدة>",
  "weaknesses": "<أهم نقطتين سلبيتين في جملة واحدة>"
}"""

    user_content = f"قيّم الوثيقة التالية:\n\n{text}"

    raw = _call_groq(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        model=model,
    )

    if not raw:
        return {"status": "غير معروف", "score": 0.0, "recs": "",
                "error": "⚠️ فشل الاتصال بـ Groq", "domain_scores": {}}

    # Parse JSON
    try:
        clean = raw.strip()
        # إزالة markdown لو موجود
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean.strip())
    except Exception:
        return {"status": "غير معروف", "score": 0.0, "recs": raw,
                "error": None, "domain_scores": {}}

    # استخراج النتايج
    domain_scores = data.get("domain_scores", {})

    # حساب الـ overall score
    raw_score = data.get("overall_score")
    if raw_score:
        score = float(raw_score)
    elif domain_scores:
        vals = [v for v in domain_scores.values() if isinstance(v, (int, float))]
        score = sum(vals) / len(vals) if vals else 0.0
    else:
        score = 0.0

    score = round(max(0.0, min(score, 100.0)), 1)

    # تحديد الـ status
    if score >= 70:
        status = "معتمد"
    elif score >= 50:
        status = "مؤجل"
    else:
        status = "غير معتمد"

    # التوصيات
    strengths  = data.get("strengths", "")
    weaknesses = data.get("weaknesses", "")
    recs = f"نقاط القوة: {strengths}\nنقاط الضعف: {weaknesses}" if strengths or weaknesses else ""

    return {
        "status":       status,
        "score":        score,
        "recs":         recs,
        "error":        None,
        "domain_scores": domain_scores,
    }


def check_space_status() -> bool:
    """للتوافق مع الكود القديم"""
    return bool(GROQ_API_KEY)