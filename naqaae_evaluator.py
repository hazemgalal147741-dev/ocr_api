"""
naqaae_evaluator.py
بيحل محل naqaae_client.py
بيقيّم الوثيقة على 12 معيار NAQAAE الرسمية عن طريق Groq
"""

import os
import json
import requests
from typing import Optional

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# المعايير الرسمية الـ 12 لاعتماد المؤسسات (دليل اعتماد كليات ومعاهد التعليم العالي،
# الهيئة القومية لضمان جودة التعليم والاعتماد NAQAAE، إصدار 2015 المعدل)
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
            print(f"[Groq/NAQAAE] HTTP {resp.status_code}: {resp.text[:500]}")
            return None
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Groq/NAQAAE] Exception: {e}")
        return None


def analyze_with_naqaae(text: str, model: str = DEFAULT_MODEL) -> dict:
    """
    نفس الـ interface القديم بتاع naqaae_client.py
    بيرجع: {"status", "score", "recs", "error", "domain_scores"}
    domain_scores دلوقتي بتغطي المعايير الـ 12 الرسمية لـ NAQAAE
    """
    if not text or not text.strip():
        return {"status": None, "score": None, "recs": None,
                "error": "⚠️ النص فاضي", "domain_scores": {}}

    # اختصار النص لو كبير
    words = text.split()
    if len(words) > 1200:
        text = " ".join(words[:1200])

    domains_str = "\n".join(f'  "{d}": <رقم من 0 لـ 100>' for d in DOMAINS)

    system = """أنت خبير تقييم مؤسسي متخصص في معايير الاعتماد المؤسسي الرسمية
للهيئة القومية لضمان جودة التعليم والاعتماد (NAQAAE) في مصر — دليل اعتماد
كليات ومعاهد التعليم العالي (إصدار 2015 المعدل)، وهو يتضمن 12 معيار رسمي.

مهمتك: تقييم الوثيقة المؤسسية على المعايير الـ 12 وإعطاء درجة لكل معيار.

قواعد التقييم:
- قيّم فقط ما هو موجود فعلاً في النص
- لو المعيار مش مذكور في النص → أعطه درجة منخفضة (10-30)
- لو المعيار مذكور بشكل سطحي → 30-60
- لو المعيار مفصّل مع أدلة → 60-85
- لو المعيار شامل ومتكامل → 85-100
- ملحوظة: معيار "الدراسات العليا" قد لا ينطبق على كل المؤسسات (كليات بدون
  برامج دراسات عليا)؛ في هذه الحالة قيّمه بدرجة متوسطة (50) واذكر ذلك في recs
  بدلاً من تصفيره بالكامل

أعد JSON فقط بهذا الشكل بالضبط، بدون أي نص خارجه:
{
  "domain_scores": {
""" + domains_str + """
  },
  "overall_score": <متوسط المعايير الـ 12>,
  "strengths": "<أهم نقطتين إيجابيتين في جملة واحدة>",
  "weaknesses": "<أهم نقطتين سلبيتين في جملة واحدة>"
}"""

    user_content = f"قيّم الوثيقة التالية وفق المعايير الـ 12 الرسمية لـ NAQAAE:\n\n{text}"

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