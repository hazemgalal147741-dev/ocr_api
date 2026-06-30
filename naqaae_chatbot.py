"""
naqaae_chatbot.py
شات بوت يتكلم باللهجة المصرية، بيجاوب على:
  1. أسئلة عن نتيجة تقييم وثيقة معينة (context بيتبعت في الـ request)
  2. أسئلة عامة عن معايير NAQAAE

Endpoint مستقل، بيتضاف في main.py
"""

import os
import json
import requests
from typing import Optional, List
from pydantic import BaseModel

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }


# ─── System Prompt — شخصية الشات بوت ──────────────────────────────────────────
SYSTEM_PROMPT = """انت "مساعد QualifAI" — شات بوت بتتكلم باللهجة المصرية العامية بس بشكل
محترم ومفهوم، زي ما اتنين زملاء بيتكلموا في الشغل. مش رسمي زيادة عن اللزوم، ومش
"كول" بشكل متكلف. هدفك تساعد المستخدم يفهم نتيجة تقييم وثيقته أو يعرف معلومات
عن معايير NAQAAE.

قواعد مهمة:
- اتكلم مصري طبيعي (مثلاً: "تمام"، "كويس"، "خد بالك"، "يعني"، "علشان")
  من غير ما تستخدم لهجات تانية (سعودي، شامي، إلخ)
- لو فيه نتيجة تقييم (evaluation context) موجودة معاك في الرسالة، استخدمها
  في إجابتك. لا تخترع أرقام أو معايير غير موجودة في الـ context.
- لو السؤال عن معيار معين من معايير NAQAAE الـ 12، اشرحه بشكل مبسط.
- لو مفيش context تقييم وسأل المستخدم عن "نتيجتي" أو "السكور بتاعي"،
  قوله بوضوح إنه محتاج يرفع وثيقة الأول أو يبعت نتيجة تقييم سابقة.
- خليك مختصر ومباشر — من 2 لـ 5 جمل في الرد العادي، إلا لو السؤال
  معقد ومحتاج شرح أطول.
- لو حد سأل سؤال بعيد تمامًا عن الموضوع (مش عن NAQAAE ولا عن نتيجة تقييمه)،
  رجّعه بلطف للموضوع: "أنا متخصص في أسئلة الاعتماد المؤسسي وتقييم الوثايق،
  لو عندك سؤال في الموضوع ده أنا تحت أمرك"
"""

DOMAIN_INFO = {
    "التخطيط الاستراتيجي": "يعني هل المؤسسة عندها خطة واضحة للمستقبل، فيها رؤية ورسالة وأهداف محددة وقابلة للقياس.",
    "القيادة والحوكمة": "ده عن هيكل الإدارة — مجلس الكلية، اللجان، وإزاي القرارات بتتاخد وتتوثق.",
    "إدارة الجودة والتطوير": "هل فيه نظام داخلي بيراجع الأداء بشكل دوري ويحسّنه.",
    "أعضاء هيئة التدريس والهيئة المعاونة": "مؤهلات الدكاترة والمعيدين، وهل فيه خطط لتطويرهم المهني.",
    "الجهاز الإداري": "الموظفين الإداريين وكفاءتهم في تسيير شغل الكلية.",
    "الموارد المالية والمادية": "الميزانية والمرافق — هل كافية وكويسة وموثقة بشفافية.",
    "المعايير الأكاديمية والبرامج التعليمية": "هل البرامج بتتراجع بشكل دوري ومتوافقة مع سوق العمل.",
    "التدريس والتعلم": "طرق التدريس المستخدمة وهل بتواكب احتياجات الطلاب.",
    "الطالب والخريجون": "خدمات الطلاب، الإرشاد الأكاديمي، ومعدلات توظيف الخريجين.",
    "البحث العلمي والأنشطة العلمية": "النشر العلمي والأبحاث اللي الكلية بتعملها.",
    "الدراسات العليا": "برامج الماجستير والدكتوراه لو موجودة في الكلية.",
    "المشاركة المجتمعية وتنمية البيئة": "علاقة الكلية بالمجتمع والشراكات مع سوق العمل.",
}


class ChatMessage(BaseModel):
    role: str       # "user" أو "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    evaluation_context: Optional[dict] = None
    # evaluation_context الشكل المتوقع (اختياري بالكامل):
    # {
    #   "overall_score": 72.5,
    #   "status": "معتمد",
    #   "domain_scores": {...},
    #   "strengths": "...",
    #   "weaknesses": "..."
    # }


class ChatResponse(BaseModel):
    reply: str
    error: Optional[str] = None


def _build_context_block(ctx: Optional[dict]) -> str:
    """يحول الـ evaluation context لنص يتفهمه الشات بوت"""
    if not ctx:
        return "لا توجد نتيجة تقييم متاحة حالياً في هذا السياق."

    lines = ["نتيجة التقييم المتاحة للوثيقة الحالية:"]
    if "overall_score" in ctx:
        lines.append(f"- الدرجة الكلية: {ctx['overall_score']} / 100")
    if "status" in ctx:
        lines.append(f"- الحالة: {ctx['status']}")
    if ctx.get("domain_scores"):
        lines.append("- درجات المعايير:")
        for k, v in ctx["domain_scores"].items():
            lines.append(f"    • {k}: {v}")
    if ctx.get("strengths"):
        lines.append(f"- نقاط القوة: {ctx['strengths']}")
    if ctx.get("weaknesses"):
        lines.append(f"- نقاط الضعف: {ctx['weaknesses']}")

    return "\n".join(lines)


def _call_groq_chat(messages: list, model: str = DEFAULT_MODEL) -> Optional[str]:
    try:
        resp = requests.post(
            GROQ_API_URL,
            json={
                "model":       model,
                "messages":    messages,
                "temperature": 0.6,   # أعلى من التقييم — عايزينه طبيعي ومش جامد
                "max_tokens":  600,
            },
            headers=_headers(),
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"[Groq/Chat] HTTP {resp.status_code}: {resp.text[:300]}")
            return None
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Groq/Chat] Exception: {e}")
        return None


def chat_with_assistant(req: ChatRequest, model: str = DEFAULT_MODEL) -> ChatResponse:
    """
    الدالة الأساسية للشات بوت.
    بتدمج: system prompt + سياق التقييم (لو موجود) + الـ history + رسالة المستخدم الجديدة
    """
    if not req.message or not req.message.strip():
        return ChatResponse(reply="", error="⚠️ الرسالة فاضية")

    context_block = _build_context_block(req.evaluation_context)

    full_system = f"{SYSTEM_PROMPT}\n\n══════════════\n{context_block}\n══════════════"

    messages = [{"role": "system", "content": full_system}]

    # ضيف الـ history لو موجود (آخر 10 رسايل بس عشان الـ context يفضل خفيف)
    for h in (req.history or [])[-10:]:
        if h.role in ("user", "assistant"):
            messages.append({"role": h.role, "content": h.content})

    messages.append({"role": "user", "content": req.message})

    reply = _call_groq_chat(messages, model=model)

    if not reply:
        return ChatResponse(
            reply="معلش، حصل مشكلة وأنا بحاول أرد عليك. جرب تاني كمان شوية.",
            error="⚠️ تعذر الاتصال بـ Groq"
        )

    return ChatResponse(reply=reply)
