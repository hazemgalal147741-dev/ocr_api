"""
naqaae_client.py
يكلم Gradio Space بتاع NAQAAE عن طريق gradio_client
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
SPACE_ID  = "hazemgalal1/naqaae-demo"

def _make_client():
    from gradio_client import Client
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else None
    return Client(SPACE_ID, headers=headers)


def analyze_with_naqaae(text: str) -> dict:
    if not text or not text.strip():
        return {"status": None, "score": None, "recs": None,
                "error": "⚠️ النص فاضي"}

    words = text.split()
    if len(words) > 1500:
        text = " ".join(words[:1500])

    try:
        client = _make_client()
        result = client.predict(text, api_name="/full_analyze")

        if isinstance(result, (list, tuple)) and len(result) >= 3:
            status = str(result[0])
            try:
                score = float(result[1])
            except Exception:
                score = 0.0
            recs = str(result[2])
            return {"status": status, "score": score, "recs": recs, "error": None}

        return {"status": str(result), "score": 0.0, "recs": "", "error": None}

    except Exception as e:
        err = str(e)
        if "403" in err or "allowlist" in err.lower():
            return {"status": None, "score": None, "recs": None,
                    "error": "⚠️ الـ Space مش بيسمح بالـ API — تأكد من api_open=True"}
        if "404" in err:
            return {"status": None, "score": None, "recs": None,
                    "error": "⚠️ الـ endpoint مش موجود"}
        return {"status": None, "score": None, "recs": None,
                "error": f"⚠️ خطأ: {err[:200]}"}


def check_space_status() -> bool:
    try:
        _make_client()
        return True
    except Exception:
        return False