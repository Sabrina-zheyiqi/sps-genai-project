from typing import Literal, Tuple, List
import re

SafetyLevel = Literal["safe", "warning", "emergency"]


def split_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter for English + Chinese.
    We lowercase everything to simplify matching.
    """
    text_lower = text.lower()
    parts = re.split(r"[\.!\?\n。！？]+", text_lower)
    return [p.strip() for p in parts if p.strip()]


NEG_WORDS = ["no ", "without ", "denies ", "haven't ", "hasn't ", "没有", "無", "无", "沒有"]


def sentence_has_negation(sent: str) -> bool:
    """Return True if the sentence contains a generic negation word."""
    return any(neg in sent for neg in NEG_WORDS)


def safety_check(text: str) -> Tuple[SafetyLevel, str]:
    """
    Safety layer v5:

    - Emergency:
        * chest pain + (radiating OR shortness of breath OR sweating) in the SAME
          sentence and WITHOUT negation words.
        * severe breathing like "can't breathe".
        * GI bleeding keywords in a sentence WITHOUT negation.
        * stroke-like symptoms.
        * suicidal ideation.
    - Warning:
        * single concerning symptoms (shortness of breath, blood in stool, etc.),
          regardless of negation (for simplicity).
    - Safe:
        * everything else.
    """
    t = text.lower()
    sentences = split_sentences(text)

    # =============================
    # 1. HEART ATTACK CLUSTER
    # =============================
    for s in sentences:
        chest_pain = ("chest pain" in s) or ("胸痛" in s)
        if chest_pain and not sentence_has_negation(s):
            radiating = any(
                k in s
                for k in [
                    "left arm",
                    "jaw",
                    "radiat",
                    "back",
                    "shoulder",
                    "左臂",
                    "下颌",
                    "后背",
                    "肩",
                ]
            )
            sob = any(
                k in s for k in ["shortness of breath", "short of breath", "气短", "呼吸困难"]
            )
            sweating = any(k in s for k in ["sweating", "cold sweat", "大汗"])

            if radiating or sob or sweating:
                return (
                    "emergency",
                    "Detected severe chest pain with concerning features. "
                    "Call emergency services immediately.",
                )

    # =============================
    # 2. SEVERE BREATHING
    # =============================
    if any(k in t for k in ["can't breathe", "cannot breathe", "unable to breathe"]) or any(
        k in text for k in ["喘不过来", "呼吸不过来", "严重呼吸困难"]
    ):
        return (
            "emergency",
            "Detected severe breathing difficulty. Call emergency services immediately.",
        )

    # =============================
    # 3. GI BLEEDING (sentence-level + negation)
    # =============================
    gi_keywords = ["vomiting blood", "bloody vomit", "black stool", "tarry stool", "呕血", "黑便"]

    for s in sentences:
        if any(k in s for k in gi_keywords) and not sentence_has_negation(s):
            return (
                "emergency",
                "Detected possible gastrointestinal bleeding. Please seek emergency care immediately.",
            )

    # =============================
    # 4. STROKE
    # =============================
    stroke_terms_en = ["sudden weakness on one side", "face drooping", "slurred speech"]
    stroke_terms_zh = ["突然说不出话", "一侧肢体无力", "口角歪斜"]

    if any(k in t for k in stroke_terms_en) or any(k in text for k in stroke_terms_zh):
        return (
            "emergency",
            "Detected possible stroke symptoms. Call emergency services immediately.",
        )

    # =============================
    # 5. SUICIDAL IDEATION
    # =============================
    if any(k in t for k in ["kill myself", "end my life", "suicide"]) or any(
        k in text for k in ["自杀", "想死"]
    ):
        return (
            "emergency",
            "Detected suicidal thoughts. Immediate help is required. "
            "Contact emergency services or a crisis hotline.",
        )

    # =============================
    # 6. WARNING (non-emergency concern)
    # =============================
    warning_terms_en = [
        "shortness of breath",
        "short of breath",
        "blood in stool",
        "black stool",
        "vomiting blood",
        "unintentional weight loss",
        "severe pain",
        "chest tightness",
    ]
    warning_terms_zh = ["气短", "便血", "体重下降", "胸闷"]

    has_warning = any(w in t for w in warning_terms_en) or any(
        w in text for w in warning_terms_zh
    )

    if has_warning:
        return (
            "warning",
            "Detected potentially concerning symptoms. Please seek medical evaluation soon.",
        )

    # =============================
    # 7. SAFE (default)
    # =============================
    return (
        "safe",
        "No emergency features detected. This tool provides general health information only.",
    )
