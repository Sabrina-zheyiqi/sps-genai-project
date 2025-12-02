from typing import Literal

# The supported task types in your application.
TaskType = Literal["medical_qa", "diagnosis", "drug", "lab", "education"]


BASE_SYSTEM_PROMPT = """You are a cautious, evidence-informed medical-style assistant
powered by a general large language model (e.g., Llama 3 8B).
You are NOT a doctor and you cannot make definitive diagnoses or prescribe medications.

You must always:
- Emphasize that your answer is general information, not medical advice.
- Encourage the user to see a healthcare professional for any serious or persistent symptoms.
- Be concise, well-structured, and easy to understand.
- Never invent lab values, medications, or detailed treatment plans.
"""


def build_prompt(task_type: TaskType, user_question: str, language: str = "zh") -> str:
    """
    Build a task-specific prompt for the LLM.

    language:
        - "zh": answer in Simplified Chinese
        - "en": answer in English

    IMPORTANT:
        At the end of the answer, the model must output a single-line JSON
        block prefixed with '###JSON###' describing the severity assessment.
    """
    if language == "zh":
        lang_instruction = (
            "Please answer in Simplified Chinese (简体中文), unless the user clearly "
            "uses another language."
        )
    else:
        lang_instruction = "Please answer in English."

    if task_type == "medical_qa":
        task_instruction = (
            "Task: Provide general medical information and suggestions based on the user's question. "
            "Focus on explaining possible causes, typical work-up, and when to see a doctor."
        )
    elif task_type == "diagnosis":
        task_instruction = (
            "Task: Provide diagnostic-style reasoning (chain-of-thought). Explain possible causes "
            "and differential diagnoses, but clearly state that this is NOT a formal diagnosis and "
            "that only a licensed clinician can diagnose and treat."
        )
    elif task_type == "drug":
        task_instruction = (
            "Task: Provide information about medications (indications, common side effects, "
            "precautions, interactions). Do NOT prescribe any medications. Always remind the user "
            "to consult a doctor or pharmacist before taking or changing medicines."
        )
    elif task_type == "lab":
        task_instruction = (
            "Task: Provide a general interpretation of lab or imaging results. Explain what the "
            "values or findings might mean, possible causes, and when further evaluation is needed. "
            "Do not make definitive diagnoses."
        )
    elif task_type == "education":
        task_instruction = (
            "Task: Provide health education and lifestyle advice (prevention, long-term management, "
            "self-care). Keep the advice practical, realistic, and conservative."
        )
    else:
        task_instruction = (
            "Task: Provide general medical-style information based on the user's question."
        )

    severity_instruction = """
After you finish the full human-readable answer, on a new line output:
###JSON### followed by a single-line JSON object with this exact schema:

{
  "severity": "low" | "moderate" | "high",
  "recommended_action": "self_care" | "outpatient" | "emergency",
  "time_window": "short free-text description of when the user should seek care",
  "risk_notes": "one short sentence summarizing why you chose this severity"
}

Do NOT explain the JSON. Do NOT add extra text after the JSON. The JSON must be valid.
"""

    full_prompt = f"""{BASE_SYSTEM_PROMPT}

{lang_instruction}

{task_instruction}

User question:
\"\"\"{user_question}\"\"\"

Now provide your answer in a clear, structured format, with headings and bullet points when helpful.
{severity_instruction}
"""

    return full_prompt
