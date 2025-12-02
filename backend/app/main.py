import json
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .models import AskRequest, AskResponse, SafetyResult, SeverityInfo
from .safety import safety_check
from .prompts import build_prompt
from .inference import call_llm

# Base paths for static files
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Medical Assistant (LLM + Safety Layer)",
    description=(
        "A demo medical Q&A assistant with multiple task types and a simple safety layer, "
        "implemented using FastAPI and a Hugging Face-hosted LLM."
    ),
    version="0.2.0",
)

# Enable CORS for frontend usage (you can restrict origins in production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend's origin.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files under /static instead of /
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint for Docker verification.
    This will always return {"status": "ok"} when the server is running.
    """
    return {"status": "ok"}


@app.get("/")
async def serve_index() -> FileResponse:
    """
    Serve the main HTML page for the web UI.
    """
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)


def parse_severity_json(raw_output: str) -> tuple[str, Optional[SeverityInfo]]:
    """
    Split the raw model output into:
        - human-readable answer text
        - structured severity info (if JSON is present)

    The model is instructed to output:
        ...normal answer...

        ###JSON###{"severity": "...", ...}

    If parsing fails, the entire output is treated as answer text and
    severity_info is None.
    """
    marker = "###JSON###"
    idx = raw_output.find(marker)
    if idx == -1:
        # No JSON marker found, return the whole text as the answer.
        return raw_output.strip(), None

    answer_text = raw_output[:idx].strip()
    json_candidate = raw_output[idx + len(marker):].strip()

    # Try to extract the first {...} block.
    start = json_candidate.find("{")
    end = json_candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return answer_text, None

    json_str = json_candidate[start: end + 1]

    try:
        data: Dict[str, Any] = json.loads(json_str)
    except Exception:
        return answer_text, None

    severity = data.get("severity")
    recommended_action = data.get("recommended_action")
    time_window = data.get("time_window")
    risk_notes = data.get("risk_notes")

    severity_info = SeverityInfo(
        severity=severity,
        recommended_action=recommended_action,
        time_window=time_window,
        risk_notes=risk_notes,
    )
    return answer_text, severity_info


@app.post("/api/ask", response_model=AskResponse)
async def ask_medical(request: AskRequest) -> AskResponse:
    """
    Main endpoint for medical-style questions.

    Flow:
        1. Run a keyword-based safety check on the raw user question.
        2. If the safety check classifies it as an emergency:
              - return an emergency warning and do NOT call the LLM.
        3. Otherwise:
              - build a task-specific prompt (medical Q&A, diagnosis, drug, lab, education),
              - call the LLM with optional temperature and max_tokens overrides,
              - parse the JSON severity block (if present),
              - return the answer plus the safety info and severity info.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Safety layer
    level, safety_msg = safety_check(question)
    safety_result = SafetyResult(level=level, message=safety_msg)

    # 2. If emergency -> short-circuit and do not call the LLM.
    if level == "emergency":
        return AskResponse(
            answer=(
                "Your description contains signs that may indicate a medical emergency.\n\n"
                "⚠ Please call your local emergency number or go to the nearest emergency "
                "department immediately.\n\n"
                "For safety reasons, this system will not provide further online analysis "
                "or advice for potential emergency situations."
            ),
            safety=safety_result,
            severity=None,
            used_prompt=None,
        )

    # 3. Build a prompt adapted to the requested task type and language.
    prompt = build_prompt(
        task_type=request.task_type,
        user_question=question,
        language=request.language,
    )

    # 4. Determine temperature and max_tokens from the request (with sane defaults).
    temperature = request.temperature if request.temperature is not None else 0.2
    max_tokens = request.max_tokens if request.max_tokens is not None else 512

    # Clamp values to safe ranges.
    temperature = max(0.0, min(temperature, 1.5))
    max_tokens = max(64, min(max_tokens, 1024))

    # 5. Call the LLM.
    try:
        raw_output = call_llm(prompt, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM: {e}")

    # 6. Split into human-readable answer + severity JSON.
    answer_text, severity_info = parse_severity_json(raw_output)

    return AskResponse(
        answer=answer_text,
        safety=safety_result,
        severity=severity_info,
        used_prompt=prompt,
    )
