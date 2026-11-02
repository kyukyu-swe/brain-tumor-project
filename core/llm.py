"""
LLM explanations with Gemini.

Key safety idea: the LLM NEVER sees the image. It only gets
  (1) the model's prediction + confidence, and
  (2) checked facts from tumor_facts.json,
and it is told to use ONLY those facts. This limits hallucination.

If there is no API key (or the API fails), we fall back to a plain
English text built directly from the facts, so the apps still work.
"""
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

FACTS_PATH = Path(__file__).with_name("tumor_facts.json")
FACTS = json.loads(FACTS_PATH.read_text(encoding="utf-8"))

DEFAULT_MODEL = "gemini-flash-latest"
LANGUAGES = ["English", "Thai", "Burmese (Myanmar)", "Japanese"]
MODES = ["patient", "student", "doctor_note"]

SYSTEM_RULES = """You are a careful medical-education assistant inside a student project demo.
Rules you must follow:
1. Use ONLY the facts given in FACTS. Do not add other medical facts, numbers, survival rates or drug names.
2. Never say the person HAS a disease. Say what the AI model predicted, and that only a doctor can diagnose.
3. Never give a treatment decision or dosage. You may describe the usual next steps listed in FACTS.
4. If the model is unsure (refer = true), say clearly that the result is uncertain and a specialist should check it.
5. Write in simple, common words. Short sentences. No jargon unless you explain it.
6. Write the whole answer in the requested LANGUAGE.
"""

MODE_INSTRUCTIONS = {
    "patient": """AUDIENCE: a patient or family member who is worried and is not a medical expert.
Write 150-250 words with these parts:
- What the AI result means (1-2 sentences, gentle tone)
- What this type of tumor is, in simple words
- Usual next steps
- 3-5 questions to ask the doctor (as a list)
- End with one sentence: this is not a diagnosis; please talk to your doctor.""",

    "student": """AUDIENCE: a medical student learning to read brain MRI.
Write 120-200 words:
- Name the TRUE class and explain the typical visual clues on MRI for it
- Where this tumor type usually appears
- If the student's guess was wrong, explain briefly how to tell the guessed class and the true class apart
- If the AI was wrong, mention it as a reminder that AI makes mistakes""",

    "doctor_note": """AUDIENCE: a doctor reviewing a triage list.
Write a short draft note (max 80 words), neutral and factual:
- AI prediction with probability, P(tumor), priority
- Whether the safety rule flagged it for review and why
- Mark clearly as 'AI draft, not a report; requires radiologist review'.""",
}


# ---- Cost protection for a public app: max Gemini calls per hour (all users together) ----
MAX_CALLS_PER_HOUR = int(os.getenv("MAX_LLM_CALLS_PER_HOUR", "100"))
_calls = deque()
_lock = threading.Lock()


def _llm_allowed():
    now = time.time()
    with _lock:
        while _calls and now - _calls[0] > 3600:
            _calls.popleft()
        if len(_calls) >= MAX_CALLS_PER_HOUR:
            return False
        _calls.append(now)
        return True


def _facts_for(label, extra_labels=()):
    keys = [label, *[l for l in extra_labels if l and l != label]]
    return {k: FACTS[k] for k in keys if k in FACTS}


def _build_prompt(result, language, mode, student_guess=None, true_label=None):
    info = {
        "ai_prediction": result.label,
        "ai_confidence": round(result.confidence, 3),
        "all_probabilities": {k: round(v, 3) for k, v in result.probs.items()},
        "p_tumor": round(result.p_tumor, 3),
        "priority": result.priority,
        "refer": result.refer,
        "refer_reasons": result.reasons,
    }
    if mode == "student":
        info["true_label"] = true_label
        info["student_guess"] = student_guess
    facts = _facts_for(true_label or result.label, [result.label, student_guess])
    return (
        f"LANGUAGE: {language}\n\n{MODE_INSTRUCTIONS[mode]}\n\n"
        f"MODEL_OUTPUT:\n{json.dumps(info, ensure_ascii=False, indent=2)}\n\n"
        f"FACTS:\n{json.dumps(facts, ensure_ascii=False, indent=2)}"
    )


def fallback_text(result, mode="patient", true_label=None):
    """Plain English text from the facts only (used when the LLM is not available)."""
    key = true_label or result.label
    f = FACTS[key]
    lines = []
    if mode == "student":
        lines += [f"**True class: {f['name']}**", "", f"**Visual clues:** {f['visual_clues_on_mri']}",
                  f"**Usual location:** {f['usual_location']}"]
    elif mode == "doctor_note":
        lines += [f"AI draft (not a report): predicted {result.display_name} "
                  f"({result.confidence:.0%}), P(tumor) {result.p_tumor:.0%}, priority {result.priority}."]
        lines += ["Flagged for review: " + " ".join(result.reasons)] if result.refer else []
        lines += ["Requires radiologist review."]
    else:
        lines += [f"**AI result: {f['name']}** ({result.confidence:.0%} confidence)", "",
                  f["what_it_is"], "", f["behavior"], "", f"**Usual next steps:** {f['usual_next_steps']}",
                  "", "**Questions to ask your doctor:**", *[f"- {q}" for q in f["questions_for_doctor"]], "",
                  "_This is not a diagnosis. Please talk to your doctor._"]
        if result.refer:
            lines.insert(1, "⚠️ The AI is not sure about this result. A specialist should check the scan.")
    return "\n".join(lines)


def explain(result, language="English", mode="patient", student_guess=None, true_label=None):
    """Return an explanation string. Falls back to fact-based text if Gemini is unavailable."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback_text(result, mode, true_label) + "\n\n_(LLM off: no GEMINI_API_KEY in .env)_"
    if not _llm_allowed():
        return fallback_text(result, mode, true_label) + "\n\n_(LLM hourly limit reached, showing basic facts instead.)_"
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
            contents=_build_prompt(result, language, mode, student_guess, true_label),
            config=types.GenerateContentConfig(system_instruction=SYSTEM_RULES, temperature=0.3),
        )
        return response.text.strip()
    except Exception as e:  # network error, quota, wrong model name, ...
        return fallback_text(result, mode, true_label) + f"\n\n_(LLM error, showing basic facts instead: {e})_"
