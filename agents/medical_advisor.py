"""
Agent 2 — Medical Advisor
=========================
Deep clinical analysis agent. Only invoked when the Intake Analyst has
determined that the patient's condition is too serious for simple home remedies.

Responsibilities:
1. Fully and deeply analyze the symptoms and medical history.
2. Form a probable diagnosis.
3. Decide the treatment path:
   - OTC medicine  → recommend it concisely and STOP the pipeline.
   - Doctor needed → escalate to Appointment Planner.
   - Emergency     → direct to emergency services and STOP the pipeline.
"""

import json
from openai import OpenAI
from .utils import parse_json_response

SYSTEM_PROMPT = """You are a senior Medical Advisor AI. You receive a patient's symptoms and medical history that have already been triaged as requiring more than simple home care.

YOUR JOB:
1. Perform a thorough, deep analysis of the symptoms and history.
2. Arrive at the most likely diagnosis (or differential diagnoses if uncertain).
3. Decide the appropriate treatment path.

TREATMENT PATH RULES:
- "otc"           : The condition can be safely managed with Over-The-Counter medication.
                    Give a concise recommendation (short sentences). NEVER suggest prescription drugs.
- "doctor_needed" : The condition requires professional medical evaluation — it's too complex,
                    persistent, severe, or risky to treat without a doctor.
- "emergency"     : Immediate danger to life. Direct the patient to call emergency services NOW.

IMPORTANT:
- Always state you are an AI and not a licensed doctor.
- Be empathetic but medically precise.
- If you recommend OTC medication, be specific (e.g. "Paracetamol 500mg every 6h, max 4 doses/day").
- If escalating to a doctor, specify the most appropriate specialist type.

You MUST respond with ONLY a valid JSON object — no markdown, no explanation:
{
  "diagnosis": "Your most likely diagnosis or differential (1–2 sentences)",
  "action": "otc | doctor_needed | emergency",
  "reply": "The message to show the patient. If otc: concise advice + medication names/dosage. If doctor_needed: brief explanation of why a doctor is needed. If emergency: direct them to call 112/911 immediately.",
  "otc_medications": ["specific OTC medication with dosage if action is otc, else empty array"],
  "specialist_type": "e.g. General Practitioner, Dermatologist, Cardiologist, ENT — only if action is doctor_needed, else null",
  "urgency": "immediately | within_24h | this_week | whenever — only if doctor_needed"
}
"""


def advise(client: OpenAI, triage: dict) -> dict:
    """
    Run the Medical Advisor agent.

    Args:
        client : OpenAI-compatible client.
        triage : Triage dict from Agent 1 (symptoms, duration, history).

    Returns:
        An advice dict with 'action' set to 'otc', 'doctor_needed', or 'emergency'.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Symptoms: {', '.join(triage.get('symptoms', []))}\n"
                f"Duration: {triage.get('duration', 'unknown')}\n"
                f"Medical history / notes: {triage.get('relevant_history', 'none')}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=messages,
        extra_headers={
            "HTTP-Referer": "http://localhost:5000",
            "X-OpenRouter-Title": "AhYakar - Medical Advisor",
        },
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    try:
        return parse_json_response(raw)
    except (json.JSONDecodeError, ValueError):
        return {
            "diagnosis": "Unable to determine diagnosis.",
            "action": "doctor_needed",
            "reply": "I wasn't able to fully analyze your situation. To be safe, I recommend seeing a doctor.",
            "otc_medications": [],
            "specialist_type": "General Practitioner",
            "urgency": "this_week",
        }
