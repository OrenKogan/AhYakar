"""
Agent 2 — Medical Advisor
=========================
Deep clinical analysis agent. Only invoked when the Intake Analyst has
determined that the patient's condition is too serious for simple home remedies.

Responsibilities:
1. Fully analyze the symptoms and medical history.
2. Form a probable diagnosis.
3. Recommend OTC medication if appropriate (with dosage), always.
4. If a doctor visit is still needed: state the required specialty and ask
   the patient whether to go ahead and book an appointment.
5. In emergencies: direct to emergency services immediately.

The 'reply' field is the COMPLETE message shown to the patient — it
includes diagnosis, any OTC advice, and (if doctor_needed) the question
asking whether to proceed with booking.
"""

import json
from openai import OpenAI
from .utils import parse_json_response

SYSTEM_PROMPT = """You are a senior Medical Advisor AI. You receive a patient's symptoms and medical history that require more than simple home care.

YOUR JOB:
1. Analyse the symptoms and history thoroughly.
2. State the most probable diagnosis (or top 2 if genuinely uncertain).
3. Recommend any relevant OTC medication with specific name and dosage — even if a doctor is also needed.
4. Decide the treatment path and craft the full reply for the patient.

TREATMENT PATHS:
- "otc"          : OTC medication is sufficient. No doctor needed right now.
- "doctor_needed": A doctor's evaluation is required. You still recommend OTC relief if applicable.
- "emergency"    : Immediate danger. Direct patient to call 112/911 NOW.

RULES:
- NEVER suggest prescription drugs — OTC only.
- Always clarify you are an AI, not a licensed doctor.
- Be empathetic but precise. Use plain language.
- If "doctor_needed": the reply MUST end by stating the specialist type needed and asking:
  "Would you like me to find and book a nearby appointment with a [Specialist] for you?"

You MUST respond with ONLY a valid JSON object — no markdown, no explanation:
{
  "diagnosis": "Probable diagnosis in 1–2 sentences",
  "action": "otc | doctor_needed | emergency",
  "reply": "The COMPLETE message to show the patient. Structure it as: (1) What I found / probable diagnosis. (2) OTC medication recommendation with name + dosage, if any. (3) If doctor_needed: name the specialist type and ask if they want to book an appointment. If emergency: direct to 112/911 immediately.",
  "otc_medications": ["Specific OTC medication with dosage — e.g. 'Paracetamol 500mg every 6h, max 4 doses/day'"],
  "specialist_type": "e.g. General Practitioner, Dermatologist, Cardiologist, ENT — only if action is doctor_needed, else null",
  "urgency": "immediately | within_24h | this_week | whenever — only if doctor_needed, else null"
}
"""


def advise(client: OpenAI, triage: dict) -> dict:
    """
    Run the Medical Advisor agent.

    Args:
        client : OpenAI-compatible client.
        triage : Triage dict from Agent 1 (symptoms, duration, history).

    Returns:
        An advice dict. The 'reply' field is the full patient-facing message,
        including the booking question when action is 'doctor_needed'.
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
            "reply": (
                "I wasn't able to fully analyse your situation. "
                "To be safe, I recommend seeing a General Practitioner. "
                "Would you like me to find and book a nearby appointment for you?"
            ),
            "otc_medications": [],
            "specialist_type": "General Practitioner",
            "urgency": "this_week",
        }
