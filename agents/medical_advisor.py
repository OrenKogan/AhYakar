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

SYSTEM_PROMPT = """You are a senior Medical Advisor AI. Answer directly, concisely, and without evasion.

RULES:
- Direct answers only. No filler.
- State the most probable diagnosis (1 sentence).
- Always tell the patient WHAT TO DO: specific OTC medication with name + dosage, and any relevant self-care steps (rest, hydration, diet, etc.).
- NEVER suggest prescription drugs.
- Do NOT ask about scheduling appointments — another agent handles that.
- Append exactly one line at the end: "I am an AI, not a licensed doctor."

TREATMENT PATH DECISION — be strict:
- "otc" : Use this for the MAJORITY of cases. Use it when: the condition is a common illness
  (cold, flu, sore throat, headache, mild fever, stomach ache, diarrhea, skin rash, UTI symptoms,
  muscle pain, allergies, ear pain, eye irritation, etc.) AND there are no red flags below.
  Always give actionable OTC advice + self-care steps.

- "doctor_needed" : Use ONLY when at least one clear red flag is present:
    * Symptoms persisting >7 days with no improvement despite OTC treatment
    * Fever >39.5°C lasting more than 48h, or any fever in infants <3 months
    * Severe or worsening pain that OTC meds cannot manage
    * Blood in urine, stool, vomit, or sputum
    * Difficulty breathing or swallowing
    * Sudden neurological symptoms (confusion, numbness, vision changes, severe headache)
    * Suspected fracture, deep wound, or injury requiring assessment
    * Symptoms strongly suggesting a condition that requires diagnosis (e.g. appendicitis, heart issue)
    * Patient mentions a chronic condition is worsening beyond usual management

- "emergency" : Immediate life threat only (chest pain, stroke signs, anaphylaxis, unconsciousness,
  severe bleeding). Direct patient to call 112/911 immediately.

You MUST respond with ONLY a valid JSON object — no markdown, no explanation:
{
  "diagnosis": "Probable diagnosis in 1 sentence",
  "action": "otc | doctor_needed | emergency",
  "reply": "Short, direct message. Always include: (1) diagnosis, (2) what to do — specific OTC medication with name + dosage AND self-care steps. If doctor_needed: add which specialist and why a doctor is needed. If emergency: call 112/911 + reason. Last line: I am an AI, not a licensed doctor.",
  "otc_medications": ["OTC medication with dosage, e.g. 'Ibuprofen 400mg every 8h with food, max 3 doses/day'"],
  "specialist_type": "e.g. General Practitioner, Dermatologist, ENT — only if doctor_needed, else null",
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
