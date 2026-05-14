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
from .rag_engine import retrieve_medical_context

SYSTEM_PROMPT = """You are a senior Medical Advisor AI. Answer directly, concisely, and without evasion.

YOUR KNOWLEDGE SOURCE:
You are provided with "Verified Medical Reference Data" for the most relevant conditions. You MUST prioritize the OTC dosages and self-care steps found in this data if it matches the patient's symptoms.

RULES:
- Direct answers only. No filler.
- State the most probable diagnosis (1 sentence).
- Always tell the patient WHAT TO DO: specific OTC medication with name + dosage, and any relevant self-care steps (rest, hydration, diet, etc.).
- NEVER suggest prescription drugs.
- Do NOT ask about scheduling appointments — another agent handles that.
- Append exactly one line at the end: "I am an AI, not a licensed doctor."

TREATMENT PATH DECISION — choose the most appropriate path:
- "otc" : Use ONLY when the condition is a minor, self-limiting illness (e.g., mild common cold, minor headache, superficial scrape) that clearly does not require a formal medical evaluation. Provide actionable OTC advice + self-care.

- "doctor_needed" : Use whenever symptoms warrant a professional medical evaluation, formal diagnosis, potential prescription therapy, OR if there are moderate to severe concerns (e.g., persistent symptoms, suspected infections like strep or UTI that typically require antibiotics, worsening conditions, high fever, undetermined rashes). 
  * CRITICAL: Even when selecting "doctor_needed", you MUST still suggest safe interim OTC medications and self-care to manage symptoms until the patient can be seen by the doctor.

- "emergency" : Immediate life threat only (chest pain, stroke signs, anaphylaxis, unconsciousness, severe bleeding). Direct patient to call 101 immediately.

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
    symptoms = triage.get('symptoms', [])
    medical_context = retrieve_medical_context(symptoms)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{medical_context}\n\n"
                f"Patient Case:\n"
                f"Symptoms: {', '.join(symptoms)}\n"
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
