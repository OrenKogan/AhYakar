"""
Agent 3 — Medical Advisor
=========================
Chief of Triage and Patient Communication. It reviews the technical assessment 
from the Clinical Diagnostician and decides the final treatment path and 
medication recommendations.

Responsibilities:
1. Review the clinical assessment and red flags.
2. Decide the final treatment path (OTC vs Doctor vs Emergency).
3. Recommend specific OTC medications and dosages.
4. Craft the final, friendly, and direct response to the patient.
"""

import json
from openai import OpenAI
from .utils import parse_json_response

SYSTEM_PROMPT = """You are the Senior Medical Advisor. Your role is to take a technical clinical assessment and turn it into a final triage decision and patient-facing plan.

YOUR INPUT:
You will receive:
1. Patient Profile (Symptoms, History)
2. Clinical Assessment (from the Diagnostician) including potential conditions and red flags.

RULES:
- Direct answers only. No filler.
- State the most probable diagnosis (1 sentence).
- Always tell the patient WHAT TO DO: recommend a MAXIMUM of 1 or 2 essential OTC medications with name + dosage, and any relevant self-care steps (rest, hydration, diet, etc.). Do NOT overwhelm the patient with too many medications.
- IMPORTANT: Always use common ISRAELI brand names for medications (e.g., recommend 'Acamol' or 'Dexamol' instead of Acetaminophen, 'Nurofen' or 'Advil' instead of Ibuprofen). This is critical for our local pharmacy search engine to work.
- NEVER suggest prescription drugs.
- If a doctor visit is needed, ask the patient whether to go ahead and book an appointment.
- Append exactly one line at the end: "I am an AI, not a licensed doctor."

TREATMENT PATH DECISION:
- "otc": For minor, self-limiting conditions. Prioritize this if symptoms are <3 days and no severe red flags were identified in the assessment.
- "doctor_needed": For persistent symptoms (>3 days), high fever, or if the Diagnostician identified significant clinical concerns/red flags.
- "emergency": Immediate life threat only.
- "more_info_needed": Use if the Diagnostician's assessment is "Inconclusive" or if a specific follow-up is needed to rule out a red flag.

You MUST respond with ONLY a valid JSON object:
{
  "diagnosis": "The primary condition identified by the Diagnostician (1 sentence)",
  "action": "otc | doctor_needed | emergency | more_info_needed",
  "reply": "Short, direct message. If more_info_needed, ask your follow-up question. Else, include: (1) diagnosis, (2) what to do including medication name and dosage. Last line: I am an AI, not a licensed doctor.",
  "otc_medications": ["List a MAXIMUM of 1 or 2 essential OTC medication names ONLY in HEBREW (e.g., 'אקמול', 'נורופן'). This is critical for our local pharmacy search engine."],
  "specialist_type": "e.g. General Practitioner — only if doctor_needed, else null",
  "urgency": "immediately | within_24h | this_week | whenever — only if doctor_needed, else null"
}
"""

def advise(client: OpenAI, profile: dict, clinical_assessment: dict, chat_history: list) -> dict:
    """
    Run the Medical Advisor.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    
    # Add history for context (last 3 messages)
    for msg in chat_history[-3:]:
        messages.append(msg)

    # Combine input data
    input_data = (
        f"PATIENT PROFILE:\n"
        f"Symptoms: {', '.join(profile.get('symptoms', []))}\n"
        f"Duration: {profile.get('duration', 'unknown')}\n\n"
        f"CLINICAL ASSESSMENT (from Diagnostician):\n"
        f"Assessment: {clinical_assessment.get('clinical_assessment')}\n"
        f"Red Flags: {', '.join(clinical_assessment.get('red_flags_identified', []))}\n"
        f"Potential Diagnoses: {json.dumps(clinical_assessment.get('potential_diagnoses'))}"
    )

    messages.append({"role": "user", "content": input_data})

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
    except:
        return {
            "diagnosis": "Inconclusive.",
            "action": "doctor_needed",
            "reply": "Based on my analysis, I recommend seeing a General Practitioner to be safe. I am an AI, not a licensed doctor.",
            "otc_medications": [],
            "specialist_type": "General Practitioner",
            "urgency": "this_week"
        }
