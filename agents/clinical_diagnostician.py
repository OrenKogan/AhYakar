"""
Agent 2 — Clinical Diagnostician
===============================
Focuses purely on clinical analysis. It takes the triage data and any visual 
evidence to form a technical medical assessment. It does NOT talk to the 
patient directly; its output is for the Medical Advisor.

Responsibilities:
1. Analyze symptoms, duration, and medical history.
2. Evaluate visual evidence from images (if provided).
3. Identify potential diagnoses and their likelihood.
4. Flag any "Red Flags" or high-risk indicators.
"""

import json
from openai import OpenAI
from .utils import parse_json_response

SYSTEM_PROMPT = """You are a Clinical Diagnostician. Your role is to provide a technical medical assessment based on patient data and images.

YOUR GOAL:
Analyze the provided information to identify the most likely conditions and any clinical risks. Be technical and precise.

VISION INSTRUCTIONS:
If images are provided, perform a detailed dermatological or clinical examination. Note inflammation, distribution, color, and texture.

You MUST respond with ONLY a valid JSON object:
{
  "clinical_assessment": "Technical summary of the patient's state.",
  "potential_diagnoses": [
    {"condition": "Name", "likelihood": "High/Medium/Low", "reasoning": "Why?"}
  ],
  "red_flags_identified": ["list of concerning signs found, or empty"],
  "visual_findings": "Summary of findings from images, or null if no image."
}
"""

def analyze(client: OpenAI, triage: dict, chat_history: list, image_data: str | list[str] = None) -> dict:
    """
    Run the Clinical Diagnostician.
    """
    symptoms = triage.get('symptoms', [])
    
    user_text = (
        f"Patient Data:\n"
        f"Symptoms: {', '.join(symptoms)}\n"
        f"Duration: {triage.get('duration', 'unknown')}\n"
        f"Relevant history: {triage.get('relevant_history', 'none')}"
    )
    
    user_content = [{"type": "text", "text": user_text}]
    if image_data:
        images = [image_data] if isinstance(image_data, str) else image_data
        for img in images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=messages,
        extra_headers={
            "HTTP-Referer": "http://localhost:5000",
            "X-OpenRouter-Title": "AhYakar - Clinical Diagnostician",
        },
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    try:
        return parse_json_response(raw)
    except:
        return {
            "clinical_assessment": "Inconclusive due to parsing error.",
            "potential_diagnoses": [],
            "red_flags_identified": [],
            "visual_findings": None
        }
