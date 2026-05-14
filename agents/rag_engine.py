import json
import os
import requests
import logging

logger = logging.getLogger(__name__)

KNOWLEDGE_FILE = os.path.join(os.path.dirname(__file__), "..", "knowledge", "medical_reference.json")

def online_retrieve_fallback(query: str) -> str:
    """
    Fetches real-time medical condition data from the National Library of Medicine (NLM).
    Used as a fallback when local knowledge is insufficient.
    """
    try:
        # NLM ClinicalTables API - Condition Search
        url = f"https://clinicaltables.nlm.nih.gov/api/conditions/v3/search?terms={query}&max=3"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Format: [total, ids, None, [[name], ...]]
            if data and data[0] > 0:
                conditions = [c[0] for c in data[3]]
                return (
                    f"### Online Clinical Data (Source: National Library of Medicine)\n"
                    f"Real-time lookup for '{query}' suggests these possible clinical conditions: {', '.join(conditions)}.\n"
                    f"Note: These results are retrieved live from clinical tables and may require professional verification."
                )
    except Exception as e:
        logger.error(f"Online RAG fallback failed: {e}")
    
    return ""

def retrieve_medical_context(user_symptoms: list[str]) -> str:
    """
    Hybrid RAG: Searches local knowledge first, falls back to NLM Online API if needed.
    """
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
    except Exception as e:
        logger.error(f"Error loading knowledge base: {e}")
        return ""

    matched_entries = []
    user_words = set()
    for s in user_symptoms:
        user_words.update(s.lower().split())
    
    for entry in knowledge_base:
        condition_words = set(entry["condition"].lower().split())
        symptom_words = set()
        for s in entry["symptoms"]:
            symptom_words.update(s.lower().split())
        
        symptom_match = len(user_words.intersection(symptom_words))
        condition_match = len(user_words.intersection(condition_words))
        score = symptom_match + (condition_match * 2)
        
        if score > 0:
            matched_entries.append((score, entry))
            
    matched_entries.sort(key=lambda x: x[0], reverse=True)
    
    context_parts = []
    
    # If we have local matches, use them
    if matched_entries:
        context_parts.append("### Verified Medical Reference Data (Source: MedlinePlus/Mayo Clinic/CDC)")
        for score, entry in matched_entries[:2]:
            context_parts.append(
                f"**Condition: {entry['condition']}**\n"
                f"- **Typical Symptoms**: {', '.join(entry['symptoms'])}\n"
                f"- **Recommended OTC**: {', '.join(entry['otc_recommendations'])}\n"
                f"- **Self-care**: {entry['self_care']}\n"
                f"- **Emergency Red Flags**: {', '.join(entry['red_flags'])}"
            )
    
    # If score is low or no local matches, call the Online RAG
    top_score = matched_entries[0][0] if matched_entries else 0
    if top_score < 2:
        query = " ".join(user_symptoms)
        online_data = online_retrieve_fallback(query)
        if online_data:
            context_parts.append(online_data)

    if not context_parts:
        return "No specific medical reference found. Proceed with standard clinical safety assessment."

    return "\n\n---\n\n".join(context_parts)
