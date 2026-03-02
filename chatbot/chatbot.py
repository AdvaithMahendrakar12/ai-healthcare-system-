"""
Main Chatbot Logic
Combines FAISS retrieval with keyword boosting and safety rules.
"""

import re
from utils.retriever import retrieve

# Emergency keywords — always override with emergency message
EMERGENCY_KEYWORDS = [
    "heart attack", "can't breathe", "cannot breathe", "stroke",
    "unconscious", "not breathing", "severe bleeding", "overdose",
    "poisoning", "anaphylaxis", "chest crushing", "call ambulance",
    "emergency", "dying", "help me"
]

# Disclaimer always appended
DISCLAIMER = "\n\n⚠️ *This is general information only. Please consult a qualified healthcare professional for medical advice, diagnosis, or treatment.*"

EMERGENCY_RESPONSE = (
    "🚨 **EMERGENCY — Call emergency services IMMEDIATELY!**\n\n"
    "📞 **India:** 108 (Ambulance) | 112 (Emergency)\n"
    "📞 **USA/Canada:** 911\n"
    "📞 **UK:** 999\n"
    "📞 **Europe:** 112\n\n"
    "Do **NOT** wait — get help NOW. I am an AI and cannot provide emergency assistance."
)

FALLBACK_RESPONSE = (
    "I don't have enough specific information about that in my knowledge base. "
    "Here are some suggestions:\n\n"
    "• Describe your symptoms more specifically (e.g., 'sharp chest pain with shortness of breath')\n"
    "• Try asking about a specific condition (e.g., 'what is pneumonia?')\n"
    "• For urgent concerns, please consult a doctor or visit a clinic\n\n"
    "I cover: lungs, brain, fetal health, heart, diabetes, mental health, and general medicine."
)

CATEGORY_ICONS = {
    "lungs": "🫁",
    "brain": "🧠",
    "fetal": "👶",
    "heart": "❤️",
    "mental_health": "💙",
    "research": "🔬",
    "general": "🏥",
}


def is_emergency(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in EMERGENCY_KEYWORDS)


def format_response(results, query):
    """Format retrieved results into a clean answer."""
    if not results:
        return FALLBACK_RESPONSE + DISCLAIMER

    # Use best match
    best = results[0]
    answer = best["answer"].strip()
    category = best.get("category", "general")
    icon = CATEGORY_ICONS.get(category, "🏥")
    confidence = int(best["score"] * 100)

    # If multiple high-confidence results, append related info
    additional = []
    for r in results[1:3]:
        if r["score"] > 0.55 and r["answer"] != answer:
            additional.append(r["answer"])

    response = f"{icon} **{answer}**"

    if additional:
        response += "\n\n**Related information:**"
        for add in additional:
            response += f"\n• {add[:200]}..."

    response += DISCLAIMER
    return response, confidence, category


def chat(query: str):
    """
    Main entry point. Returns dict with response, confidence, category.
    """
    if not query or len(query.strip()) < 2:
        return {
            "response": "Please ask a health-related question.",
            "confidence": 0,
            "category": "general"
        }

    # Safety: Emergency check
    if is_emergency(query):
        return {
            "response": EMERGENCY_RESPONSE,
            "confidence": 100,
            "category": "emergency"
        }

    # Retrieve top matches from FAISS
    results = retrieve(query, top_k=5, threshold=0.28)

    if not results:
        return {
            "response": FALLBACK_RESPONSE + DISCLAIMER,
            "confidence": 0,
            "category": "general"
        }

    formatted = format_response(results, query)

    # format_response can return tuple or string
    if isinstance(formatted, tuple):
        response, confidence, category = formatted
    else:
        response, confidence, category = formatted, int(results[0]["score"] * 100), "general"

    return {
        "response": response,
        "confidence": confidence,
        "category": category,
        "matched_question": results[0]["question"]
    }
