import re
from rapidfuzz import fuzz, process
from typing import Dict

# Define ONLY truly simple patterns that should go direct
SIMPLE_PATTERNS = [
    r"^\s*(hello|hi|hey|greetings)\s*$",  # Only exact greetings
    r"^\s*(how are you|what's up)\s*$",   # Only exact casual questions
]

# Define complex patterns that should use MCP agent
COMPLEX_PATTERNS = [
    r"\b(experience|work|job|career|background)\b",
    r"\b(skill|technology|tech|programming|language|framework)\b",
    r"\b(project|build|create|develop|application|system)\b",
    r"\b(certification|cert|certified|education|degree|university)\b",
    r"\b(about|who are you|intro|tell me)\b",
    r"\b(linkedin|github|contact|email|phone)\b",
    r"\b(hire|recruit|interview|position|role)\b",
    r"\b(salary|compensation|benefits|remote|onsite)\b",
    r"\b(api|database|cloud|aws|azure|docker|kubernetes)\b",
    r"\b(ml|ai|machine learning|artificial intelligence|data)\b",
    r"\b(fastapi|django|flask|react|angular|typescript)\b",
    r"\b(python|java|javascript|sql|nosql|redis)\b"
]

INTENT_KEYWORDS = {
    'greeting': ['hello', 'hi', 'hey', 'greetings'],
    'casual': ['how are you', "what's up"],
    'complex': ['experience', 'skill', 'project', 'certification', 'about', 'who are you', 'hire', 'linkedin']
}

async def preprocess_query(query: str) -> Dict:
    """
    Detects if the query is truly simple or complex.
    Returns a dict with routing flag, detected intent, and cleaned query.
    """
    cleaned_query = query.strip().lower()
    
    # Check for complex patterns first (priority)
    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, cleaned_query):
            return {'route_direct': False, 'intent': 'complex', 'cleaned_query': cleaned_query}
    
    # Check for truly simple patterns (exact matches only)
    for pattern in SIMPLE_PATTERNS:
        if re.search(pattern, cleaned_query):
            return {'route_direct': True, 'intent': 'simple', 'cleaned_query': cleaned_query}
    
    # Check for complex keywords with fuzzy matching
    for intent, keywords in INTENT_KEYWORDS.items():
        if intent == 'complex':
            for kw in keywords:
                if fuzz.partial_ratio(kw, cleaned_query) > 80 or re.search(rf"\b{kw}\b", cleaned_query):
                    return {'route_direct': False, 'intent': intent, 'cleaned_query': cleaned_query}
    
    # Default to complex for better responses
    return {'route_direct': False, 'intent': 'complex', 'cleaned_query': cleaned_query} 