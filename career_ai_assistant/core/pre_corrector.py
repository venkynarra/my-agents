import asyncio
import os
from typing import Dict, List
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini with environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

async def pre_correct_query(query: str) -> Dict:
    """
    Use Gemini to pre-correct and enrich the query.
    """
    if not model:
        return {
            'corrected_query': query,
            'intent': 'unknown',
            'keywords': []
        }
    
    prompt = f"""
    Analyze and improve this query for a career assistant:
    Query: "{query}"
    
    Please:
    1. Fix any grammar issues
    2. Add relevant keywords (resume, GitHub, skills, experience, projects, etc.)
    3. Classify the intent (greeting, resume_request, skill_inquiry, contact, etc.)
    4. Return a JSON with: corrected_query, intent, keywords
    """
    
    try:
        response = await asyncio.to_thread(
            model.generate_content, prompt
        )
        # Parse response (simplified for now)
        corrected_query = query  # Placeholder
        intent = "general"
        keywords = ["resume", "skills"]
        
        return {
            'corrected_query': corrected_query,
            'intent': intent,
            'keywords': keywords
        }
    except Exception as e:
        # Fallback
        return {
            'corrected_query': query,
            'intent': 'unknown',
            'keywords': []
        } 