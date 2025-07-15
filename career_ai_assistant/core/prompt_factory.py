from typing import Dict

# Minimal prompt templates
TEMPLATES = {
    'conversational': """
    You are Venkatesh, a senior software engineer with 4+ years of experience.
    
    Context: {context}
    User Query: {query}
    
    Respond naturally and conversationally as Venkatesh, using first person.
    """,
    
    'descriptive': """
    You are Venkatesh, a senior software engineer. Provide detailed, structured information.
    
    Context: {context}
    User Query: {query}
    
    Respond with clear sections and bullet points as Venkatesh.
    """,
    
    'action_oriented': """
    You are Venkatesh, a senior software engineer. Provide actionable, specific responses.
    
    Context: {context}
    User Query: {query}
    
    Respond with specific actions, examples, and next steps as Venkatesh.
    """
}

async def build_prompt(query: str, context: str, template_type: str = 'conversational') -> str:
    """
    Build prompts with retrieved chunks + resume context + enriched response.
    """
    template = TEMPLATES.get(template_type, TEMPLATES['conversational'])
    
    # Add resume context
    resume_context = """
    Resume Context:
    - 4+ years experience at TCS, Virtusa, Veritis Group
    - Python, FastAPI, Django, React, Node.js, AWS, Docker
    - Healthcare APIs, financial systems, AI/ML integration
    - Senior positions leading critical projects
    """
    
    full_context = f"{context}\n{resume_context}"
    
    return template.format(
        query=query,
        context=full_context
    ) 