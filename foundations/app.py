from dotenv import load_dotenv
import os, requests, gradio as gr, asyncio, spacy, random, json
import google.generativeai as genai
from pypdf import PdfReader
from docx import Document
from rag_engine import build_vector_index, get_rag_response_async

# ─── Environment & AI setup ────────────────────────────────────────────────────
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER  = os.getenv("PUSHOVER_USER")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def push(text: str):
    """Send a pushover notification on fallback or email submit."""
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={"token": PUSHOVER_TOKEN, "user": PUSHOVER_USER, "message": text}
    )

def record_user_details(email, name="Not Provided", notes="Not Provided"):
    push(f"Email from {name}: {email} | Notes: {notes}")
    return {"status": "ok"}

def record_unknown_question(question):
    push(f"Unknown or fallback: {question}")
    return {"status": "ok"}

# ─── Main agent ────────────────────────────────────────────────────────────────
class Me:
    def __init__(self):
        self.linkedin = self.read_text_file("../agent_knowledge/profile.md")
        self.resume   = self.read_text_file("../agent_knowledge/resume.md")
        self.readmes  = self.read_text_file("../agent_knowledge/github_readme.md")
        self.projects = self.read_text_file("../agent_knowledge/experience_and_projects.md")

        self.full_text = (
            f"--- START: Resume ---\n{self.resume}\n--- END: Resume ---\n\n"
            f"--- START: LinkedIn Profile ---\n{self.linkedin}\n--- END: LinkedIn Profile ---\n\n"
            f"--- START: GitHub READMEs ---\n{self.readmes}\n--- END: GitHub READMEs ---\n\n"
            f"--- START: Detailed Projects & Experience ---\n{self.projects}\n--- END: Detailed Projects & Experience ---"
        )
        build_vector_index(self.full_text)

    def read_text_file(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file {path}: {e}")
            return ""

    def extract_pdf_text(self, path):
        # This function is no longer needed here but kept for potential future use
        try:
            return "".join(p.extract_text() or "" for p in PdfReader(path).pages)
        except Exception as e:
            print(f"Error reading pdf {path}: {e}")
            return ""

    def extract_docx_text(self, path):
        # This function is no longer needed here but kept for potential future use
        try:
            return "\n".join(p.text for p in Document(path).paragraphs if p.text.strip())
        except Exception as e:
            print(f"Error reading docx {path}: {e}")
            return ""

    async def route_query(self, query):
        """Uses an LLM to route the user's query to the correct tool or response."""
        prompt = f"""
You are an intelligent query routing agent. Your task is to analyze a user's message and identify all distinct user intents. You must respond with a JSON object containing a list of these intents.

The possible intents are:
- "greeting": For any kind of greeting (e.g., "hello", "how are you").
- "about_me": For general questions about Venkatesh Narra (e.g., "tell me about yourself", "who are you").
- "get_linkedin": For requests specifically for the LinkedIn URL.
- "get_github": For requests specifically for the GitHub URL.
- "get_leetcode": For requests specifically for the LeetCode URL.
- "rag_query": For any other question about skills, experience, projects, work, or any topic that requires looking up information. This is the default for substantive questions.

Rules:
- A single message can have multiple intents. Include all of them.
- For a "rag_query" intent, you MUST also include a "rag_query_text" field in the JSON, containing the part of the user message that needs to be answered by the retrieval system.
- If a message contains multiple, distinct topics for the RAG system, combine them into a single, coherent "rag_query_text".

Example 1:
User message: "hey, how's it going?"
Response: {{"intents": ["greeting"]}}

Example 2:
User message: "tell me about your work and give me your github profile"
Response: {{"intents": ["get_github", "rag_query"], "rag_query_text": "tell me about your work"}}

Example 3:
User message: "what is your special talent and tell me about your recent work and give me your github projects"
Response: {{"intents": ["get_github", "rag_query"], "rag_query_text": "what is your special talent and what is your recent work, including projects from github?"}}

Example 4:
User message: "give me your linkedin and github"
Response: {{"intents": ["get_linkedin", "get_github"]}}

User message: "{query}"
Response:
"""
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = await model.generate_content_async(prompt)
            clean_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(clean_response)
        except (Exception, json.JSONDecodeError) as e:
            print(f"Error routing query: {e}")
            return {"intents": ["rag_query"], "rag_query_text": query} # Fallback

    def get_greeting(self):
        return random.choice([
            "Hi there! How can I help you today?",
            "Hello! Feel free to ask me about my work, skills, or projects.",
            "Hey! What can I help you with?",
        ])

    def get_about_me(self):
        return ("I am Venkatesh Narra, a Full-Stack Python Developer with a Master's in MIS. "
                "I specialize in building scalable AI-powered applications and backend systems using technologies like FastAPI, Django, and cloud services like AWS and Azure. "
                "My work involves developing everything from data-intensive applications to multi-model chatbots. Feel free to ask about specific projects or skills!")

    async def get_rag_answer(self, query):
        try:
            response = await get_rag_response_async(query, self.full_text)
            if "I don't have enough information" in response:
                record_unknown_question(query)
            return response
        except Exception as e:
            print(f"Error during RAG lookup: {e}")
            record_unknown_question(f"SYSTEM ERROR on query: {query}")
            return "Sorry, I'm having a little trouble right now. Please try again in a moment."

    async def chat(self, message, history):
        if "@" in message and "." in message:
            record_user_details(email=message, notes="Email submitted via chat")
            return "Got your email. I'll follow up soon."
        
        route = await self.route_query(message)
        intents = route.get("intents", [])
        
        tasks = []
        for intent in intents:
            if intent == "greeting":
                tasks.append(asyncio.create_task(asyncio.to_thread(self.get_greeting)))
            elif intent == "about_me":
                tasks.append(asyncio.create_task(asyncio.to_thread(self.get_about_me)))
            elif intent == "get_linkedin":
                tasks.append(asyncio.create_task(asyncio.to_thread(lambda: "My LinkedIn profile is https://www.linkedin.com/in/venkateswara-narra-91170b34a/")))
            elif intent == "get_github":
                tasks.append(asyncio.create_task(asyncio.to_thread(lambda: "You can find my projects on GitHub: https://github.com/venkynarra")))
            elif intent == "get_leetcode":
                tasks.append(asyncio.create_task(asyncio.to_thread(lambda: "My LeetCode profile is https://leetcode.com/u/pravnarri/")))
            elif intent == "rag_query":
                rag_query_text = route.get("rag_query_text", message)
                tasks.append(self.get_rag_answer(rag_query_text))
        
        if not tasks: # Fallback if routing fails
             tasks.append(self.get_rag_answer(message))

        responses = await asyncio.gather(*tasks)
        return "\n\n".join(filter(None, responses))

# ─── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = Me()
    gr.ChatInterface(
        agent.chat,
        title="Venkatesh Narra - AI Assistant",
        description="""
        Welcome! I am Venkatesh Narra's AI assistant.
        You can ask me about my skills, projects, and professional experience.
        Feel free to ask complex questions or multiple questions at once.
        """,
        examples=[
            "Tell me about your work and provide me your linkedin url",
            "what is your special talent and tell me about your recent work and give me your github projects",
            "give me your linkedin and github profile"
        ],
        type="messages"
    ).launch()
