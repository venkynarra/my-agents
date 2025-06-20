from dotenv import load_dotenv
import os
import requests
from pypdf import PdfReader
from docx import Document
import gradio as gr
import google.generativeai as genai

# 🔐 Load .env variables
load_dotenv(override=True)

# ✅ Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro")

# 📬 Pushover Logging
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

# 👤 Main Agent Class
class Me:
    def __init__(self):
        self.name = "Venkatesh Narra"
        self.linkedin = self.extract_pdf_text("../agent_knowledge/Profile.pdf")
        self.resume = self.extract_pdf_text("../agent_knowledge/resume.pdf")
        self.readmes = self.extract_docx_text("../agent_knowledge/github_readme.docx")
        self.prompt = self.system_prompt()

    def extract_pdf_text(self, path):
        reader = PdfReader(path)
        return "".join(page.extract_text() or "" for page in reader.pages)

    def extract_docx_text(self, path):
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    def system_prompt(self):
        return f"""
You are acting as {self.name}, a professional AI assistant. Answer clearly and helpfully based on:
- Resume
- LinkedIn Profile
- GitHub README and Projects

You are expected to:
- Handle technical, behavioral, and project-related questions
- Politely reject irrelevant or off-topic questions
- Offer links to GitHub, LinkedIn, and LeetCode when requested
- Log emails using record_user_details(email, name, notes)
- Log unknown questions using record_unknown_question(question)

Resume:
{self.resume}

LinkedIn:
{self.linkedin}

GitHub Projects:
{self.readmes}
"""

    def chat(self, message, history):
        user_input = message.lower().strip()

        # 👋 Greetings
        if user_input in ["hi", "hello", "hey", "yo", "hii", "hai"]:
            return "Hi there! Feel free to ask me about my work, skills, or projects."

        # 🎯 Explicit intro
        if "intro" in user_input or "tell me about yourself" in user_input:
            return (
                "**Introduction:**\n"
                "I'm Venkatesh Narra, a full-stack Python developer with a Master’s in Management Information Systems.\n"
                "I specialize in building scalable backend APIs using FastAPI, Django, and Flask, and frontend interfaces using React and TypeScript.\n"
                "I’ve worked extensively with cloud services (AWS, Azure), DevOps pipelines, and AI model integrations (Gemini, Cohere, Mistral).\n\n"
                "**Current Focus:**\n"
                "- Building a multi-model AI assistant (Gemini, Cohere, Mistral)\n"
                "- Implementing Salesforce & ServiceNow RAG workflows with FastAPI + SQLite\n"
                "- Designing an automated testing agent with PDF logging and LLM response verification"
            )

        # ⚙️ Quick Info Triggers
        if "github" in user_input:
            return "🔗 GitHub: https://github.com/venkynarra"

        if "linkedin" in user_input:
            return "🔗 LinkedIn: https://www.linkedin.com/in/venkateswara-narra-91170b34a/"

        if "leetcode" in user_input:
            return "🔗 LeetCode: https://leetcode.com/u/pravnarri/"
        if "all profile" in user_input or "profile urls" in user_input or "your profiles" in user_input:
            return (
            "Certainly! Here are my professional profiles:\n\n"
            "🔗 GitHub: https://github.com/venkynarra\n"
            "🔗 LinkedIn: https://www.linkedin.com/in/venkateswara-narra-91170b34a/\n"
            "🔗 LeetCode: https://leetcode.com/u/pravnarri/"
                 )


        if "skills" in user_input or "tech stack" in user_input:
            return (
                "**Tech Stack:**\n"
                "- **Languages:** Python, JavaScript, TypeScript, Java\n"
                "- **Frameworks:** FastAPI, Flask, Django, Spring Boot, React, Node.js\n"
                "- **Cloud & DevOps:** AWS, Azure, Docker, GitHub Actions\n"
                "- **Databases:** PostgreSQL, MongoDB, SQLite\n"
                "- **AI & Tools:** Gemini, Cohere, Mistral, LoRA, QLoRA"
            )

        if "current project" in user_input or "what are you working on" in user_input:
            return (
                "**Current Projects:**\n"
                "1. **AI Testing Agent** – Validates REST endpoints and UI flows with Gemini-based output validation and auto-generated PDF reports.\n"
                "2. **Multi-Model Chat App** – CLI and Web-based assistant comparing Gemini, Mistral, and Cohere responses in real-time with fallback and retry logic.\n"
                "3. **Salesforce RAG Platform** – FastAPI backend with SQLite vector store that indexes and retrieves Salesforce/ServiceNow records for chat-based exploration."
            )

        if any(phrase in user_input for phrase in ["your name", "who are you"]):
            return "I'm Venkatesh Narra – a full-stack developer and AI platform engineer."

        # 🚫 Irrelevant topics
        if any(phrase in user_input for phrase in [
            "joke", "movie", "celebrity", "weather", "song", "love", "married", "politics", "age", "food", "drink"
        ]):
            return "Let’s stay focused on professional topics like my experience, projects, or skills."

        # 🧠 Gemini fallback for open-ended or behavioral/technical questions
        try:
            chat = model.start_chat(history=[{"role": "user", "parts": [self.prompt]}])
            response = chat.send_message(message)
            return response.text
        except Exception as e:
            record_unknown_question(message)
            return "Something went wrong processing that. Please ask about my work or experience."


# 🟢 Gradio Launch
if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
