from dotenv import load_dotenv
import os
import json
import requests
from pypdf import PdfReader
from docx import Document
import gradio as gr
import google.generativeai as genai

# 🔐 Load .env variables
load_dotenv(override=True)

# ✅ Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

# 📬 Pushover to track user interest and unknown questions
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

        # Read LinkedIn
        linkedin_path = "agent_knowledge/Profile.pdf"
        self.linkedin = self.extract_pdf_text(linkedin_path)

        # Read Resume
        resume_path = "agent_knowledge/resume.pdf"
        self.resume = self.extract_pdf_text(resume_path)

        # Read GitHub README (DOCX)
        readme_path = "agent_knowledge/github_readme.docx"
        self.readmes = self.extract_docx_text(readme_path)

    def extract_pdf_text(self, path):
        reader = PdfReader(path)
        return "".join(page.extract_text() or "" for page in reader.pages)

    def extract_docx_text(self, path):
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    def system_prompt(self):
        return f"""
You are acting as {self.name}, answering questions professionally and clearly.
You have access to the following data sources:

## Resume:
{self.resume}

## LinkedIn:
{self.linkedin}

## GitHub Projects:
{self.readmes}

If a user shows interest, ask for their email and log it using record_user_details(email, name, notes).
If a question can't be answered, record it using record_unknown_question(question).
"""

    def chat(self, message, history):
        chat = model.start_chat(history=[
            {"role": "user", "parts": [self.system_prompt()]}
        ])
        response = chat.send_message(message)
        return response.text

# 🟢 Gradio Launch
if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
