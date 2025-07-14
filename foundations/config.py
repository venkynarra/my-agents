"""
Centralized configuration for the AI Career Assistant application.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# --- Configuration ───────────────────────────────────────────────────────────

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent

# Path to the SQLite database
DATABASE_URL = f"sqlite:///{PROJECT_ROOT / 'foundations' / 'career_analytics.db'}"

# SendGrid API Key for email notifications
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

# Email addresses for contact form submissions
SENDER_EMAIL = os.getenv("FROM_EMAIL")  # Use FROM_EMAIL from .env
RECIPIENT_EMAIL = os.getenv("TO_EMAIL")  # Use TO_EMAIL from .env
FROM_EMAIL = os.getenv("FROM_EMAIL")

# Google AI Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Paths ---
KNOWLEDGE_DIR = PROJECT_ROOT / "agent_knowledge"
RESUME_PATH = KNOWLEDGE_DIR / "venkatesh_narra_resume.pdf"
DB_PATH = PROJECT_ROOT / "foundations" / "career_analytics.db"

# --- General ---
KNOWLEDGE_BASE_DIR = ["agent_knowledge/"]
RAG_INDEX_PATH = "foundations/rag_index/"

# Using a Base64 encoded image to avoid network issues.
DASHBOARD_IMAGE_B64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AL+AAf/Z"

# --- Scheduling ---

# Your Calendly link for the "Schedule a Meeting" button.
# (https://calendly.com/)
CALENDLY_LINK = os.getenv("CALENDLY_LINK", "https://calendly.com/venkateshnarra368")

# --- Validation ---
def validate_config():
    """
    Validates the configuration to ensure critical paths and credentials are set.
    """
    if SENDGRID_API_KEY == "YOUR_SENDGRID_API_KEY_HERE":
        print("⚠️ WARNING: SendGrid API key is not set. Email functionality will be disabled.")
    if not os.path.exists(RESUME_PATH):
        print(f"⚠️ WARNING: Resume file not found at {RESUME_PATH}. Download may fail.")

if __name__ == "__main__":
    validate_config()
    print("✅ Configuration seems OK.") 