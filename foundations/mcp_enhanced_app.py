"""
Enhanced Career Assistant with gRPC Backend Integration
"""
import gradio as gr
import asyncio
import logging
import grpc
import os
import plotly.express as px
import pandas as pd
from datetime import datetime
import re
from google.protobuf import empty_pb2
import time
from pathlib import Path

# Import generated gRPC classes
from . import career_assistant_pb2
from . import career_assistant_pb2_grpc

# Import configuration
from .config import (
    RESUME_PATH,
    CALENDLY_LINK,
    DASHBOARD_IMAGE_B64
)

# --- Profile Content ---
try:
    with open("agent_knowledge/profile.md", "r", encoding="utf-8") as f:
        PROFILE_MD = f.read()
except FileNotFoundError:
    PROFILE_MD = "## Profile Information Not Found\nPlease ensure `agent_knowledge/profile.md` exists."

def parse_profile_md(md_content):
    """Parses the profile markdown into structured sections using a more robust method."""
    sections = {}
    current_section = None
    
    # Split by major headers (H2)
    major_sections = re.split(r'\n## ', md_content)
    
    for section in major_sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        header = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        
        sections[header.lower()] = content

    # Further parse skills into sub-categories
    skills_content = sections.get('top skills', '')
    skill_categories = {}
    current_category = None
    
    for line in skills_content.split('\n'):
        if line.startswith('### '):
            current_category = line.replace('###', '').strip()
            skill_categories[current_category] = []
        elif line.startswith('- ') and current_category:
            skill_categories[current_category].append(line)
            
    sections['skills_categorized'] = skill_categories

    return {
        "summary": sections.get("summary", "Not found."),
        "skills": sections.get("top skills", "Not found."),
        "experience": sections.get("experience", "Not found."),
        "education": sections.get("education", "Not found."),
        "projects": sections.get("projects", "Not found."),
        "skills_categorized": sections.get('skills_categorized', {})
    }


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-career-assistant")

# --- Constants ---
# Using a Base64 encoded image to avoid network issues.
AGI_IMAGE_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAYAAAB/h956AAAAAXNSR0IArs4c6QAAAEZJREFUeF7t0AEBAAAAgvD/X3+AGRgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA5bg4sAABsARVTAAAAABJRU5ErkJggg=="

# --- gRPC Client ---
class CareerAssistantClient:
    """gRPC client for the Career Assistant service."""
    def __init__(self, host='localhost', port=50051):
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None
        self.connect()
        logger.info(f"🔌 gRPC client connected to server at {host}:{port}")

    def connect(self):
        """Establishes a gRPC channel and stub."""
        try:
            self.channel = grpc.insecure_channel(f"{self.host}:{self.port}")
            self.stub = career_assistant_pb2_grpc.CareerAssistantStub(self.channel)
        except grpc.RpcError as e:
            logger.error(f"gRPC error connecting to server {self.host}:{self.port}: {e.details()}")
            self.channel = None
            self.stub = None

    def get_analytics(self):
        """Fetches analytics data from the gRPC server."""
        if not self.stub:
            logger.warning("gRPC stub not initialized. Cannot fetch analytics.")
            return []
        try:
            request = empty_pb2.Empty()
            response = self.stub.GetAnalyticsData(request)
            return response.interactions
        except grpc.RpcError as e:
            logger.error(f"gRPC error fetching analytics: {e.details()}")
            return []

    def generate_profile(self):
        """Generates the profile summary from the gRPC server."""
        if not self.stub:
            logger.warning("gRPC stub not initialized. Cannot generate profile.")
            return "Error: Could not generate the profile at this time."
        try:
            request = empty_pb2.Empty()
            response = self.stub.GenerateProfile(request)
            return response.content
        except grpc.RpcError as e:
            logger.error(f"gRPC error generating profile: {e.details()}")
            return "Error: Could not generate the profile at this time."

    def process_query(self, query: str, history: list):
        """Sends a query to the gRPC server and gets a response."""
        if not self.stub:
            logger.warning("gRPC stub not initialized. Cannot process query.")
            return f"Error: Could not connect to the AI service. gRPC stub not initialized."
        try:
            chat_history = [career_assistant_pb2.ChatMessage(role=msg["role"], content=msg["content"]) for msg in history]
            request = career_assistant_pb2.QueryRequest(query=query, history=chat_history)
            response = self.stub.ProcessQuery(request)
            return response.response
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.details()} (code: {e.code().name})")
            return f"Error: Could not connect to the AI service. {e.details()}"

    def submit_contact_form(self, name: str, email: str, message: str):
        """Submits the contact form to the gRPC server."""
        if not self.stub:
            logger.warning("gRPC stub not initialized. Cannot submit contact form.")
            return {"success": False, "message": "Error: Could not submit form. gRPC stub not initialized."}
        try:
            request = career_assistant_pb2.ContactFormRequest(name=name, email=email, message=message)
            response = self.stub.SubmitContactForm(request)
            return {"success": response.success, "message": response.message}
        except grpc.RpcError as e:
            logger.error(f"gRPC error on contact form: {e.details()} (code: {e.code().name})")
            return {"success": False, "message": f"Error: Could not submit form. {e.details()}"}

# --- Gradio Interface ---
def create_gradio_interface(client: CareerAssistantClient):
    """Creates and launches the Gradio interface."""
    
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="*neutral_950",
        body_background_fill_dark="*neutral_950",
        body_text_color="*neutral_200",
        body_text_color_dark="*neutral_200",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700",
        button_secondary_background_fill="*neutral_700",
        button_secondary_background_fill_hover="*neutral_800",
        border_color_accent="*primary_500",
        border_color_accent_dark="*primary_500",
        color_accent_soft="*primary_500",
        color_accent_soft_dark="*primary_500",
        link_text_color="*primary_500",
        link_text_color_dark="*primary_500",
        link_text_color_hover="*primary_400",
        link_text_color_hover_dark="*primary_400",
        link_text_color_visited="*primary_600",
        link_text_color_visited_dark="*primary_600",
        block_title_text_color="*neutral_200",
        block_title_text_color_dark="*neutral_200",
        block_label_text_color="*neutral_400",
        block_label_text_color_dark="*neutral_400",
        input_background_fill="*neutral_800",
        input_background_fill_dark="*neutral_800",
        shadow_drop="rgba(0,0,0,0.2)"
    )

    custom_css = """
        body { background-color: #111827; }
        #chatbot_window { height: 500px; background-color: #111827 !important; border: 1px solid #374151;}
        .gr-sidebar { background-color: #1f2937; border-right: 1px solid #374151; padding: 20px; }
        #title_header { text-align: center; margin-bottom: 20px; }
        #download_button { width: 100%; }
        textarea, input[type='text'], input[type='email'], input[type='password'] { 
            background-color: #374151 !important; 
            color: white !important; 
            border: 1px solid #4A5568 !important;
        }
        .gr-message-text { color: black !important; }
    """

    with gr.Blocks(
        theme=theme,
        title="AI Career Assistant for Venkatesh Narra",
        css=custom_css
    ) as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=250, elem_classes="gr-sidebar"):
                gr.Markdown("# Venkatesh Narra", elem_id="title_header")
                gr.DownloadButton(
                    "📄 Download Resume",
                    value="agent_knowledge/venkatesh_narra_resume.pdf",
                    variant="primary",
                    elem_id="download_button"
                )

            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.TabItem("🧠 AI Chatbot", id="chatbot_tab"):
                        gr.Markdown("<h2 style='text-align: center;'>Ask Me Anything</h2>")
                        chatbot_window = gr.Chatbot(elem_id="chatbot_window", label="Conversation", type="messages")
                        with gr.Row():
                            query_input = gr.Textbox(placeholder="e.g., What is your experience with AI?", show_label=False, scale=4)
                            send_button = gr.Button("▶️ Send", variant="primary", scale=1)
                        gr.Examples(
                            ["What are your top 3 technical skills?", "Summarize your experience with cloud technologies.", "What kind of role are you looking for?"],
                            inputs=query_input,
                            label="Example Questions"
                        )

                    with gr.TabItem("📊 Dashboard", id="dashboard_tab") as dashboard_tab_item:
                        gr.Markdown("<h1 style='text-align: center;'>AI Assistant Dashboard</h1>")
                        gr.Markdown("Welcome to my AI-Powered Career Hub! This dashboard provides real-time insights into user interactions with the assistant.")
                        with gr.Row():
                            refresh_button = gr.Button("🔄 Refresh Analytics")
                        with gr.Row():
                            daily_interactions_plot = gr.Plot(label="Daily Interactions")
                            hourly_activity_plot = gr.Plot(label="Hourly Activity")
                        with gr.Row():
                            frequent_queries_plot = gr.Plot(label="Most Frequent Queries")
                            
                    with gr.TabItem("👤 Profile", id="profile_tab") as profile_tab:
                        gr.Markdown("<h1 style='text-align: center;'>Professional Profile</h1>")
                        with gr.Accordion("Executive Summary", open=True):
                            summary_md = gr.Markdown("Loading...")
                        with gr.Accordion("Top Skills", open=True):
                            skills_md = gr.Markdown("Loading...")
                            with gr.Row():
                                lang_skills = gr.Markdown(label="Languages & Frameworks")
                                backend_skills = gr.Markdown(label="Backend & Databases")
                                frontend_skills = gr.Markdown(label="Frontend & Design")
                            with gr.Row():
                                data_skills = gr.Markdown(label="Data Science & ML")
                                devops_skills = gr.Markdown(label="DevOps & Tooling")
                                cloud_skills = gr.Markdown(label="Cloud & Infrastructure")
                        with gr.Accordion("Professional Experience", open=False):
                            experience_md = gr.Markdown("Loading...")
                        with gr.Accordion("Education", open=False):
                            education_md = gr.Markdown("Loading...")
                        with gr.Accordion("Projects", open=False):
                            projects_md = gr.Markdown("Loading...")

                    with gr.TabItem("✉️ Contact & Schedule", id="contact_tab"):
                        gr.Markdown("<h1 style='text-align: center;'>Contact Me</h1>")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Get in Touch")
                                gr.Markdown("Have a question or a project proposal? Fill out the form below.")
                                with gr.Column():
                                    name = gr.Textbox(label="Your Name")
                                    email = gr.Textbox(label="Your Email")
                                    message = gr.Textbox(label="Message", lines=4)
                                    contact_btn = gr.Button("Send", variant="primary")
                                contact_status = gr.Markdown(visible=True, value="")
                            with gr.Column():
                                gr.Markdown("### Schedule a Meeting")
                                gr.Markdown("Or schedule a 30-minute meeting with me directly via Calendly:")
                                gr.Markdown(f"<a href='{CALENDLY_LINK}' target='_blank' style='color: #3B82F6; text-decoration: none;'>📅 Schedule on Calendly</a>")


        # --- Function Definitions ---
        async def chat_wrapper(query, history):
            """Wraps the gRPC client call to process a query."""
            if not history:
                history = []
            
            # Add user message to history, ensuring it's in the correct format
            history.append({"role": "user", "content": query})
            
            # Prepare history for API call
            history_for_api = [msg for msg in history if isinstance(msg, dict) and "role" in msg and "content" in msg]
            
            response = client.process_query(query, history_for_api)
            history.append({"role": "assistant", "content": response})
            return "", history

        def handle_contact_submission(name_val, email_val, msg_val):
            """Handles the submission of the contact form."""
            try:
                # Validate inputs
                if not name_val.strip():
                    return "❌ Please enter your name."
                if not email_val.strip():
                    return "❌ Please enter your email address."
                if not msg_val.strip():
                    return "❌ Please enter a message."
                
                # Basic email validation
                import re
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, email_val.strip()):
                    return "❌ Please enter a valid email address."
                
                # Submit the form
                result = client.submit_contact_form(name_val, email_val, msg_val)
                
                # Check if submission was successful
                if result["success"]:
                    return f"✅ **Success!** Thank you {name_val}! Your message has been sent successfully. You should receive a confirmation email at {email_val} shortly."
                else:
                    return f"❌ **Error:** {result['message']}"
                    
            except Exception as e:
                logger.error(f"Error in contact form submission: {e}")
                return "❌ **Error:** There was an issue sending your message. Please try again or contact directly via email."

        def update_profile():
            """Loads actual content from knowledge files and formats it for the profile UI."""
            try:
                # Load actual content from knowledge files
                profile_path = Path("agent_knowledge/profile.md")
                experience_path = Path("agent_knowledge/experience_and_projects.md")
                resume_path = Path("agent_knowledge/resume.md")
                
                # Read profile content
                if profile_path.exists():
                    with open(profile_path, 'r', encoding='utf-8') as f:
                        profile_content = f.read()
                else:
                    profile_content = "Profile information not found."
                
                # Read experience content
                if experience_path.exists():
                    with open(experience_path, 'r', encoding='utf-8') as f:
                        experience_content = f.read()
                else:
                    experience_content = "Experience information not found."
                
                # Read resume content
                if resume_path.exists():
                    with open(resume_path, 'r', encoding='utf-8') as f:
                        resume_content = f.read()
                else:
                    resume_content = "Resume information not found."
                
                # Extract summary from profile
                summary_match = re.search(r'## Summary\s*\n(.*?)(?=\n##|\n---|\Z)', profile_content, re.DOTALL)
                summary = summary_match.group(1).strip() if summary_match else "Professional with 4+ years of experience in full-stack development and AI/ML."
                
                # Extract skills sections from profile
                skills_section = ""
                
                # AI & ML Skills
                ai_ml_match = re.search(r'### AI & Machine Learning\s*\n(.*?)(?=\n###|\n##|\Z)', profile_content, re.DOTALL)
                ai_ml_skills = ai_ml_match.group(1).strip() if ai_ml_match else "- TensorFlow, PyTorch, Scikit-learn\n- LLMs, NLP, Deep Learning"
                
                # Backend Skills
                backend_match = re.search(r'### Backend Development\s*\n(.*?)(?=\n###|\n##|\Z)', profile_content, re.DOTALL)
                backend_skills = backend_match.group(1).strip() if backend_match else "- Python, Java, C++\n- FastAPI, Django, Spring Boot"
                
                # Frontend Skills
                frontend_match = re.search(r'### Frontend Development\s*\n(.*?)(?=\n###|\n##|\Z)', profile_content, re.DOTALL)
                frontend_skills = frontend_match.group(1).strip() if frontend_match else "- JavaScript, TypeScript, React\n- Node.js, Express.js"
                
                # Cloud & DevOps Skills
                cloud_match = re.search(r'### Cloud & DevOps\s*\n(.*?)(?=\n###|\n##|\Z)', profile_content, re.DOTALL)
                cloud_skills = cloud_match.group(1).strip() if cloud_match else "- AWS, Azure, Google Cloud\n- Docker, Kubernetes, CI/CD"
                
                # Extract experience section
                exp_match = re.search(r'## Experience\s*\n(.*?)(?=\n##|\Z)', profile_content, re.DOTALL)
                experience_summary = exp_match.group(1).strip() if exp_match else experience_content[:500] + "..."
                
                # Extract education section
                edu_match = re.search(r'## Education\s*\n(.*?)(?=\n##|\Z)', profile_content, re.DOTALL)
                education = edu_match.group(1).strip() if edu_match else "**Master of Science, Computer Science** - George Mason University (2022-2024)\n**Bachelor of Technology, Computer Science** - GITAM University (2017-2021)"
                
                # Extract projects section
                projects_match = re.search(r'## Projects\s*\n(.*?)(?=\n##|\Z)', profile_content, re.DOTALL)
                projects = projects_match.group(1).strip() if projects_match else experience_content[:300] + "..."
                
                # Format all skills
                all_skills = f"### AI & Machine Learning\n{ai_ml_skills}\n\n### Backend Development\n{backend_skills}\n\n### Frontend Development\n{frontend_skills}\n\n### Cloud & DevOps\n{cloud_skills}"
                
                return (
                    summary,  # summary_md
                    all_skills,  # skills_md
                    experience_summary,  # experience_md
                    education,  # education_md
                    projects,  # projects_md
                    ai_ml_skills,  # lang_skills (reusing for AI/ML)
                    backend_skills,  # backend_skills
                    frontend_skills,  # frontend_skills
                    "**Data Science & ML:** TensorFlow, PyTorch, Scikit-learn, LangChain, LlamaIndex",  # data_skills
                    "**DevOps & Tools:** Docker, Kubernetes, Jenkins, GitLab CI, GitHub Actions",  # devops_skills
                    cloud_skills  # cloud_skills
                )
                
            except Exception as e:
                logger.error(f"Error loading profile content: {e}")
                # Fallback content
                return (
                    "Full-stack developer with 4+ years of experience in AI/ML and web development.",
                    "**Skills:** Python, JavaScript, React, AWS, Docker, TensorFlow",
                    "Experience at Veritis Group Inc, TCS, and Virtusa",
                    "MS Computer Science - George Mason University",
                    "AI-powered testing, clinical decision support, loan origination platform",
                    "Python, Java, JavaScript",
                    "FastAPI, Django, Spring Boot",
                    "React, Node.js, Angular",
                    "TensorFlow, PyTorch, Scikit-learn",
                    "Docker, Kubernetes, Jenkins",
                    "AWS, Azure, Google Cloud"
                )

        def update_analytics():
            """Fetches and displays analytics data."""
            logger.info("📊 Fetching analytics data for dashboard...")
            interactions = client.get_analytics()
            
            if not interactions:
                # Return empty plots and a message if no data
                logger.info("No interaction data to plot. Displaying message.")
                no_data_df = pd.DataFrame({'message': ["No interaction data available yet."]})
                fig = px.text(no_data_df, x='message', y=[0], text_font_size=20)
                fig.update_layout(xaxis_visible=False, yaxis_visible=False, showlegend=False)
                return fig, fig, fig
            
            df = pd.DataFrame([{
                'timestamp': pd.to_datetime(ix.timestamp),
                'query': ix.query
            } for ix in interactions])

            # 1. Daily Interactions
            daily_counts = df.set_index('timestamp').resample('D').count()['query'].rename("count")
            fig_daily = px.bar(daily_counts, x=daily_counts.index, y='count', title="Total Daily Interactions")
            fig_daily.update_layout(xaxis_title="Date", yaxis_title="Number of Interactions")
            
            # 2. Hourly Activity
            hourly_counts = df['timestamp'].dt.hour.value_counts().sort_index().rename("count")
            fig_hourly = px.bar(hourly_counts, x=hourly_counts.index, y='count', title="User Activity by Hour")
            fig_hourly.update_layout(xaxis_title="Hour of Day", yaxis_title="Number of Interactions")

            # 3. Frequent Queries
            query_counts = df['query'].str.lower().value_counts().nlargest(10).rename("count")
            fig_frequent = px.bar(query_counts, x=query_counts.index, y='count', title="Top 10 Most Frequent Queries")
            fig_frequent.update_layout(xaxis_title="Query", yaxis_title="Count")

            return fig_daily, fig_hourly, fig_frequent
        
        # --- Event Handlers ---
        send_button.click(
            fn=chat_wrapper,
            inputs=[query_input, chatbot_window],
            outputs=[query_input, chatbot_window]
        )
        query_input.submit(
            fn=chat_wrapper,
            inputs=[query_input, chatbot_window],
            outputs=[query_input, chatbot_window]
        )

        dashboard_tab_item.select(
            fn=update_analytics,
            outputs=[daily_interactions_plot, hourly_activity_plot, frequent_queries_plot]
        )
        refresh_button.click(
            fn=update_analytics,
            outputs=[daily_interactions_plot, hourly_activity_plot, frequent_queries_plot]
        )

        # Pre-load profile data when its tab is selected
        profile_tab.select(
            fn=update_profile,
            outputs=[
                summary_md, skills_md, experience_md, education_md, projects_md,
                lang_skills, backend_skills, frontend_skills, data_skills, devops_skills, cloud_skills
            ]
        )

        def handle_contact_with_form_reset(name_val, email_val, msg_val):
            """Handle contact submission and clear form on success."""
            result = handle_contact_submission(name_val, email_val, msg_val)
            
            # If successful, clear the form fields
            if "Success!" in result:
                return result, "", "", ""  # Clear name, email, message
            else:
                return result, name_val, email_val, msg_val  # Keep values on error
        
        contact_btn.click(
            fn=handle_contact_with_form_reset,
            inputs=[name, email, message],
            outputs=[contact_status, name, email, message]
        )

    return demo

def main():
    """Initialize and launch the Gradio interface with a retry mechanism."""
    client = None
    max_retries = 5
    retry_delay = 3  # seconds

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to connect to the gRPC server...")
            # Check if the server is ready
            channel = grpc.insecure_channel('localhost:50051')
            grpc.channel_ready_future(channel).result(timeout=retry_delay)
            
            logger.info("✅ gRPC server is ready.")
            client = CareerAssistantClient()
            break  # Exit loop on successful connection
        except (grpc.FutureTimeoutError, grpc.RpcError) as e:
            logger.warning(f"Connection failed. Retrying in {retry_delay} seconds... Error: {e}")
            if attempt + 1 == max_retries:
                logger.critical("❌ Could not connect to gRPC server after multiple retries. Exiting.")
                # Display a user-friendly error in the UI if possible
                with gr.Blocks() as demo:
                    gr.Markdown("#  критическая ошибка\nНе удалось подключиться к серверу AI. Пожалуйста, перезапустите приложение.")
                demo.launch(server_name="0.0.0.0")
                return
            time.sleep(retry_delay)

    if client:
        demo = create_gradio_interface(client)
        demo.launch(server_name="0.0.0.0")

if __name__ == "__main__":
    main()