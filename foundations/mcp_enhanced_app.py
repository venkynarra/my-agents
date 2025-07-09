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
            return f"Error: Could not submit form. gRPC stub not initialized."
        try:
            request = career_assistant_pb2.ContactFormRequest(name=name, email=email, message=message)
            response = self.stub.SubmitContactForm(request)
            return response.message
        except grpc.RpcError as e:
            logger.error(f"gRPC error on contact form: {e.details()} (code: {e.code().name})")
            return f"Error: Could not submit form. {e.details()}"

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
                            
                    with gr.TabItem("👤 Profile", id="profile_tab"):
                        gr.Markdown("<h1 style='text-align: center;'>Professional Profile</h1>")
                        with gr.Accordion("Executive Summary", open=True):
                            summary_md = gr.Markdown("Loading...")
                        with gr.Accordion("Top Skills", open=True):
                            skills_categorized_md = gr.Markdown("Loading...")
                        with gr.Accordion("Projects", open=False):
                            projects_md = gr.Markdown("Loading...")
                        with gr.Accordion("Professional Experience", open=False):
                            experience_md = gr.Markdown("Loading...")
                        with gr.Accordion("Education", open=False):
                            education_md = gr.Markdown("Loading...")

                    with gr.TabItem("✉️ Contact & Schedule", id="contact_tab"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Get in Touch")
                                gr.Markdown("Have a question or a project proposal? Fill out the form below.")
                                with gr.Column():
                                    name = gr.Textbox(label="Your Name")
                                    email = gr.Textbox(label="Your Email")
                                    message = gr.Textbox(label="Message", lines=4)
                                    contact_btn = gr.Button("Send", variant="primary")
                                contact_status = gr.Markdown(visible=False)
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

        async def handle_contact_submission(name_val, email_val, msg_val):
            """Handles the submission of the contact form."""
            return client.submit_contact_form(name_val, email_val, msg_val)

        def update_profile():
            """Parses the profile markdown and updates the UI components."""
            parsed_sections = parse_profile_md(PROFILE_MD)
            
            # Format categorized skills for display
            skills_text = ""
            for category, skills in parsed_sections['skills_categorized'].items():
                skills_text += f"**{category}**\n"
                skills_text += "\n".join(skills) + "\n\n"
                
            return (
                parsed_sections["summary"],
                skills_text,
                parsed_sections["projects"],
                parsed_sections["experience"],
                parsed_sections["education"]
            )

        def update_analytics():
            """Fetches analytics and updates the plots."""
            logger.info("📊 Fetching analytics data for dashboard...")
            interactions = client.get_analytics()
            
            if not interactions:
                logger.warning("No interaction data to plot.")
                # Create empty plots with titles to avoid errors
                fig_daily = px.bar(title='Daily Interactions').update_layout(template="plotly_dark")
                fig_hourly = px.line(title='Hourly Activity').update_layout(template="plotly_dark")
                fig_frequent = px.treemap(title='Most Frequent Queries').update_layout(template="plotly_dark")
                return fig_daily, fig_hourly, fig_frequent

            # Convert to DataFrame
            df = pd.DataFrame([{
                "timestamp": ix.timestamp.split('.')[0],
                "query": ix.query.strip().lower(),
            } for ix in interactions])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 1. Daily Interactions
            daily_counts = df.resample('D', on='timestamp').size().reset_index(name='count')
            fig_daily = px.bar(daily_counts, x='timestamp', y='count', title='Daily Interactions', labels={'timestamp': 'Date', 'count': 'Interactions'})
            fig_daily.update_layout(template="plotly_dark")

            # 2. Hourly Activity
            df['hour'] = df['timestamp'].dt.hour
            hourly_counts = df.groupby('hour').size().reset_index(name='count')
            fig_hourly = px.line(hourly_counts, x='hour', y='count', title='Hourly Activity', markers=True, labels={'hour': 'Hour of Day', 'count': 'Interactions'})
            fig_hourly.update_layout(template="plotly_dark")

            # 3. Most Frequent Queries (using regex to simplify)
            df['clean_query'] = df['query'].str.lower().str.replace(r'[^a-z\s]', '', regex=True).str.strip()
            query_counts = df['clean_query'].value_counts().nlargest(10).reset_index()
            query_counts.columns = ['query', 'count']
            fig_frequent = px.treemap(query_counts, path=['query'], values='count', title='Most Frequent Queries')
            fig_frequent.update_layout(template="plotly_dark", treemapcolorway = ["#7B68EE", "#50C878", "#1E90FF", "#FFD700", "#FF4500"])
            
            return fig_daily, fig_hourly, fig_frequent
            
        # --- Event Handlers ---
        # Chatbot interactions
        query_input.submit(chat_wrapper, [query_input, chatbot_window], [query_input, chatbot_window])
        send_button.click(chat_wrapper, [query_input, chatbot_window], [query_input, chatbot_window])

        # Contact form
        contact_btn.click(
            handle_contact_submission, 
            [name, email, message], 
            contact_status
        )

        # Initial data load for tabs
        demo.load(update_profile, outputs=[summary_md, skills_categorized_md, projects_md, experience_md, education_md])
        demo.load(update_analytics, outputs=[daily_interactions_plot, hourly_activity_plot, frequent_queries_plot])

        # Refresh analytics when the dashboard tab is selected or the refresh button is clicked
        dashboard_tab_item.select(
            fn=update_analytics, 
            outputs=[daily_interactions_plot, hourly_activity_plot, frequent_queries_plot]
        )
        refresh_button.click(
            fn=update_analytics, 
            outputs=[daily_interactions_plot, hourly_activity_plot, frequent_queries_plot]
        )
    
    # Launch the Gradio app
    demo.launch(server_name="0.0.0.0")

def main():
    # Initialize and launch the Gradio interface
    client = CareerAssistantClient()
    demo = create_gradio_interface(client)
    # Launch the Gradio app with server_name="0.0.0.0" to be accessible externally
    demo.launch(server_name="0.0.0.0")

if __name__ == "__main__":
    main()