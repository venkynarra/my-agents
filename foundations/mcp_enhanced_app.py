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
    # Split content by lines and reconstruct sections
    current_section = None
    section_content = []

    # Define the headers to look for, case-insensitively
    headers = ["summary", "top skills", "experience", "education"]

    for line in md_content.splitlines():
        # Check if the line is a new section header
        line_lower = line.strip().lower()
        if line_lower in headers:
            # If we were in a section, save its content
            if current_section:
                sections[current_section] = "\n".join(section_content).strip()
            
            # Start the new section
            current_section = line_lower
            section_content = []
        elif current_section:
            # Add the line to the current section's content
            section_content.append(line.strip())

    # Save the last section
    if current_section:
        sections[current_section] = "\n".join(section_content).strip()

    return {
        "summary": sections.get("summary", "Not found."),
        "skills": sections.get("top skills", "Not found."),
        "experience": sections.get("experience", "Not found."),
        "education": sections.get("education", "Not found."),
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
    def __init__(self, host='localhost', port=50052):
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
            request = career_assistant_pb2.google_dot_protobuf_dot_empty__pb2.Empty()
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
            request = career_assistant_pb2.google_dot_protobuf_dot_empty__pb2.Empty()
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

                    with gr.TabItem("📊 Dashboard", id="dashboard_tab"):
                        gr.Markdown("<h1 style='text-align: center;'>AI Assistant Dashboard</h1>")
                        gr.Markdown("Welcome to my AI-Powered Career Hub! This dashboard provides real-time insights into user interactions with the assistant.")
                        with gr.Row():
                            daily_interactions_plot = gr.Plot(label="Daily Interactions")
                            hourly_activity_plot = gr.Plot(label="Hourly Activity")
                        with gr.Row():
                            frequent_queries_plot = gr.Plot(label="Most Frequent Queries")
                            
                    with gr.TabItem("👤 Profile", id="profile_tab"):
                        profile_content = gr.Markdown("Loading profile...")

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
                                gr.Markdown("Ready to connect? Book a time that works for you using my Calendly link below.")
                                gr.HTML(f'<a href="{CALENDLY_LINK}" target="_blank" style="display:inline-block;padding:10px 20px;background-color:#007bff;color:white;text-decoration:none;border-radius:5px;">Book a Meeting on Calendly</a>')


        # --- Event Handlers ---
        async def chat_wrapper(query, history):
            if not history:
                history = []
            history.append({"role": "user", "content": query})
            response = await asyncio.to_thread(client.process_query, query, history)
            history.append({"role": "assistant", "content": response})
            return "", history

        async def handle_contact_submission(name_val, email_val, msg_val):
            response_msg = await asyncio.to_thread(client.submit_contact_form, name_val, email_val, msg_val)
            return gr.update(value=f"**{response_msg}**", visible=True)

        def update_profile():
            """Fetches the AI-generated profile and updates the UI."""
            content = client.generate_profile()
            return content

        def update_analytics():
            """Fetches analytics and returns a Plotly figure."""
            interactions = client.get_analytics()
            if not interactions:
                # Return empty plots if no data
                return None, None, None
            
            df = pd.DataFrame(
                [
                    {
                        "timestamp": datetime.strptime(ix.timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S'),
                        "query": ix.query.strip().lower(),
                    }
                    for ix in interactions
                ]
            )
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            
            # Plot 1: Daily Interactions
            daily_counts = df.groupby('date').size().reset_index(name='counts')
            fig_daily = px.bar(
                daily_counts, x='date', y='counts', title="Daily Interactions",
                labels={'date': 'Date', 'counts': 'Number of Interactions'}, template="plotly_dark"
            )

            # Plot 2: Hourly Activity
            hourly_counts = df.groupby('hour').size().reset_index(name='counts')
            fig_hourly = px.bar(
                hourly_counts, x='hour', y='counts', title="Interactions by Hour of Day",
                labels={'hour': 'Hour of Day', 'counts': 'Number of Interactions'}, template="plotly_dark"
            )

            # Plot 3: Frequent Queries
            query_counts = df['query'].value_counts().nlargest(5).reset_index()
            query_counts.columns = ['query', 'counts']
            fig_queries = px.bar(
                query_counts, y='query', x='counts', title="Top 5 Most Frequent Queries",
                orientation='h', labels={'query': 'Query', 'counts': 'Count'}, template="plotly_dark"
            )
            fig_queries.update_yaxes(autorange="reversed")

            return fig_daily, fig_hourly, fig_queries

        # Link data loading to tab selection/app load
        demo.load(update_analytics, [], [daily_interactions_plot, hourly_activity_plot, frequent_queries_plot])
        demo.load(update_profile, [], [profile_content])

        query_input.submit(chat_wrapper, [query_input, chatbot_window], [query_input, chatbot_window])
        send_button.click(chat_wrapper, [query_input, chatbot_window], [query_input, chatbot_window])
        contact_btn.click(handle_contact_submission, [name, email, message], [contact_status])

    return demo

def main():
    # Initialize and launch the Gradio interface
    client = CareerAssistantClient()
    interface = create_gradio_interface(client)
    interface.launch(server_name="0.0.0.0", server_port=7862, share=True)
    logger.info("🚀 Gradio UI is running at http://localhost:7862")

if __name__ == "__main__":
    main() 