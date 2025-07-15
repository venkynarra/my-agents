import gradio as gr
import asyncio
import time
from datetime import datetime
from career_ai_assistant.router.router import route_query
from career_ai_assistant.monitoring.logging import performance_monitor
from career_ai_assistant.core.email_utils import email_manager

async def chat_interface(message: str, history: list) -> tuple:
    """
    Main chat interface that routes queries through the architecture.
    Returns history as a list of dicts with 'role' and 'content' keys (Gradio messages format).
    """
    start_time = time.time()
    try:
        # Route the query through our architecture
        result = await route_query(message)
        response_time = time.time() - start_time
        # Log the request
        await performance_monitor.log_request(
            query=message,
            response_time=response_time,
            source=result['source'],
            cached=result['cached']
        )
        # Format response based on source
        if result['cached']:
            response = f"💾 {result['response']} (cached)"
        else:
            response = result['response']
        user_msg = {"role": "user", "content": message}
        bot_msg = {"role": "assistant", "content": response}
        return "", history + [user_msg, bot_msg]
    except Exception as e:
        # Log error
        await performance_monitor.log_error(e, "chat_interface")
        error_response = f"Sorry, I encountered an error: {str(e)}"
        user_msg = {"role": "user", "content": message}
        error_msg = {"role": "assistant", "content": error_response}
        return "", history + [user_msg, error_msg]

async def submit_contact_form(name: str, email: str, subject: str, message: str) -> str:
    """
    Handle contact form submission and send email.
    """
    try:
        form_data = {
            'name': name,
            'email': email,
            'subject': subject,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send email
        success = await email_manager.send_contact_form_email(form_data)
        
        if success:
            return "✅ Thank you! Your message has been sent successfully. I'll get back to you soon!"
        else:
            return "❌ Sorry, there was an error sending your message. Please try again or contact me directly."
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

async def submit_meeting_request(name: str, email: str, preferred_date: str, preferred_time: str, meeting_type: str, notes: str) -> str:
    """
    Handle meeting request submission and send email.
    """
    try:
        meeting_data = {
            'name': name,
            'email': email,
            'preferred_date': preferred_date,
            'preferred_time': preferred_time,
            'meeting_type': meeting_type,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send email
        success = await email_manager.send_meeting_request_email(meeting_data)
        
        if success:
            return "✅ Meeting request sent successfully! I'll review your request and get back to you soon."
        else:
            return "❌ Sorry, there was an error sending your meeting request. Please try again."
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="Venkatesh's AI Career Assistant",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1000px;
            margin: auto;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🤖 Venkatesh's AI Career Assistant
        
        Hi! I'm Venkatesh, a senior software engineer with 4+ years of experience.
        Ask me about my skills, projects, experience, or anything else!
        
        **Features:**
        - ⚡ Sub-2 second responses
        - 🧠 Smart routing and caching
        - 📚 Dynamic knowledge base
        - 🔄 Fallback systems
        - 📧 Contact form integration
        - 📅 Meeting scheduling
        """)
        
        with gr.Tabs():
            # Chat Tab
            with gr.TabItem("💬 Chat"):
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    container=True,
                    bubble_full_width=False,
                    type="messages"
                )
                
                msg = gr.Textbox(
                    placeholder="Ask me about my skills, experience, projects, or anything else...",
                    show_label=False,
                    container=False
                )
                
                clear = gr.Button("Clear Chat")
                
                # Event handlers
                msg.submit(
                    chat_interface,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot],
                    show_progress=True
                )
                
                clear.click(
                    lambda: ([], ""),
                    outputs=[chatbot, msg]
                )
            
            # Contact Form Tab
            with gr.TabItem("📧 Contact"):
                with gr.Row():
                    with gr.Column():
                        contact_name = gr.Textbox(
                            label="Name",
                            placeholder="Your name"
                        )
                        contact_email = gr.Textbox(
                            label="Email",
                            placeholder="your.email@example.com"
                        )
                    with gr.Column():
                        contact_subject = gr.Textbox(
                            label="Subject",
                            placeholder="What's this about?"
                        )
                
                contact_message = gr.Textbox(
                    label="Message",
                    placeholder="Tell me more about your inquiry...",
                    lines=5
                )
                
                contact_submit = gr.Button("Send Message", variant="primary")
                contact_result = gr.Textbox(label="Status", interactive=False)
                
                contact_submit.click(
                    submit_contact_form,
                    inputs=[contact_name, contact_email, contact_subject, contact_message],
                    outputs=contact_result
                )
            
            # Meeting Request Tab
            with gr.TabItem("📅 Schedule Meeting"):
                with gr.Row():
                    with gr.Column():
                        meeting_name = gr.Textbox(
                            label="Name",
                            placeholder="Your name"
                        )
                        meeting_email = gr.Textbox(
                            label="Email",
                            placeholder="your.email@example.com"
                        )
                    with gr.Column():
                        meeting_date = gr.Textbox(
                            label="Preferred Date",
                            placeholder="YYYY-MM-DD"
                        )
                        meeting_time = gr.Textbox(
                            label="Preferred Time",
                            placeholder="e.g., 2:00 PM"
                        )
                
                meeting_type = gr.Dropdown(
                    choices=["Technical Discussion", "Project Collaboration", "Career Advice", "General Chat"],
                    label="Meeting Type",
                    value="Technical Discussion"
                )
                
                meeting_notes = gr.Textbox(
                    label="Additional Notes",
                    placeholder="Any specific topics or questions you'd like to discuss?",
                    lines=3
                )
                
                meeting_submit = gr.Button("Request Meeting", variant="primary")
                meeting_result = gr.Textbox(label="Status", interactive=False)
                
                meeting_submit.click(
                    submit_meeting_request,
                    inputs=[meeting_name, meeting_email, meeting_date, meeting_time, meeting_type, meeting_notes],
                    outputs=meeting_result
                )
            
            # Performance Stats Tab
            with gr.TabItem("📊 Performance"):
                stats_btn = gr.Button("Get Performance Stats")
                stats_output = gr.JSON()
                
                async def get_stats():
                    stats = await performance_monitor.get_stats()
                    return stats
                    
                stats_btn.click(
                    get_stats,
                    outputs=stats_output
                )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    ) 