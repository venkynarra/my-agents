"""
Email sending utilities for the Career Assistant.
"""
import logging
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from .config import SENDGRID_API_KEY, FROM_EMAIL

logger = logging.getLogger(__name__)

def send_contact_email(name: str, email: str, message: str) -> bool:
    """
    Sends a contact form email TO the user who submitted the form.
    This serves as an acknowledgment/confirmation email.
    """
    try:
        if not SENDGRID_API_KEY or SENDGRID_API_KEY == "YOUR_SENDGRID_API_KEY_HERE":
            logger.error("SendGrid API key not configured properly.")
            return False
            
        sg = SendGridAPIClient(api_key=SENDGRID_API_KEY)
        
        # Email content - acknowledgment to the user
        subject = "Thank you for contacting Venkatesh Narra!"
        
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2 style="color: #2563eb;">Thank you for your message!</h2>
            
            <p>Hi {name},</p>
            
            <p>Thank you for reaching out! I have received your message and will get back to you as soon as possible.</p>
            
            <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #2563eb; margin: 20px 0;">
                <h3 style="margin-top: 0; color: #2563eb;">Your Message:</h3>
                <p style="margin-bottom: 0; font-style: italic;">"{message}"</p>
            </div>
            
            <p>I typically respond within 24-48 hours. In the meantime, feel free to:</p>
            <ul>
                <li>Check out my <a href="https://github.com/venkynarra" style="color: #2563eb;">GitHub profile</a></li>
                <li>Connect with me on <a href="https://linkedin.com/in/venkatesh-narra" style="color: #2563eb;">LinkedIn</a></li>
                <li>Schedule a meeting directly using my <a href="https://calendly.com/venkateshnarra368" style="color: #2563eb;">Calendly link</a></li>
            </ul>
            
            <p>Best regards,<br>
            <strong>Venkatesh Narra</strong><br>
            Full-Stack Python Developer | AI/ML Engineer</p>
            
            <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
            <p style="font-size: 12px; color: #666;">
                This is an automated confirmation email. Please do not reply to this email.
            </p>
        </body>
        </html>
        """
        
        plain_text_content = f"""
        Thank you for your message!
        
        Hi {name},
        
        Thank you for reaching out! I have received your message and will get back to you as soon as possible.
        
        Your Message: "{message}"
        
        I typically respond within 24-48 hours. In the meantime, feel free to check out my GitHub profile or connect with me on LinkedIn.
        
        Best regards,
        Venkatesh Narra
        Full-Stack Python Developer | AI/ML Engineer
        
        This is an automated confirmation email.
        """
        
        mail = Mail(
            from_email=FROM_EMAIL,
            to_emails=email,  # Send TO the user's email address
            subject=subject,
            html_content=html_content,
            plain_text_content=plain_text_content
        )
        
        response = sg.send(mail)
        logger.info(f"✅ Confirmation email sent successfully to {email}. Status: {response.status_code}")
        return True
        
    except Exception as e:
        logger.error(f"❌ An exception occurred while sending email: {e}")
        return False
    
