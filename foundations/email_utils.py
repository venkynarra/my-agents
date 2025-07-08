"""
Email sending utilities for the Career Assistant.
"""
import logging
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from .config import SENDGRID_API_KEY, SENDER_EMAIL, RECIPIENT_EMAIL

logger = logging.getLogger(__name__)

def send_contact_email(name: str, email: str, message: str) -> bool:
    """
    Sends an email notification for a new contact form submission using SendGrid.

    Args:
        name: The name of the person submitting the form.
        email: The email address of the person submitting the form.
        message: The message content.

    Returns:
        True if the email was sent successfully, False otherwise.
    """
    if SENDGRID_API_KEY == "YOUR_SENDGRID_API_KEY_HERE":
        logger.warning("SendGrid API key not configured. Skipping email.")
        return False

    mail_message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=RECIPIENT_EMAIL,
        subject=f"New Contact Form Submission from {name}",
        html_content=f"""
            <h3>You have a new message from your AI Career Assistant!</h3>
            <p><strong>Name:</strong> {name}</p>
            <p><strong>Email:</strong> {email}</p>
            <hr>
            <p><strong>Message:</strong></p>
            <p>{message}</p>
        """
    )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(mail_message)
        if response.status_code >= 200 and response.status_code < 300:
            logger.info(f"✅ Contact form email sent successfully to {RECIPIENT_EMAIL}")
            return True
        else:
            # Log the detailed error from SendGrid's response body
            error_details = response.body.decode('utf-8') if response.body else "No additional details provided."
            logger.error(f"❌ Failed to send email. Status: {response.status_code}. Details: {error_details}")
            return False
    except Exception as e:
        # Catch potential HTTP errors to get more details
        if hasattr(e, 'body'):
            error_details = e.body.decode('utf-8') if e.body else "No additional details."
            logger.error(f"❌ An exception occurred while sending email: {e}. Details: {error_details}")
        else:
            logger.error(f"❌ An exception occurred while sending email: {e}")
        return False
    
