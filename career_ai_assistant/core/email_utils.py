import os
import asyncio
from typing import Dict, Optional
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class EmailManager:
    def __init__(self):
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        self.from_email = os.getenv("FROM_EMAIL", "venkynarra20@gmail.com")
        self.to_email = os.getenv("TO_EMAIL", "venkynarra20@gmail.com")
        
        if self.sendgrid_api_key:
            self.sg = SendGridAPIClient(api_key=self.sendgrid_api_key)
        else:
            self.sg = None
            logger.warning("SendGrid API key not found. Email functionality disabled.")
    
    async def send_contact_form_email(self, form_data: Dict) -> bool:
        """
        Send contact form submission email.
        """
        if not self.sg:
            logger.error("SendGrid not configured")
            return False
            
        try:
            subject = f"New Contact Form Submission from {form_data.get('name', 'Unknown')}"
            
            # Create email content
            content = f"""
            New contact form submission:
            
            Name: {form_data.get('name', 'Not provided')}
            Email: {form_data.get('email', 'Not provided')}
            Subject: {form_data.get('subject', 'Not provided')}
            Message: {form_data.get('message', 'Not provided')}
            
            Submitted at: {form_data.get('timestamp', 'Unknown')}
            """
            
            message = Mail(
                from_email=Email(self.from_email),
                to_emails=To(self.to_email),
                subject=subject,
                plain_text_content=Content("text/plain", content)
            )
            
            # Send email
            response = await asyncio.to_thread(self.sg.send, message)
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Contact form email sent successfully: {response.status_code}")
                return True
            else:
                logger.error(f"Failed to send email: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending contact form email: {e}")
            return False
    
    async def send_meeting_request_email(self, meeting_data: Dict) -> bool:
        """
        Send meeting request email.
        """
        if not self.sg:
            logger.error("SendGrid not configured")
            return False
            
        try:
            subject = f"Meeting Request from {meeting_data.get('name', 'Unknown')}"
            
            content = f"""
            New meeting request:
            
            Name: {meeting_data.get('name', 'Not provided')}
            Email: {meeting_data.get('email', 'Not provided')}
            Preferred Date: {meeting_data.get('preferred_date', 'Not provided')}
            Preferred Time: {meeting_data.get('preferred_time', 'Not provided')}
            Meeting Type: {meeting_data.get('meeting_type', 'Not provided')}
            Additional Notes: {meeting_data.get('notes', 'Not provided')}
            
            Requested at: {meeting_data.get('timestamp', 'Unknown')}
            """
            
            message = Mail(
                from_email=Email(self.from_email),
                to_emails=To(self.to_email),
                subject=subject,
                plain_text_content=Content("text/plain", content)
            )
            
            response = await asyncio.to_thread(self.sg.send, message)
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Meeting request email sent successfully: {response.status_code}")
                return True
            else:
                logger.error(f"Failed to send meeting request email: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending meeting request email: {e}")
            return False
    
    async def send_notification_email(self, subject: str, message: str) -> bool:
        """
        Send general notification email.
        """
        if not self.sg:
            logger.error("SendGrid not configured")
            return False
            
        try:
            mail = Mail(
                from_email=Email(self.from_email),
                to_emails=To(self.to_email),
                subject=subject,
                plain_text_content=Content("text/plain", message)
            )
            
            response = await asyncio.to_thread(self.sg.send, mail)
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Notification email sent successfully: {response.status_code}")
                return True
            else:
                logger.error(f"Failed to send notification email: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending notification email: {e}")
            return False

# Global email manager instance
email_manager = EmailManager() 