# extractors/email_extractor.py
import email
from email import policy
from email.parser import BytesParser
from typing import Dict, List
import logging
from html import unescape
import re

logger = logging.getLogger(__name__)

class EmailExtractor:
    def extract_text(self, file) -> str:
        """Extract text from .eml file with comprehensive handling"""
        try:
            # Parse the email with error handling
            if hasattr(file, 'read'):
                # Reset position first
                if hasattr(file, 'seek'):
                    file.seek(0)
                msg = BytesParser(policy=policy.default).parse(file)
            else:
                # If it's a path string
                with open(file, 'rb') as f:
                    msg = BytesParser(policy=policy.default).parse(f)
            
            # Extract headers safely
            text = "=== EMAIL CONTENT ===\n\n"
            text += f"From: {msg.get('From', 'Unknown')}\n"
            text += f"To: {msg.get('To', 'Unknown')}\n"
            text += f"Subject: {msg.get('Subject', 'No Subject')}\n"
            text += f"Date: {msg.get('Date', 'Unknown')}\n"
            
            # Add CC if present
            if msg.get('Cc'):
                text += f"Cc: {msg['Cc']}\n"
            
            text += "\n--- BODY ---\n"
            
            # Get email body with better extraction
            body = self.get_email_body(msg)
            text += body + "\n"
            
            # List attachments
            attachments = self.get_attachments(msg)
            if attachments:
                text += f"\n--- ATTACHMENTS ({len(attachments)}) ---\n"
                for att in attachments:
                    text += f"- {att}\n"
            
            # Add email metadata
            text += f"\n--- METADATA ---\n"
            text += f"Message-ID: {msg.get('Message-ID', 'Unknown')}\n"
            text += f"Content-Type: {msg.get_content_type()}\n"
            
            logger.info(f"Extracted {len(text)} characters from email")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting email: {str(e)}")
            return f"Error extracting email content: {str(e)}"
    
    def get_email_body(self, msg) -> str:
        """Extract email body text with comprehensive handling"""
        body_parts = []
        
        try:
            if msg.is_multipart():
                # Walk through all parts
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition", ""))
                    
                    # Skip attachments
                    if "attachment" in content_disposition:
                        continue
                    
                    # Extract text parts
                    if content_type == "text/plain":
                        try:
                            text = part.get_content()
                            if isinstance(text, str):
                                body_parts.append(("Plain Text", text))
                        except Exception as e:
                            logger.warning(f"Error extracting plain text: {e}")
                    
                    elif content_type == "text/html":
                        try:
                            html_content = part.get_content()
                            if isinstance(html_content, str):
                                # Basic HTML to text conversion
                                text = self.html_to_text(html_content)
                                body_parts.append(("HTML (converted)", text))
                        except Exception as e:
                            logger.warning(f"Error extracting HTML: {e}")
            else:
                # Single part message
                content_type = msg.get_content_type()
                try:
                    content = msg.get_content()
                    if content_type == "text/html":
                        content = self.html_to_text(content)
                    body_parts.append((content_type, content))
                except Exception as e:
                    logger.warning(f"Error extracting single part: {e}")
            
            # Combine all body parts
            if not body_parts:
                return "[No readable content found in email body]"
            
            # Prefer plain text over HTML
            plain_text_parts = [text for (type, text) in body_parts if "Plain" in type]
            if plain_text_parts:
                return "\n\n".join(plain_text_parts)
            else:
                return "\n\n".join([text for (type, text) in body_parts])
                
        except Exception as e:
            logger.error(f"Error in get_email_body: {e}")
            return f"[Error extracting body: {str(e)}]"
    
    def html_to_text(self, html: str) -> str:
        """Basic HTML to text conversion"""
        try:
            # Remove script and style elements
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
            
            # Replace breaks and paragraphs with newlines
            html = re.sub(r'<br\s*/?>', '\n', html)
            html = re.sub(r'</p>', '\n\n', html)
            html = re.sub(r'</div>', '\n', html)
            
            # Remove all other tags
            html = re.sub(r'<[^>]+>', '', html)
            
            # Unescape HTML entities
            text = unescape(html)
            
            # Clean up excessive whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r' +', ' ', text)
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error converting HTML to text: {e}")
            return html
    
    def get_attachments(self, msg) -> List[str]:
        """Get list of attachment filenames"""
        attachments = []
        
        try:
            for part in msg.walk():
                content_disposition = str(part.get("Content-Disposition", ""))
                
                if "attachment" in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        attachments.append(filename)
                    else:
                        # Try to generate a filename
                        content_type = part.get_content_type()
                        attachments.append(f"unnamed_{content_type.replace('/', '_')}")
        except Exception as e:
            logger.error(f"Error getting attachments: {e}")
                    
        return attachments