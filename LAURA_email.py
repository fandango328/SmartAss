from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.message import EmailMessage
import base64
from datetime import datetime
import json
from typing import Dict, List, Union, Optional
from googleapiclient.errors import HttpError
import traceback

class EmailManager:
    def __init__(self, credentials: Credentials):
        """Initialize EmailManager with Google credentials"""
        self.credentials = credentials
        try:
            self.service = build("gmail", "v1", credentials=self.credentials)
            print("Gmail service initialized successfully")
        except Exception as e:
            print(f"Error initializing Gmail service: {e}")
            raise

        self.last_email_results = {
            "emails": [],
            "timestamp": None
        }

        # Email importance configuration
        self.importance_config = {
            "important_senders": [
                "boss@company.com",
                "vp@company.com",
                "@executive-team.company.com",
            ],
            "action_keywords": [
                "urgent",
                "action required", 
                "action item",
                "please review",
                "deadline",
                "asap",
                "by tomorrow",
                "by eod",
                "assigned to you",
            ],
            "important_topics": [
                "quarterly review",
                "performance review",
                "key project",
                "budget approval",
            ]
        }

    def extract_email_body(self, payload: Dict) -> str:
        """Extract email body content from Gmail API response"""
        try:
            body = ""
            
            if "body" in payload and payload["body"].get("data"):
                body = base64.urlsafe_b64decode(payload["body"]["data"].encode("ASCII")).decode("utf-8")
            elif "parts" in payload:
                for part in payload["parts"]:
                    if part["mimeType"] == "text/plain" and part["body"].get("data"):
                        body = base64.urlsafe_b64decode(part["body"]["data"].encode("ASCII")).decode("utf-8")
                        break
                    elif "parts" in part:
                        body = self.extract_email_body(part)
                        if body:
                            break
            
            return body
        except Exception as e:
            print(f"Error extracting email body: {e}")
            traceback.print_exc()
            return ""

    def determine_email_importance(self, email: Dict) -> tuple:
        """
        Analyze email importance and provide explanation
        Returns: (importance_level: int, explanation: str)
        """
        try:
            importance_score = 0
            reasons = []
            
            sender_email = email.get('sender_email', '').lower()
            
            # Check sender importance
            if any(important_sender.lower() == sender_email 
                  for important_sender in self.importance_config["important_senders"] 
                  if not important_sender.startswith('@')):
                importance_score += 3
                sender_name = email.get('sender_name', sender_email)
                reasons.append(f"From important person: {sender_name}")
            elif any(domain.lower() in sender_email 
                    for domain in self.importance_config["important_senders"] 
                    if domain.startswith('@')):
                importance_score += 2
                reasons.append(f"From important team")
            
            # Check action items/urgency
            email_text = (email.get('subject', '') + ' ' + email.get('snippet', '')).lower()
            action_matches = [keyword for keyword in self.importance_config["action_keywords"] 
                            if keyword.lower() in email_text]
            if action_matches:
                importance_score += 2
                reasons.append(f"Contains action keywords: {', '.join(action_matches[:2])}")
            
            # Check important topics
            topic_matches = [topic for topic in self.importance_config["important_topics"] 
                           if topic.lower() in email_text]
            if topic_matches:
                importance_score += 1
                reasons.append(f"Related to important topic: {', '.join(topic_matches[:2])}")
            
            final_score = min(5, importance_score)
            explanation = "; ".join(reasons) if reasons else "No special importance detected"
            
            return (final_score, explanation)
            
        except Exception as e:
            print(f"Error determining email importance: {e}")
            traceback.print_exc()
            return (0, "Error analyzing importance")

    def read_emails(self, max_results: int = 5, unread_only: bool = False, 
                   query: str = None, include_content: bool = False, 
                   mark_as_read: bool = False) -> str:
        """Retrieve and read emails from Gmail with importance detection"""
        try:
            query_parts = []
            if unread_only:
                query_parts.append("is:unread")
            if query:
                query_parts.append(query)
            
            final_query = " ".join(query_parts) if query_parts else None
            
            messages_result = self.service.users().messages().list(
                userId="me",
                q=final_query,
                maxResults=max_results
            ).execute()
            
            messages = messages_result.get("messages", [])
            
            if not messages:
                return "No emails found matching your criteria."

            emails = []
            important_emails = []
            regular_emails = []
            
            for message in messages:
                msg = self.service.users().messages().get(
                    userId="me", 
                    id=message["id"],
                    format="full" if include_content else "metadata"
                ).execute()
                
                headers = {header["name"].lower(): header["value"] 
                          for header in msg["payload"]["headers"]}
                
                email_data = {
                    "id": msg["id"],
                    "thread_id": msg["threadId"],
                    "subject": headers.get("subject", "(No subject)"),
                    "sender_name": headers.get("from", "").split("<")[0].strip(),
                    "sender_email": headers.get("from", ""),
                    "date": headers.get("date", ""),
                    "to": headers.get("to", ""),
                    "cc": headers.get("cc", ""),
                    "labels": msg["labelIds"],
                    "snippet": msg.get("snippet", ""),
                    "unread": "UNREAD" in msg["labelIds"]
                }
                
                if "<" in email_data["sender_email"] and ">" in email_data["sender_email"]:
                    email_data["sender_email"] = email_data["sender_email"].split("<")[1].split(">")[0]
                
                if include_content:
                    email_data["body"] = self.extract_email_body(msg["payload"])
                
                importance_score, importance_reason = self.determine_email_importance(email_data)
                email_data["importance"] = importance_score
                email_data["importance_reason"] = importance_reason
                
                emails.append(email_data)
                
                if importance_score >= 3:
                    important_emails.append(email_data)
                else:
                    regular_emails.append(email_data)

            if mark_as_read:
                email_ids_to_mark = [email["id"] for email in emails if email["unread"]]
                if email_ids_to_mark:
                    self.service.users().messages().batchModify(
                        userId="me",
                        body={
                            "ids": email_ids_to_mark,
                            "removeLabelIds": ["UNREAD"]
                        }
                    ).execute()
                    
                    for email in emails:
                        if email["id"] in email_ids_to_mark:
                            email["unread"] = False
                            if "UNREAD" in email["labels"]:
                                email["labels"].remove("UNREAD")

            self.last_email_results = {
                "emails": emails,
                "timestamp": datetime.now(),
                "important": important_emails,
                "regular": regular_emails
            }

            response = ""
            if important_emails:
                response += f"You have {len(important_emails)} important email{'s' if len(important_emails) > 1 else ''}:\n\n"
                for i, email in enumerate(important_emails, 1):
                    response += f"{i}. From: {email['sender_name']} - {email['subject']}\n"
                    response += f"   {email['importance_reason']}\n"
                    response += f"   {email['snippet'][:100]}...\n\n"

            if regular_emails:
                if important_emails:
                    response += f"You also have {len(regular_emails)} other email{'s' if len(regular_emails) > 1 else ''}:\n\n"
                else:
                    response += f"You have {len(regular_emails)} email{'s' if len(regular_emails) > 1 else ''}:\n\n"
                
                for i, email in enumerate(regular_emails, 1):
                    response += f"{i}. From: {email['sender_name']} - {email['subject']}\n"
                    response += f"   {email['snippet'][:50]}...\n\n"

            response += "You can ask me to read any specific email in full or take actions like marking them as read or archiving."
            return response

        except Exception as e:
            print(f"Error reading emails: {e}")
            traceback.print_exc()
            return f"Sorry, I encountered an error while trying to read your emails: {str(e)}"

    def email_action(self, email_id: str, action: str) -> str:
        """Perform action on a specific email"""
        try:
            actions = {
                "archive": ({"removeLabelIds": ["INBOX"]}, "archived"),
                "mark_read": ({"removeLabelIds": ["UNREAD"]}, "marked as read"),
                "mark_unread": ({"addLabelIds": ["UNREAD"]}, "marked as unread"),
                "star": ({"addLabelIds": ["STARRED"]}, "starred"),
                "unstar": ({"removeLabelIds": ["STARRED"]}, "unstarred")
            }

            if action == "trash":
                self.service.users().messages().trash(
                    userId="me",
                    id=email_id
                ).execute()
                return "Email moved to trash."
            
            if action not in actions:
                return f"Unknown action: {action}"

            modification, status = actions[action]
            self.service.users().messages().modify(
                userId="me",
                id=email_id,
                body=modification
            ).execute()
            
            return f"Email {status} successfully."

        except Exception as e:
            print(f"Error performing email action: {e}")
            traceback.print_exc()
            return f"Sorry, I encountered an error while trying to {action} the email: {str(e)}"

    def draft_email(self, subject: str, content: str, recipient: str = "") -> str:
        """Create an email draft"""
        try:
            message = EmailMessage()
            message.set_content(content)
            if recipient:
                message["To"] = recipient
            message["Subject"] = subject
            
            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            create_message = {"message": {"raw": encoded_message}}
            
            draft = self.service.users().drafts().create(
                userId="me", 
                body=create_message
            ).execute()

            if not draft or "message" not in draft:
                raise ValueError("Invalid response from Gmail API")
                
            return "Email has been drafted successfully."
            
        except HttpError as error:
            print(f"Gmail API error: {error}")
            traceback.print_exc()
            return f"There was an error trying to draft an email: {error}"
        except Exception as e:
            print(f"Unexpected error in draft_email: {e}")
            traceback.print_exc()
            return f"An unexpected error occurred while drafting the email: {str(e)}"
