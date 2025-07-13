import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Core dependencies (install with: pip install llama-stack-client google-api-python-client slack-sdk)
from llama_stack_client import LlamaStackClient
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmailSummary:
    sender: str
    subject: str
    summary: str
    priority: str
    action_required: bool
    meeting_info: Optional[Dict] = None

@dataclass
class MeetingNote:
    meeting_id: str
    title: str
    date: datetime
    participants: List[str]
    summary: str
    action_items: List[str]
    key_decisions: List[str]

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, llama_client: LlamaStackClient, name: str):
        self.llama_client = llama_client
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def execute(self, *args, **kwargs):
        pass
    
    async def get_llm_response(self, prompt: str, system_prompt: str = "") -> str:
        """Get response from Llama model"""
        try:
            response = await self.llama_client.inference.chat_completion(
                model_id="meta-llama/Llama-3.2-3B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            return response.completion_message.content
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}")
            return ""

class EmailAgent(BaseAgent):
    """Agent responsible for email processing and summarization"""
    
    def __init__(self, llama_client: LlamaStackClient, gmail_service):
        super().__init__(llama_client, "EmailAgent")
        self.gmail_service = gmail_service
    
    async def execute(self, max_emails: int = 10) -> List[EmailSummary]:
        """Process and summarize recent emails"""
        try:
            # Get recent emails
            results = self.gmail_service.users().messages().list(
                userId='me', 
                q='is:unread',
                maxResults=max_emails
            ).execute()
            
            messages = results.get('messages', [])
            summaries = []
            
            for message in messages:
                email_summary = await self._process_email(message['id'])
                if email_summary:
                    summaries.append(email_summary)
            
            return summaries
            
        except Exception as e:
            self.logger.error(f"Email processing failed: {e}")
            return []
    
    async def _process_email(self, message_id: str) -> Optional[EmailSummary]:
        """Process individual email"""
        try:
            message = self.gmail_service.users().messages().get(
                userId='me', 
                id=message_id
            ).execute()
            
            headers = message['payload'].get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
            
            # Extract email body
            body = self._extract_email_body(message['payload'])
            
            # Summarize with LLM
            summary_prompt = f"""
            Subject: {subject}
            From: {sender}
            Content: {body[:2000]}  # Truncate for token limits
            
            Please provide a concise summary and analysis.
            """
            
            system_prompt = """
            You are an email analysis assistant. For each email, provide:
            1. A brief summary (2-3 sentences)
            2. Priority level (High/Medium/Low)
            3. Whether action is required (Yes/No)
            4. If it mentions meetings, extract meeting details
            
            Format as JSON with keys: summary, priority, action_required, meeting_info
            """
            
            response = await self.get_llm_response(summary_prompt, system_prompt)
            
            try:
                analysis = json.loads(response)
                return EmailSummary(
                    sender=sender,
                    subject=subject,
                    summary=analysis.get('summary', ''),
                    priority=analysis.get('priority', 'Low'),
                    action_required=analysis.get('action_required', False),
                    meeting_info=analysis.get('meeting_info')
                )
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return EmailSummary(
                    sender=sender,
                    subject=subject,
                    summary=response[:200],
                    priority='Medium',
                    action_required=True
                )
                
        except Exception as e:
            self.logger.error(f"Email processing failed for {message_id}: {e}")
            return None
    
    def _extract_email_body(self, payload: Dict) -> str:
        """Extract text from email payload"""
        body = ""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body = self._decode_base64(data)
                    break
        elif payload['mimeType'] == 'text/plain':
            data = payload['body']['data']
            body = self._decode_base64(data)
        
        return body
    
    def _decode_base64(self, data: str) -> str:
        """Decode base64 email content"""
        import base64
        return base64.urlsafe_b64decode(data).decode('utf-8')

class CalendarAgent(BaseAgent):
    """Agent for calendar management and meeting reminders"""
    
    def __init__(self, llama_client: LlamaStackClient, calendar_service):
        super().__init__(llama_client, "CalendarAgent")
        self.calendar_service = calendar_service
    
    async def execute(self, hours_ahead: int = 24) -> List[Dict]:
        """Get upcoming meetings and set reminders"""
        try:
            now = datetime.utcnow()
            time_max = now + timedelta(hours=hours_ahead)
            
            events_result = self.calendar_service.events().list(
                calendarId='primary',
                timeMin=now.isoformat() + 'Z',
                timeMax=time_max.isoformat() + 'Z',
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            upcoming_meetings = []
            
            for event in events:
                meeting_info = await self._analyze_meeting(event)
                if meeting_info:
                    upcoming_meetings.append(meeting_info)
            
            return upcoming_meetings
            
        except Exception as e:
            self.logger.error(f"Calendar processing failed: {e}")
            return []
    
    async def _analyze_meeting(self, event: Dict) -> Optional[Dict]:
        """Analyze meeting and determine importance"""
        try:
            title = event.get('summary', 'No title')
            start_time = event['start'].get('dateTime', event['start'].get('date'))
            attendees = event.get('attendees', [])
            description = event.get('description', '')
            
            # Analyze meeting importance with LLM
            analysis_prompt = f"""
            Meeting: {title}
            Description: {description}
            Attendees: {len(attendees)} people
            
            Analyze this meeting's importance and suggest reminder timing.
            """
            
            system_prompt = """
            Analyze meeting importance and provide JSON response with:
            - importance: High/Medium/Low
            - reminder_minutes: [15, 60, 1440] for different reminder times
            - preparation_needed: boolean
            - meeting_type: one-on-one/team/presentation/other
            """
            
            analysis = await self.get_llm_response(analysis_prompt, system_prompt)
            
            return {
                'id': event['id'],
                'title': title,
                'start_time': start_time,
                'attendees': [a.get('email', '') for a in attendees],
                'description': description,
                'analysis': analysis,
                'meet_link': self._extract_meet_link(event)
            }
            
        except Exception as e:
            self.logger.error(f"Meeting analysis failed: {e}")
            return None
    
    def _extract_meet_link(self, event: Dict) -> Optional[str]:
        """Extract Google Meet link from event"""
        conference_data = event.get('conferenceData', {})
        entry_points = conference_data.get('entryPoints', [])
        
        for entry in entry_points:
            if entry.get('entryPointType') == 'video':
                return entry.get('uri')
        
        return None

class MeetingNotesAgent(BaseAgent):
    """Agent for taking and summarizing meeting notes"""
    
    def __init__(self, llama_client: LlamaStackClient, docs_service):
        super().__init__(llama_client, "MeetingNotesAgent")
        self.docs_service = docs_service
    
    async def execute(self, meeting_transcript: str, meeting_info: Dict) -> MeetingNote:
        """Process meeting transcript and create notes"""
        try:
            # Analyze transcript with LLM
            analysis_prompt = f"""
            Meeting: {meeting_info.get('title', 'Unknown Meeting')}
            Transcript: {meeting_transcript}
            
            Please analyze this meeting and provide structured notes.
            """
            
            system_prompt = """
            Create structured meeting notes in JSON format with:
            - summary: Brief overview of the meeting
            - key_decisions: List of decisions made
            - action_items: List of action items with owners if mentioned
            - important_topics: Main topics discussed
            - next_steps: What happens next
            """
            
            analysis = await self.get_llm_response(analysis_prompt, system_prompt)
            
            try:
                notes_data = json.loads(analysis)
            except json.JSONDecodeError:
                # Fallback structure
                notes_data = {
                    'summary': analysis[:500],
                    'key_decisions': [],
                    'action_items': [],
                    'important_topics': [],
                    'next_steps': []
                }
            
            # Create meeting note object
            meeting_note = MeetingNote(
                meeting_id=meeting_info.get('id', ''),
                title=meeting_info.get('title', ''),
                date=datetime.now(),
                participants=meeting_info.get('attendees', []),
                summary=notes_data.get('summary', ''),
                action_items=notes_data.get('action_items', []),
                key_decisions=notes_data.get('key_decisions', [])
            )
            
            # Save to Google Docs
            await self._save_to_docs(meeting_note, notes_data)
            
            return meeting_note
            
        except Exception as e:
            self.logger.error(f"Meeting notes processing failed: {e}")
            return None
    
    async def _save_to_docs(self, meeting_note: MeetingNote, notes_data: Dict):
        """Save meeting notes to Google Docs"""
        try:
            # Create document
            doc_title = f"Meeting Notes - {meeting_note.title} - {meeting_note.date.strftime('%Y-%m-%d')}"
            
            document = {
                'title': doc_title
            }
            
            doc = self.docs_service.documents().create(body=document).execute()
            doc_id = doc.get('documentId')
            
            # Prepare content
            content = f"""
MEETING NOTES

Meeting: {meeting_note.title}
Date: {meeting_note.date.strftime('%Y-%m-%d %H:%M')}
Participants: {', '.join(meeting_note.participants)}

SUMMARY
{meeting_note.summary}

KEY DECISIONS
{chr(10).join(f"• {decision}" for decision in meeting_note.key_decisions)}

ACTION ITEMS
{chr(10).join(f"• {item}" for item in meeting_note.action_items)}

IMPORTANT TOPICS
{chr(10).join(f"• {topic}" for topic in notes_data.get('important_topics', []))}

NEXT STEPS
{chr(10).join(f"• {step}" for step in notes_data.get('next_steps', []))}
            """
            
            # Insert content
            requests = [
                {
                    'insertText': {
                        'location': {'index': 1},
                        'text': content
                    }
                }
            ]
            
            self.docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={'requests': requests}
            ).execute()
            
            self.logger.info(f"Meeting notes saved to Google Docs: {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save to Google Docs: {e}")

class NotificationAgent(BaseAgent):
    """Agent for sending notifications via Slack"""
    
    def __init__(self, llama_client: LlamaStackClient, slack_client: WebClient):
        super().__init__(llama_client, "NotificationAgent")
        self.slack_client = slack_client
    
    async def execute(self, message: str, channel: str = None, urgent: bool = False):
        """Send notification to Slack"""
        try:
            if urgent:
                message = f"🚨 URGENT: {message}"
            
            response = self.slack_client.chat_postMessage(
                channel=channel or '#general',
                text=message,
                username='AI Assistant'
            )
            
            self.logger.info(f"Notification sent to Slack: {response['ts']}")
            
        except SlackApiError as e:
            self.logger.error(f"Slack notification failed: {e}")

class AgenticWorkSystem:
    """Main orchestrator for the agentic work system"""
    
    def __init__(self):
        self.llama_client = None
        self.agents = {}
        self.google_services = {}
        self.slack_client = None
        
    async def initialize(self, config: Dict):
        """Initialize all services and agents"""
        try:
            # Initialize Llama Stack client
            self.llama_client = LlamaStackClient(
                base_url=config.get('llama_stack_url', 'http://localhost:5000')
            )
            
            # Initialize Google services
            await self._setup_google_services(config['google_credentials'])
            
            # Initialize Slack client
            self.slack_client = WebClient(token=config['slack_token'])
            
            # Initialize agents
            self.agents = {
                'email': EmailAgent(self.llama_client, self.google_services['gmail']),
                'calendar': CalendarAgent(self.llama_client, self.google_services['calendar']),
                'notes': MeetingNotesAgent(self.llama_client, self.google_services['docs']),
                'notification': NotificationAgent(self.llama_client, self.slack_client)
            }
            
            logger.info("Agentic work system initialized successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    async def _setup_google_services(self, credentials_path: str):
        """Setup Google API services"""
        SCOPES = [
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/calendar',
            'https://www.googleapis.com/auth/documents'
        ]
        
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        self.google_services = {
            'gmail': build('gmail', 'v1', credentials=creds),
            'calendar': build('calendar', 'v3', credentials=creds),
            'docs': build('docs', 'v1', credentials=creds)
        }
    
    async def run_daily_workflow(self):
        """Execute the complete daily workflow"""
        try:
            logger.info("Starting daily workflow...")
            
            # 1. Process emails
            email_summaries = await self.agents['email'].execute()
            
            # 2. Check calendar for upcoming meetings
            upcoming_meetings = await self.agents['calendar'].execute()
            
            # 3. Send summary notification
            await self._send_daily_summary(email_summaries, upcoming_meetings)
            
            # 4. Schedule meeting reminders
            await self._schedule_meeting_reminders(upcoming_meetings)
            
            logger.info("Daily workflow completed successfully")
            
        except Exception as e:
            logger.error(f"Daily workflow failed: {e}")
            await self.agents['notification'].execute(
                f"Daily workflow failed: {str(e)}", 
                urgent=True
            )
    
    async def _send_daily_summary(self, emails: List[EmailSummary], meetings: List[Dict]):
        """Send daily summary to Slack"""
        summary_parts = ["📋 Daily Summary\n"]
        
        if emails:
            summary_parts.append(f"📧 Emails processed: {len(emails)}")
            high_priority = [e for e in emails if e.priority == 'High']
            if high_priority:
                summary_parts.append(f"⚠️ High priority emails: {len(high_priority)}")
        
        if meetings:
            summary_parts.append(f"📅 Upcoming meetings: {len(meetings)}")
            for meeting in meetings[:3]:  # Show first 3
                summary_parts.append(f"• {meeting['title']} - {meeting['start_time']}")
        
        summary_message = "\n".join(summary_parts)
        await self.agents['notification'].execute(summary_message)
    
    async def _schedule_meeting_reminders(self, meetings: List[Dict]):
        """Schedule reminders for upcoming meetings"""
        for meeting in meetings:
            try:
                start_time = datetime.fromisoformat(meeting['start_time'].replace('Z', '+00:00'))
                now = datetime.now().replace(tzinfo=start_time.tzinfo)
                
                # Send reminder if meeting is within 30 minutes
                if timedelta(minutes=15) <= (start_time - now) <= timedelta(minutes=30):
                    reminder_msg = f"🔔 Reminder: '{meeting['title']}' starts in {int((start_time - now).total_seconds() // 60)} minutes"
                    if meeting.get('meet_link'):
                        reminder_msg += f"\nJoin: {meeting['meet_link']}"
                    
                    await self.agents['notification'].execute(reminder_msg, urgent=True)
                    
            except Exception as e:
                logger.error(f"Failed to schedule reminder for {meeting.get('title', 'Unknown')}: {e}")
    
    async def process_meeting_transcript(self, transcript: str, meeting_info: Dict):
        """Process meeting transcript and create notes"""
        try:
            meeting_note = await self.agents['notes'].execute(transcript, meeting_info)
            
            if meeting_note:
                # Send notification about completed notes
                await self.agents['notification'].execute(
                    f"📝 Meeting notes completed for '{meeting_note.title}' and saved to Google Docs"
                )
                
                return meeting_note
            
        except Exception as e:
            logger.error(f"Meeting transcript processing failed: {e}")
            return None

# Example usage and configuration
async def main():
    """Main function to demonstrate the system"""
    
    # Configuration
    config = {
        'llama_stack_url': 'http://localhost:5000',  # Update with your Llama Stack URL
        'google_credentials': 'path/to/google_credentials.json',
        'slack_token': 'your-slack-bot-token'
    }
    
    # Initialize system
    system = AgenticWorkSystem()
    await system.initialize(config)
    
    # Run daily workflow
    await system.run_daily_workflow()
    
    # Example: Process a meeting transcript
    sample_transcript = """
    Meeting started at 2:00 PM
    John: Let's review the project timeline
    Sarah: We need to finish the MVP by next Friday
    Mike: I'll handle the backend integration
    John: Great, let's set up a follow-up meeting for Thursday
    """
    
    sample_meeting_info = {
        'id': 'meeting123',
        'title': 'Project Review',
        'attendees': ['john@company.com', 'sarah@company.com', 'mike@company.com']
    }
    
    meeting_note = await system.process_meeting_transcript(sample_transcript, sample_meeting_info)
    
    if meeting_note:
        print(f"Meeting notes created: {meeting_note.summary}")

if __name__ == "__main__":
    import os
    asyncio.run(main())
