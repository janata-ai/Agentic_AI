# AI Agentic system for daily work automation! 

## Key Features

1. Multi-Agent Architecture: Each agent has a specific responsibility:

**EmailAgent**: Processes and summarizes emails
**CalendarAgent**: Manages calendar events and reminders
**MeetingNotesAgent**: Takes meeting notes and saves to Google Docs
**NotificationAgent**: Sends alerts via Slack


2. Core Functionality:

**Email Processing**: Reads unread emails, summarizes content, determines priority
**Calendar Management**: Checks upcoming meetings, analyzes importance
**Meeting Notes**: Processes transcripts, extracts action items and decisions
**Smart Notification**s: Sends reminders and summaries to Slack


3. Integration Points:

Google Gmail API for email processing
Google Calendar API for meeting management
Google Docs API for note storage
Slack API for notifications
Llama Stack for AI processing

4. Setup Instructions

A. Install Dependencies:

bashpip install llama-stack-client google-api-python-client slack-sdk google-auth-oauthlib

B. Configure Google APIs:

Enable Gmail, Calendar, and Docs APIs in Google Cloud Console
Download credentials JSON file
Update the path in the config


C. Set up Slack Bot:

Create a Slack app and bot
Get the bot token
Add to your workspace


D. Llama Stack Setup:

Install and run Llama Stack locally
Update the URL in config


E. Usage Examples
The system automatically:

Processes your unread emails every morning
Sends daily summaries to Slack
Reminds you of upcoming meetings
Takes notes during meetings (when fed transcripts)
Saves structured notes to Google Docs

The system is modular and extensible - you can easily add new agents for different tasks or modify existing ones for your specific workflow needs.
