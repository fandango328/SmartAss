#!/usr/bin/env python3

# Standard Library Imports
import os
import json
import glob
import gc
import psutil
import random
import traceback
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from google_creds import creds
from googleapiclient.discovery import build
from typing import List, Dict, Any
# Third-Party Imports
import requests

# Local Imports
from secret import GOOGLE_MAPS_API_KEY
from config import (
    ACTIVE_PERSONA,
    DEBUG_CALENDAR,
    CHAT_LOG_DIR,
    CHAT_LOG_MAX_TOKENS,
    CHAT_LOG_RECOVERY_TOKENS,
    SYSTEM_STATE_COMMANDS
)

# Global Variables
token_manager = None
LAST_TASKS_RESULT = {
    "tasks": [],
    "timestamp": None
}


def get_current_time(format: str = "both") -> str:
    """Get the current time in the local timezone"""
    current = datetime.now()
    if format == "time":
        return current.strftime("%I:%M %p")
    elif format == "date":
        return current.strftime("%B %d, %Y")
    else:  # both
        return current.strftime("%I:%M %p on %B %d, %Y")

def get_location(format: str) -> str:
    print("DEBUG: Entering get_location function")

    try:
        print("DEBUG: Attempting to import requests")
        import requests
        print("DEBUG: Successfully imported requests")

        # Get WiFi access points
        print("DEBUG: Attempting to scan WiFi networks")
        import subprocess

        # For Linux/Raspberry Pi
        cmd = "iwlist wlan0 scan | grep -E 'Address|Signal|ESSID'"
        print(f"DEBUG: Running command: {cmd}")
        output = subprocess.check_output(cmd, shell=True).decode()
        print(f"DEBUG: Command output length: {len(output)}")

        # Parse WiFi data
        print("DEBUG: Parsing WiFi data")
        wifi_data = []
        current_ap = {}

        for line in output.split('\n'):
            if 'Address' in line:
                if current_ap:
                    wifi_data.append(current_ap)
                current_ap = {'macAddress': line.split('Address: ')[1].strip()}
            elif 'Signal' in line:
                try:
                    # Handle different signal strength formats
                    signal_part = line.split('=')[1].split(' ')[0]
                    if '/' in signal_part:
                        # Handle format like "70/70"
                        numerator, denominator = map(int, signal_part.split('/'))
                        # Convert to dBm (typical range -100 to 0)
                        signal_level = -100 + (numerator * 100) // denominator
                    else:
                        # Direct dBm value
                        signal_level = int(signal_part)
                    current_ap['signalStrength'] = signal_level
                except Exception as e:
                    print(f"DEBUG: Error parsing signal strength: {e}")
                    current_ap['signalStrength'] = -50  # Default middle value
            elif 'ESSID' in line:
                current_ap['ssid'] = line.split('ESSID:')[1].strip('"')

        if current_ap:
            wifi_data.append(current_ap)

        print(f"DEBUG: Found {len(wifi_data)} WiFi access points")

        if not wifi_data:
            return "No WiFi access points found"

        # Google Geolocation API request
        print("DEBUG: Preparing Google Geolocation API request")
        url = "https://www.googleapis.com/geolocation/v1/geolocate"
        params = {"key": GOOGLE_MAPS_API_KEY}  # Using key from secret.py
        data = {"wifiAccessPoints": wifi_data}

        try:
            print("DEBUG: Making POST request to Google Geolocation API")
            response = requests.post(url, params=params, json=data)
            print(f"DEBUG: Geolocation API response status: {response.status_code}")
            response.raise_for_status()  # Raise exception for bad status codes
            location = response.json()

            if 'error' in location:
                return f"Google API error: {location['error']['message']}"

            lat = location['location']['lat']
            lng = location['location']['lng']
            print(f"DEBUG: Retrieved coordinates: {lat}, {lng}")

            if format == "coordinates":
                return f"Current coordinates are: {lat}, {lng}"

            # Get address if needed
            if format in ["address", "both"]:
                print("DEBUG: Preparing Google Geocoding API request")
                geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json"
                params = {
                    "latlng": f"{lat},{lng}",
                    "key": GOOGLE_MAPS_API_KEY
                }

                print("DEBUG: Making GET request to Google Geocoding API")
                address_response = requests.get(geocode_url, params=params)
                print(f"DEBUG: Geocoding API response status: {address_response.status_code}")
                address_response.raise_for_status()
                address_data = address_response.json()

                if address_data['status'] == 'OK' and address_data['results']:
                    address = address_data['results'][0]['formatted_address']
                    print(f"DEBUG: Retrieved address: {address}")

                    if format == "address":
                        return f"Current location is: {address}"
                    else:  # both
                        return f"Current location is: {address}\nCoordinates: {lat}, {lng}"
                else:
                    return f"Coordinates found ({lat}, {lng}) but could not determine address"

        except requests.exceptions.RequestException as e:
            print(f"DEBUG: Request error: {e}")
            return f"Error communicating with Google API: {str(e)}"

    except subprocess.CalledProcessError as e:
        print(f"DEBUG: Error scanning WiFi networks: {e}")
        return f"Error scanning WiFi networks: {str(e)}"
    except Exception as e:
        import traceback
        print(f"DEBUG: Unexpected error in get_location: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return f"Unexpected error: {str(e)}"
        
def manage_tasks(action, title=None, notes=None, due_date=None, task_id=None,
                status=None, max_results=10, list_id=None):
    """
    Manage Google Tasks - create, update, list, complete, or delete tasks
    """
    global LAST_TASKS_RESULT

    try:
        # Build the Tasks API service
        service = build('tasks', 'v1', credentials=creds)

        # Get default task list if not specified
        if not list_id:
            lists = service.tasklists().list().execute()
            list_id = lists['items'][0]['id'] if 'items' in lists else None
            if not list_id:
                return "No task lists found. Please create a task list first."

        # HANDLE DIFFERENT ACTIONS

        # CREATE a new task
        if action == "create":
            if not title:
                return "Task title is required for creating a task."

            # Prepare the task data
            task_data = {
                'title': title,
            }

            if notes:
                task_data['notes'] = notes

            if due_date:
                # Convert natural language dates to ISO format
                parsed_date = parse_natural_date(due_date)
                if parsed_date:
                    # Google Tasks API expects RFC 3339 timestamp
                    task_data['due'] = parsed_date.isoformat() + 'Z'  # UTC time

            # Create the task
            result = service.tasks().insert(tasklist=list_id, body=task_data).execute()

            return f"Task '{title}' has been created successfully."

        # UPDATE an existing task
        elif action == "update":
            if not task_id:
                return "Task ID is required for updating a task."

            # Get the current task data
            task = service.tasks().get(tasklist=list_id, task=task_id).execute()

            # Update the fields if provided
            if title:
                task['title'] = title

            if notes:
                task['notes'] = notes

            if due_date:
                # Convert natural language dates to ISO format
                parsed_date = parse_natural_date(due_date)
                if parsed_date:
                    task['due'] = parsed_date.isoformat() + 'Z'  # UTC time

            if status:
                task['status'] = status

            # Update the task
            result = service.tasks().update(tasklist=list_id, task=task_id, body=task).execute()

            return f"Task '{task['title']}' has been updated successfully."

        # COMPLETE a task
        elif action == "complete":
            if not task_id:
                return "Task ID is required for completing a task."

            # Get the current task data
            task = service.tasks().get(tasklist=list_id, task=task_id).execute()

            # Mark as completed
            task['status'] = 'completed'

            # Update the task
            result = service.tasks().update(tasklist=list_id, task=task_id, body=task).execute()

            return f"Task '{task['title']}' has been marked as completed."

        # DELETE a task
        elif action == "delete":
            if not task_id:
                return "Task ID is required for deleting a task."

            # Get the task title first for confirmation
            task = service.tasks().get(tasklist=list_id, task=task_id).execute()
            task_title = task.get('title', 'Unnamed task')

            # Delete the task
            service.tasks().delete(tasklist=list_id, task=task_id).execute()

            return f"Task '{task_title}' has been deleted."

        # LIST tasks
        elif action == "list":
            # Get tasks
            tasks_result = service.tasks().list(
                tasklist=list_id,
                maxResults=max_results,
                showCompleted=True,
                showHidden=False
            ).execute()

            tasks = tasks_result.get('items', [])

            if not tasks:
                return "No tasks found in this list."

            # Store tasks for later reference
            processed_tasks = []
            upcoming_tasks = []
            completed_tasks = []

            for task in tasks:
                task_info = {
                    'id': task['id'],
                    'title': task.get('title', 'Unnamed task'),
                    'status': task.get('status', 'needsAction'),
                    'notes': task.get('notes', ''),
                }

                # Parse due date if available
                if 'due' in task:
                    due_date = datetime.fromisoformat(task['due'].replace('Z', '+00:00'))
                    task_info['due_date'] = due_date.strftime('%B %d, %Y')
                else:
                    task_info['due_date'] = None

                processed_tasks.append(task_info)

                # Separate upcoming and completed tasks
                if task_info['status'] == 'completed':
                    completed_tasks.append(task_info)
                else:
                    upcoming_tasks.append(task_info)

            # Store in global variable for future reference
            LAST_TASKS_RESULT = {
                "tasks": processed_tasks,
                "timestamp": datetime.now(),
                "list_id": list_id
            }

            # Format response for voice
            response = ""

            if upcoming_tasks:
                response += f"You have {len(upcoming_tasks)} upcoming tasks:\n\n"
                for i, task in enumerate(upcoming_tasks, 1):
                    due_str = f" (Due: {task['due_date']})" if task['due_date'] else ""
                    response += f"{i}. {task['title']}{due_str}\n"
                    if task['notes']:
                        # Add a brief preview of notes if they exist
                        notes_preview = task['notes'][:50] + "..." if len(task['notes']) > 50 else task['notes']
                        response += f"   Note: {notes_preview}\n"
                response += "\n"

            if completed_tasks:
                response += f"You also have {len(completed_tasks)} completed tasks.\n\n"

            response += "You can ask me to create, update, complete, or delete specific tasks."

            return response

        else:
            return f"Unknown action: {action}. Please specify 'create', 'update', 'list', 'complete', or 'delete'."

    except Exception as e:
        print(f"Error managing tasks: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to manage your tasks: {str(e)}"

def create_task_from_email(email_id, title=None, due_date=None, priority="medium"):
    """
    Create a task based on an email
    """
    try:
        # Get email details using email_manager
        msg = email_manager.service.users().messages().get(
            userId="me",
            id=email_id,
            format="metadata"
        ).execute()

        # Extract headers
        headers = {header["name"].lower(): header["value"]
                  for header in msg["payload"]["headers"]}

        # Get email subject and sender
        subject = headers.get("subject", "(No subject)")
        sender = headers.get("from", "").split("<")[0].strip()

        # Create task title if not provided
        if not title:
            title = f"Email: {subject}"

        # Create notes with email details
        notes = f"From: {sender}\nSubject: {subject}\nEmail ID: {email_id}\n\n"
        notes += f"Snippet: {msg.get('snippet', '')}\n\n"

        # Add priority to notes
        if priority:
            notes += f"Priority: {priority.upper()}\n"

        # Create the task
        tasks_service = build('tasks', 'v1', credentials=creds)

        # Get default task list
        lists = tasks_service.tasklists().list().execute()
        list_id = lists['items'][0]['id'] if 'items' in lists else None

        if not list_id:
            return "No task lists found. Please create a task list first."

        # Prepare the task data
        task_data = {
            'title': title,
            'notes': notes
        }

        if due_date:
            # Convert natural language dates to ISO format
            parsed_date = parse_natural_date(due_date)
            if parsed_date:
                task_data['due'] = parsed_date.isoformat() + 'Z'  # UTC time

        # Create the task
        result = tasks_service.tasks().insert(tasklist=list_id, body=task_data).execute()

        # Mark the email as read and add a label if possible
        try:
            email_manager.service.users().messages().modify(
                userId="me",
                id=email_id,
                body={"removeLabelIds": ["UNREAD"], "addLabelIds": ["STARRED"]}
            ).execute()
        except Exception as label_error:
            print(f"Warning: Could not modify email labels: {label_error}")

        return f"Task '{title}' has been created from the email. The email has been marked as read and starred."

    except Exception as e:
        print(f"Error creating task from email: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to create a task from the email: {str(e)}"

def create_task_for_event(event_id, task_type="both", days_before=1, days_after=1, custom_titles=None):
    """
    Create preparation or follow-up tasks for a calendar event
    """
    try:
        # Get event details
        calendar_service = build("calendar", "v3", credentials=creds)

        event = calendar_service.events().get(calendarId='primary', eventId=event_id).execute()

        event_title = event.get('summary', 'Unnamed event')

        # Parse event start time
        if 'dateTime' in event['start']:
            start_time = datetime.fromisoformat(event['start']['dateTime'].replace('Z', '+00:00'))
        else:
            # All-day event
            start_time = datetime.fromisoformat(event['start']['date'])

        # Get tasks service    #<----- this is line 884
        tasks_service = build('tasks', 'v1', credentials=creds)

        # Get default task list
        lists = tasks_service.tasklists().list().execute()
        list_id = lists['items'][0]['id'] if 'items' in lists else None

        if not list_id:
            return "No task lists found. Please create a task list first."

        tasks_created = []

        # Create preparation tasks
        if task_type in ["preparation", "both"]:
            prep_due_date = start_time - timedelta(days=days_before)

            if custom_titles and len(custom_titles) > 0:
                prep_title = custom_titles[0]
            else:
                prep_title = f"Prepare for: {event_title}"

            # Create notes with event details
            notes = f"Event: {event_title}\n"
            notes += f"Date: {start_time.strftime('%B %d, %Y at %I:%M %p')}\n"
            notes += f"Calendar Event ID: {event_id}\n\n"

            if 'description' in event and event['description']:
                notes += f"Event Description: {event['description'][:200]}...\n\n"

            if 'location' in event and event['location']:
                notes += f"Location: {event['location']}\n"

            if 'attendees' in event and event['attendees']:
                attendees = ", ".join([attendee.get('email', '') for attendee in event['attendees'][:5]])
                notes += f"Attendees: {attendees}"
                if len(event['attendees']) > 5:
                    notes += f" and {len(event['attendees']) - 5} more"

            # Prepare the task data
            task_data = {
                'title': prep_title,
                'notes': notes,
                'due': prep_due_date.isoformat() + 'Z'  # UTC time
            }

            # Create the task
            prep_task = tasks_service.tasks().insert(tasklist=list_id, body=task_data).execute()
            tasks_created.append(prep_task['title'])

        # Create follow-up tasks
        if task_type in ["follow_up", "both"]:
            followup_due_date = start_time + timedelta(days=days_after)

            if custom_titles and len(custom_titles) > 1:
                followup_title = custom_titles[1]
            else:
                followup_title = f"Follow up on: {event_title}"

            # Create notes with event details
            notes = f"Follow-up for event: {event_title}\n"
            notes += f"Original Date: {start_time.strftime('%B %d, %Y at %I:%M %p')}\n"
            notes += f"Calendar Event ID: {event_id}\n\n"

            if 'attendees' in event and event['attendees']:
                attendees = ", ".join([attendee.get('email', '') for attendee in event['attendees'][:5]])
                notes += f"Attendees: {attendees}"
                if len(event['attendees']) > 5:
                    notes += f" and {len(event['attendees']) - 5} more"

            # Prepare the task data
            task_data = {
                'title': followup_title,
                'notes': notes,
                'due': followup_due_date.isoformat() + 'Z'  # UTC time
            }

            # Create the task
            followup_task = tasks_service.tasks().insert(tasklist=list_id, body=task_data).execute()
            tasks_created.append(followup_task['title'])

        # Return success message
        if len(tasks_created) == 1:
            return f"Task '{tasks_created[0]}' has been created for the event."
        else:
            return f"Tasks have been created for the event: {', '.join(tasks_created)}."

    except Exception as e:
        print(f"Error creating tasks for event: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to create tasks for the event: {str(e)}"

def parse_natural_date(date_str):
    """
    Parse natural language date strings into datetime objects
    """
    try:
        # Try simple cases first
        now = datetime.now()

        if date_str.lower() == "today":
            return datetime.combine(now.date(), datetime.min.time())

        elif date_str.lower() == "tomorrow":
            return datetime.combine(now.date() + timedelta(days=1), datetime.min.time())

        elif date_str.lower() == "next week":
            # Next Monday
            days_ahead = 7 - now.weekday()
            return datetime.combine(now.date() + timedelta(days=days_ahead), datetime.min.time())

        # Try to parse as ISO format or other common formats
        try:
            if "T" in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                # Try common formats
                formats = [
                    "%Y-%m-%d",
                    "%m/%d/%Y",
                    "%B %d, %Y",
                    "%b %d, %Y",
                    "%d %B %Y",
                    "%d %b %Y"
                ]

                for fmt in formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
        except:
            pass

        # Fall back to more sophisticated parsing if needed
        # In a production environment, you might use libraries like dateparser
        # For this implementation, we'll just handle the most common cases

        return None

    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None

def manage_contacts(action, name=None, email=None, phone=None, company=None,
                   relationship=None, importance=None, query=None):
    """
    Manage contacts using Google People API with enhanced metadata
    """
    try:
        # Build the People API service
        service = build('people', 'v1', credentials=creds)

        # Load custom metadata
        metadata = load_contact_metadata()

        if action == "create":
            if not name or not email:
                return "Name and email are required to create a contact."

            # Create the contact in Google
            contact_body = {
                "names": [{"givenName": name}],
                "emailAddresses": [{"value": email}]
            }

            if phone:
                contact_body["phoneNumbers"] = [{"value": phone}]

            if company:
                contact_body["organizations"] = [{"name": company}]

            result = service.people().createContact(
                body=contact_body
            ).execute()

            # Store custom metadata
            if relationship or importance:
                metadata[email] = {
                    "relationship": relationship,
                    "importance": importance
                }
                save_contact_metadata(metadata)

            return f"Contact for {name} ({email}) has been created successfully."

        elif action == "search":
            if not query:
                return "Search query is required."

            # Search Google Contacts
            results = service.people().searchContacts(
                query=query,
                readMask="names,emailAddresses,phoneNumbers,organizations"
            ).execute()

            connections = results.get("results", [])

            if not connections:
                return f"No contacts found matching '{query}'."

            # Format results for voice response
            response = f"I found {len(connections)} contacts matching '{query}':\n\n"

            for i, person in enumerate(connections, 1):
                person_data = person.get("person", {})

                # Extract name
                names = person_data.get("names", [])
                name = names[0].get("displayName", "Unnamed") if names else "Unnamed"

                # Extract email
                emails = person_data.get("emailAddresses", [])
                email = emails[0].get("value", "No email") if emails else "No email"

                # Extract company
                orgs = person_data.get("organizations", [])
                company = orgs[0].get("name", "") if orgs else ""

                # Get custom metadata
                meta = metadata.get(email, {})
                importance = meta.get("importance", "")
                relationship = meta.get("relationship", "")

                # Format entry
                response += f"{i}. {name} - {email}\n"
                if company:
                    response += f"   Company: {company}\n"
                if relationship:
                    response += f"   Relationship: {relationship}\n"
                if importance:
                    response += f"   Importance: {importance}\n"
                response += "\n"

            return response

        elif action == "list":
            # List contacts from Google
            results = service.people().connections().list(
                resourceName='people/me',
                pageSize=100,
                personFields='names,emailAddresses,organizations'
            ).execute()

            connections = results.get('connections', [])

            if not connections:
                return "You don't have any contacts saved."

            # Group by importance if available
            important_contacts = []
            regular_contacts = []

            for person in connections:
                # Extract name
                names = person.get("names", [])
                name = names[0].get("displayName", "Unnamed") if names else "Unnamed"

                # Extract email
                emails = person.get("emailAddresses", [])
                email = emails[0].get("value", "") if emails else ""

                if not email:
                    continue

                # Get metadata
                meta = metadata.get(email, {})
                importance = meta.get("importance", "")

                contact_info = {
                    "name": name,
                    "email": email,
                    "importance": importance
                }

                if importance == "high":
                    important_contacts.append(contact_info)
                else:
                    regular_contacts.append(contact_info)

            # Format response
            response = ""

            if important_contacts:
                response += f"You have {len(important_contacts)} important contacts:\n\n"
                for contact in important_contacts:
                    response += f"- {contact['name']} ({contact['email']})\n"
                response += "\n"

            response += f"You have {len(regular_contacts)} other contacts."

            if len(connections) > 10:
                response += " You can ask me to search for specific contacts if needed."

            return response

        elif action == "get":
            if not email:
                return "Email address is required to get contact details."

            # Search for the contact
            results = service.people().searchContacts(
                query=email,
                readMask="names,emailAddresses,phoneNumbers,organizations,addresses"
            ).execute()

            connections = results.get("results", [])

            if not connections:
                return f"No contact found with email '{email}'."

            person_data = connections[0].get("person", {})

            # Extract details
            names = person_data.get("names", [])
            name = names[0].get("displayName", "Unnamed") if names else "Unnamed"

            phones = person_data.get("phoneNumbers", [])
            phone = phones[0].get("value", "No phone") if phones else "No phone"

            orgs = person_data.get("organizations", [])
            company = orgs[0].get("name", "No company") if orgs else "No company"
            title = orgs[0].get("title", "") if orgs else ""

            addresses = person_data.get("addresses", [])
            address = addresses[0].get("formattedValue", "No address") if addresses else "No address"

            # Get metadata
            meta = metadata.get(email, {})
            importance = meta.get("importance", "")
            relationship = meta.get("relationship", "")

            # Format response
            response = f"Contact details for {name}:\n\n"
            response += f"Email: {email}\n"
            response += f"Phone: {phone}\n"
            if company:
                response += f"Company: {company}\n"
            if title:
                response += f"Title: {title}\n"
            if relationship:
                response += f"Relationship: {relationship}\n"
            if importance:
                response += f"Importance: {importance}\n"
            if address != "No address":
                response += f"Address: {address}\n"

            return response

        elif action == "update":
            if not email:
                return "Email address is required to update a contact."

            # Search for the contact first
            results = service.people().searchContacts(
                query=email,
                readMask="names,emailAddresses,phoneNumbers,organizations,addresses"
            ).execute()

            connections = results.get("results", [])

            if not connections:
                return f"No contact found with email '{email}'."

            person_data = connections[0].get("person", {})
            resource_name = person_data.get("resourceName")

            # Prepare fields to update
            update_person_fields = []
            contact_body = {}

            if name:
                contact_body["names"] = [{"givenName": name}]
                update_person_fields.append("names")

            if phone:
                contact_body["phoneNumbers"] = [{"value": phone}]
                update_person_fields.append("phoneNumbers")

            if company:
                contact_body["organizations"] = [{"name": company}]
                update_person_fields.append("organizations")

            # Update Google contact if we have standard fields
            if update_person_fields:
                result = service.people().updateContact(
                    resourceName=resource_name,
                    updatePersonFields=','.join(update_person_fields),
                    body=contact_body
                ).execute()

            # Update metadata
            updated_meta = False
            if email not in metadata:
                metadata[email] = {}

            if relationship:
                metadata[email]["relationship"] = relationship
                updated_meta = True

            if importance:
                metadata[email]["importance"] = importance
                updated_meta = True

            if updated_meta:
                save_contact_metadata(metadata)

            return f"Contact information for {name or email} has been updated successfully."

        else:
            return f"Unknown action: {action}. Please specify 'get', 'create', 'update', 'list', or 'search'."

    except Exception as e:
        print(f"Error managing contacts: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to manage contacts: {str(e)}"

def load_contact_metadata():
    """Load custom contact metadata from JSON file"""
    try:
        metadata_file = "contacts_metadata.json"
        if not os.path.exists(metadata_file):
            # Create empty metadata file if it doesn't exist
            with open(metadata_file, "w") as f:
                json.dump({}, f)
            return {}

        with open(metadata_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading contact metadata: {e}")
        return {}

def save_contact_metadata(metadata):
    """Save custom contact metadata to JSON file"""
    try:
        with open("contacts_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving contact metadata: {e}")

def estimate_tokens(message):
    """
    Quickly estimate token count for chat log management without API calls.
    Used for log trimming and context loading operations.

    Args:
        message: Either a message dictionary or string content

    Returns:
        int: Estimated token count

    Note:
        - This is an estimation function for internal log management
        - Uses character-based approximation (4 chars ≈ 1 token)
        - For exact counts in API calls, use token_manager's count_message_tokens
    """
    try:
        # Handle dictionary messages (typical chat format)
        if isinstance(message, dict):
            content = message.get("content", "")

            # Handle tool messages with multiple content blocks
            if isinstance(content, list):
                # Sum up the length of each content block
                total_chars = sum(len(str(block)) for block in content)
                return total_chars // 4

            # Regular message content
            return len(str(content)) // 4

        # Handle direct string input
        if isinstance(message, str):
            return len(message) // 4

        # Handle any other type by converting to string
        return len(str(message)) // 4

    except Exception as e:
        print(f"Warning: Error in token estimation: {e}")
        return 0  # Safe fallback

# In function_definitions.py
# Make sure 'List' and 'Dict' are imported from 'typing' at the top of the file
# from typing import List, Dict, Any (if not already present)
# You might also need: from config import CHAT_LOG_MAX_TOKENS (if used as a default in the new function)

def trim_chat_log(log: List[Dict[str, Any]], 
                  token_manager_instance: Any,
                  max_tokens_limit: int) -> List[Dict[str, Any]]:
    """
    Trims the chat log to be under a specified token limit using the provided token_manager.
    Prioritizes keeping the most recent user/assistant pairs and handles unpaired messages.
    """
    
    # Debug: Print current conversation structure
    print(f"\n=== TRIM_CHAT_LOG DEBUG ===")
    print(f"Input log length: {len(log)}")
    print(f"Max tokens limit: {max_tokens_limit}")
    print("Conversation structure:")
    for i, msg in enumerate(log):
        role = msg.get('role', 'unknown')
        content_preview = str(msg.get('content', ''))[:50] + '...' if len(str(msg.get('content', ''))) > 50 else str(msg.get('content', ''))
        print(f"  {i}: {role} - '{content_preview}'")
    
    working_log = list(log)
    if len(working_log) < 1:
        print("Log too short, returning as-is")
        return working_log
    
    # Check if we end with an unpaired user message
    ends_with_unpaired_user = (len(working_log) % 2 == 1 and 
                              working_log[-1].get("role") == "user")
    
    trimmed_log_reversed = []
    current_tokens = 0
    
    # Handle the unpaired user message at the end if present
    if ends_with_unpaired_user:
        unpaired_user = working_log[-1]
        try:
            unpaired_tokens = token_manager_instance.count_message_tokens([unpaired_user])
            if unpaired_tokens <= max_tokens_limit:
                trimmed_log_reversed.append(unpaired_user)
                current_tokens += unpaired_tokens
                print(f"Keeping unpaired user message: {unpaired_tokens} tokens")
            else:
                print(f"Unpaired user message too large ({unpaired_tokens} tokens), skipping")
        except Exception as e:
            print(f"Error counting tokens for unpaired user message: {e}")
        
        # Remove the unpaired message from consideration for pairs
        working_log = working_log[:-1]
    
    # Now process pairs from the remaining log
    i = len(working_log) - 1
    pair_count = 0
    
    while i > 0:
        assistant_msg = working_log[i]
        user_msg = working_log[i-1]
        
        # Check for valid user-assistant pair
        if assistant_msg.get("role") == "assistant" and user_msg.get("role") == "user":
            try:
                pair_tokens = token_manager_instance.count_message_tokens([user_msg, assistant_msg])
                print(f"Pair {pair_count + 1}: {pair_tokens} tokens (user + assistant)")
                
                if current_tokens + pair_tokens <= max_tokens_limit:
                    trimmed_log_reversed.append(assistant_msg)
                    trimmed_log_reversed.append(user_msg)
                    current_tokens += pair_tokens
                    pair_count += 1
                    print(f"  ✓ Kept pair {pair_count} (total: {current_tokens} tokens)")
                else:
                    print(f"  ✗ Pair would exceed limit, stopping")
                    break
                    
            except Exception as e:
                print(f"Error counting tokens for pair: {e}")
                i -= 2
                continue
        else:
            print(f"Non-pair found at positions {i-1}-{i}: {user_msg.get('role')} -> {assistant_msg.get('role')}")
            # Skip this non-pair and continue
            
        i -= 2
    
    final_trimmed_log = list(reversed(trimmed_log_reversed))
    
    print(f"Final result: {len(final_trimmed_log)} messages, {current_tokens} tokens")
    print(f"Kept {pair_count} complete pairs" + (f" + 1 unpaired user message" if ends_with_unpaired_user and trimmed_log_reversed else ""))
    print("=== END TRIM_CHAT_LOG DEBUG ===\n")
    
    # If we ended up with nothing and had something originally, keep at least the last message
    if not final_trimmed_log and log:
        print("WARNING: Trimming resulted in empty log, keeping last message as fallback")
        return [log[-1]]
    
    return final_trimmed_log

def save_to_log_file(message: Dict[str, Any]) -> None:
    """
    Save a message to the daily chat log JSON file.

    This function takes a message dictionary with 'role' and 'content',
    adds a timestamp, and appends it to the daily chat log JSON file.
    Skips protocol-only messages (tool_use/tool_result blocks).

    Args:
        message (dict): The message to save, containing 'role' and 'content' keys
    """
    # Protocol discipline: skip tool_use/tool_result blocks
    content = message.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and first.get("type") in ["tool_use", "tool_result"]:
            print("DEBUG: Skipping save_to_log_file for protocol artifact (tool_use/tool_result)")
            return

    print(f"\n**** SAVE_TO_LOG_FILE CALLED: {message['role']} - '{str(message['content'])[:30]}...' ****\n")

    # Create the filename based on today's date (YYYY-MM-DD format)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(CHAT_LOG_DIR, f"chat_log_{today}.json")

    # Create the directory if it doesn't exist
    os.makedirs(CHAT_LOG_DIR, exist_ok=True)

    # Create the log entry by copying the message and adding a timestamp
    log_entry = {
        "role": message["role"],
        "content": message["content"],
        "timestamp": datetime.now().isoformat()
    }

    # Read existing logs from the file
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Corrupted log file {log_file}, starting fresh")
            # Create a backup of the corrupted file
            backup_file = f"{log_file}.corrupted_{int(time.time())}"
            try:
                os.rename(log_file, backup_file)
                print(f"Corrupted file backed up to: {backup_file}")
            except Exception as e:
                print(f"Failed to backup corrupted file: {e}")

    # Append the new log entry to the existing logs
    logs.append(log_entry)

    # Create a temporary file to write to (for atomic write)
    tmp_file = f"{log_file}.tmp"

    try:
        # Write the updated logs to the temporary file
        with open(tmp_file, "w") as f:
            json.dump(logs, f, indent=2)

        # Rename the temporary file to the actual log file (atomic operation)
        os.replace(tmp_file, log_file)
    except Exception as e:
        print(f"Error saving to log file: {e}")
        # Clean up the temporary file if there was an error
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception as cleanup_error:
                print(f"Failed to cleanup temp file: {cleanup_error}")
        raise

def get_chat_messages_for_api():
    """
    Retrieve clean message history for API calls

    Returns:
        list: Messages formatted for API use
    """
    messages = []
    log_dir = os.path.join(CHAT_LOG_DIR, "conversation")
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = f"{log_dir}/chat_log_{today}.json"

    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
                messages = [entry["api_message"] for entry in logs]
        except Exception as e:
            print(f"Error reading chat log: {e}")

    return messages

def get_day_schedule() -> str:
    """Get all events for today"""
    try:
        service = get_calendar_service()
        if not service:
            return "Failed to initialize calendar service"

        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        events_result = service.events().list(
            calendarId="primary",
            timeMin=start_of_day.isoformat(),
            timeMax=end_of_day.isoformat(),
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        events = events_result.get("items", [])

        if not events:
            return "No events scheduled for today."

        schedule = []
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
            schedule.append(f"- {start_time.strftime('%I:%M %p')}: {event['summary']}")

        return "Today's schedule:\n" + "\n".join(schedule)

    except Exception as e:
        if DEBUG_CALENDAR:
            print(f"Error getting day schedule: {e}")
        return f"Error retrieving schedule: {str(e)}"

def update_calendar_event(event_id, summary=None, start_time=None, end_time=None,
                         description=None, location=None, attendees=None):
    """
    Update an existing calendar event
    """
    try:
        service = build("calendar", "v3", credentials=creds)

        # Get the current event
        event = service.events().get(calendarId='primary', eventId=event_id).execute()

        # Update fields if provided
        if summary:
            event['summary'] = summary

        if description:
            event['description'] = description

        if location:
            event['location'] = location

        if start_time:
            if 'dateTime' in event['start']:
                event['start']['dateTime'] = start_time
            else:
                # Convert all-day event to timed event
                event['start'] = {'dateTime': start_time, 'timeZone': 'America/Los_Angeles'}

        if end_time:
            if 'dateTime' in event['end']:
                event['end']['dateTime'] = end_time
            else:
                # Convert all-day event to timed event
                event['end'] = {'dateTime': end_time, 'timeZone': 'America/Los_Angeles'}

        if attendees:
            event['attendees'] = [{'email': email} for email in attendees]

        # Update the event
        updated_event = service.events().update(
            calendarId='primary',
            eventId=event_id,
            body=event,
            sendUpdates='all'
        ).execute()

        # Format a nice response
        event_time = ""
        if 'dateTime' in updated_event['start']:
            start_dt = datetime.fromisoformat(updated_event['start']['dateTime'].replace('Z', '+00:00'))
            start_time_str = start_dt.strftime('%I:%M %p on %B %d, %Y')
            event_time = f"at {start_time_str}"
        else:
            start_date = datetime.fromisoformat(updated_event['start']['date'])
            event_time = f"on {start_date.strftime('%B %d, %Y')}"

        return f"Calendar event '{updated_event['summary']}' {event_time} has been updated successfully."

    except Exception as e:
        print(f"Error updating calendar event: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to update the calendar event: {str(e)}"

def cancel_calendar_event(event_id, notify_attendees=True, cancellation_message=None):
    """
    Cancel an existing calendar event
    """
    try:
        service = build("calendar", "v3", credentials=creds)

        # Get the event details first for a better response message
        event = service.events().get(calendarId='primary', eventId=event_id).execute()
        event_summary = event.get('summary', 'Unnamed event')

        # Add cancellation message if provided
        if cancellation_message and notify_attendees:
            # We can't directly add a cancellation message via the API
            # So we'll update the event description first
            original_description = event.get('description', '')
            event['description'] = f"CANCELLED: {cancellation_message}\n\n{original_description}"

            service.events().update(
                calendarId='primary',
                eventId=event_id,
                body=event,
                sendUpdates='all'
            ).execute()

        # Delete the event
        service.events().delete(
            calendarId='primary',
            eventId=event_id,
            sendUpdates='all' if notify_attendees else 'none'
        ).execute()

        notification_status = "Attendees have been notified" if notify_attendees else "Attendees were not notified"
        return f"Calendar event '{event_summary}' has been cancelled successfully. {notification_status}."

    except Exception as e:
        print(f"Error cancelling calendar event: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while trying to cancel the calendar event: {str(e)}"

def sanitize_messages_for_api(chat_log):
    """
    Sanitize chat log messages for API consumption.
    Ensures messages have required fields and proper structure.
    Filters out historical tool_use/tool_result blocks—only allows them if present as the latest message (i.e., mid tool-use cycle).

    Args:
        chat_log: List of conversation messages

    Returns:
        list: Sanitized messages ready for API consumption
    """
    if not chat_log:
        return []

    sanitized = []
    for idx, msg in enumerate(chat_log):
        if "role" not in msg or "content" not in msg:
            continue

        # Skip historical tool_use/tool_result blocks
        # Only allow tool_use/tool_result lists if it's the very last message (for in-progress tool-use cycle)
        is_last = idx == len(chat_log) - 1
        if isinstance(msg["content"], list) and msg["content"]:
            first = msg["content"][0]
            if isinstance(first, dict) and first.get("type") in ["tool_use", "tool_result"]:
                if not is_last:
                    continue
                # If last, include as-is for API
                sanitized.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
                continue

        # Otherwise, include regular assistant/user messages
        if isinstance(msg["content"], str):
            sanitized.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        elif isinstance(msg["content"], list):
            # Only non-protocol lists (should be rare, but handle gracefully)
            sanitized.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # Check if last message is from user (Anthropic expects last message to be user for new input)
    if sanitized and sanitized[-1]["role"] != "user":
        return sanitized[:-1]

    return sanitized

def find_calendar_event(description, time_range_days=7):
    """Helper function to find calendar events matching a description"""
    try:
        service = build("calendar", "v3", credentials=creds)

        # Set time range for search
        now = datetime.now(timezone.utc)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=time_range_days)).isoformat()

        # Get events in the time range
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])

        # Filter events by description
        description = description.lower()
        matching_events = []

        for event in events:
            # Check against summary, description and location
            event_summary = event.get('summary', '').lower()
            event_description = event.get('description', '').lower()
            event_location = event.get('location', '').lower()

            # Simple fuzzy matching
            if (description in event_summary or
                description in event_description or
                description in event_location):

                start = event['start'].get('dateTime', event['start'].get('date'))

                # Format the start time
                if 'T' in start:  # This is a datetime
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    start_str = start_dt.strftime('%I:%M %p on %B %d, %Y')
                else:  # This is a date
                    start_dt = datetime.fromisoformat(start)
                    start_str = start_dt.strftime('%B %d, %Y')

                matching_events.append({
                    'id': event['id'],
                    'summary': event.get('summary', 'Unnamed event'),
                    'start': start,
                    'start_formatted': start_str,
                    'location': event.get('location', 'No location'),
                    'attendees': [attendee.get('email') for attendee in event.get('attendees', [])]
                })

        return matching_events

    except Exception as e:
        print(f"Error finding calendar events: {e}")
        traceback.print_exc()
        return []

def create_calendar_event(summary: str, start_time: str, end_time: str,
                        description: str = "", attendees: list = None) -> str:
    """Create a calendar event and send invites"""
    print("DEBUG: Starting create_calendar_event")
    try:
        service = get_calendar_service()
        if not service:
            return "Failed to initialize calendar service"

        # Use your local timezone
        local_timezone = "America/Los_Angeles"  # Adjust this to your timezone

        event_body = {
            "summary": summary,
            "description": description,
            "start": {
                "dateTime": start_time,
                "timeZone": local_timezone
            },
            "end": {
                "dateTime": end_time,
                "timeZone": local_timezone
            }
        }

        if attendees:
            event_body["attendees"] = [{"email": email} for email in attendees]
            event_body["sendUpdates"] = "all"

        print(f"DEBUG: Attempting to create event with body: {event_body}")

        event = service.events().insert(
            calendarId="primary",
            body=event_body,
            sendUpdates="all"
        ).execute()

        return f"Event created successfully: {event.get('htmlLink')}"
    except Exception as e:
        print(f"DEBUG: Error in create_calendar_event: {e}")
        return f"Error creating event: {str(e)}"

def optimize_memory():
    """Optimize memory before loading large models"""
    gc.collect()
    process = psutil.Process(os.getpid())
    if hasattr(gc, 'freeze'):
        gc.freeze()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    print(f"Memory usage before model load:")
    print(f"RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"Memory percentage: {memory_percent:.1f}%")
    return memory_info
    
def load_recent_context(token_manager=None, token_limit=None):
    """
    Load recent conversation context from log files, filtering out system commands
    and skipping protocol-only tool_use/tool_result messages.

    Args:
        token_manager: TokenManager instance for counting tokens
        token_limit: Optional override for token limit (default: from config)

    Returns:
        list: Filtered chat messages
    """
    if token_limit is None:
        token_limit = CHAT_LOG_RECOVERY_TOKENS

    # Ensure chat logs directory exists
    log_dir = CHAT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    # Get list of log files
    try:
        files = sorted(glob.glob(f"{log_dir}/chat_log_*.json"), reverse=True)[:2]
    except Exception as e:
        print(f"Error scanning log directory: {e}")
        return []

    filtered_messages = []
    current_tokens = 0

    # Process from newest to oldest
    for file in files:
        try:
            if not os.path.exists(file):
                continue

            with open(file, "r") as f:
                logs = json.load(f)

            # Process messages from newest to oldest
            for log_entry in reversed(logs):
                # Handle both old and new format
                if isinstance(log_entry, dict):
                    if "role" in log_entry and "content" in log_entry:
                        message = log_entry
                    elif "api_message" in log_entry:
                        message = log_entry["api_message"]
                    else:
                        continue
                else:
                    continue

                content = message.get("content", "")
                role = message.get("role")

                # Protocol discipline: skip tool_use/tool_result blocks
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict) and first.get("type") in ["tool_use", "tool_result"]:
                        continue

                # Ensure message has required fields and non-empty content
                if isinstance(content, str):
                    content_value = content.strip()
                else:
                    content_value = content

                if not content_value or not role:
                    continue

                formatted_message = {
                    "role": role,
                    "content": content_value
                }

                # Skip system commands
                content_lower = content_value.lower() if isinstance(content_value, str) else ""
                is_system_command = False

                for command_type, actions in SYSTEM_STATE_COMMANDS.items():
                    for action, commands in actions.items():
                        if any(cmd.lower() in content_lower for cmd in commands):
                            is_system_command = True
                            break
                    if is_system_command:
                        break

                if (("voice" in content_lower or "boys" in content_lower) and
                    any(word in content_lower for word in ["calibrat", "detect"])):
                    is_system_command = True

                if not is_system_command:
                    if token_manager:
                        try:
                            msg_tokens = token_manager.count_message_tokens([formatted_message])
                            if current_tokens + msg_tokens > token_limit:
                                break
                            current_tokens += msg_tokens
                        except Exception as e:
                            print(f"Error counting tokens: {e}")
                            continue

                    filtered_messages.insert(0, formatted_message)

        except Exception as e:
            print(f"Error loading from {file}: {e}")
            continue

    print(f"Loaded {len(filtered_messages)} messages from previous conversation")
    return filtered_messages
        
def get_audio_path(category, context=None, persona=None):
    """
    Centralized audio path resolution with fallbacks
    
    Args:
        category: Main audio category (wake, tool, timeout, etc.)
        context: Optional subcategory (loaded, use, etc.)
        persona: Override active persona (useful for testing)
    
    Returns:
        Path object to the appropriate audio directory
    """
    # Use specified persona or import active one from config
    if persona:
        active_persona = persona.lower()
    else:
        # Direct import to get current value
        from config import ACTIVE_PERSONA
        active_persona = ACTIVE_PERSONA.lower()

    # Base directory containing all sound categories
    base_sound_dir = Path(f"/home/user/LAURA/sounds/{active_persona}")
    
    # Special case mappings
    if category == "wake" and context in ["Laura.pmdl", "Wake_up_Laura.pmdl", "GD_Laura.pmdl"]:
        # Map wake word models to context folders
        context_map = {
            "Laura.pmdl": "standard",
            "Wake_up_Laura.pmdl": "sleepy",
            "GD_Laura.pmdl": "frustrated"
        }
        folder = context_map.get(context, "standard")
        primary_path = base_sound_dir / "wake_sentences" / folder
    
    elif category == "file" and context:
        primary_path = base_sound_dir / "file_sentences" / context
    
    elif category == "tool" and context:
        if context == "use":
            primary_path = base_sound_dir / "tool_sentences" / "use"
        elif context in ["enabled", "disabled"]:
            primary_path = base_sound_dir / "tool_sentences" / "status" / context
        else:
            primary_path = base_sound_dir / "tool_sentences" / context
    
    else:
        # Default to main category folder
        primary_path = base_sound_dir / f"{category}_sentences"
        if context and (base_sound_dir / f"{category}_sentences" / context).exists():
            primary_path = base_sound_dir / f"{category}_sentences" / context
    
    # Check if path exists with audio files
    if primary_path.exists() and (list(primary_path.glob('*.mp3')) or list(primary_path.glob('*.wav'))):
        return primary_path
    
    # Try fallback to parent directory for empty subfolders
    if context and f"{category}_sentences" in str(primary_path):
        parent_path = base_sound_dir / f"{category}_sentences"
        if parent_path.exists() and (list(parent_path.glob('*.mp3')) or list(parent_path.glob('*.wav'))):
            return parent_path
    
    # Ultimate fallback - try Laura's resources if different persona
    if active_persona != "laura":
        laura_path = Path(f"/home/user/LAURA/sounds/laura")
        
        # Reconstruct the same path structure but with Laura
        if category == "wake" and context in ["Laura.pmdl", "Wake_up_Laura.pmdl", "GD_Laura.pmdl"]:
            context_map = {
                "Laura.pmdl": "standard", 
                "Wake_up_Laura.pmdl": "sleepy",
                "GD_Laura.pmdl": "frustrated"
            }
            folder = context_map.get(context, "standard")
            fallback_path = laura_path / "wake_sentences" / folder
        elif category == "file" and context:
            fallback_path = laura_path / "file_sentences" / context
        elif category == "tool" and context:
            if context == "use":
                fallback_path = laura_path / "tool_sentences" / "use"
            elif context in ["enabled", "disabled"]:
                fallback_path = laura_path / "tool_sentences" / "status" / context
            else:
                fallback_path = laura_path / "tool_sentences" / context
        else:
            fallback_path = laura_path / f"{category}_sentences"
            if context:
                context_path = laura_path / f"{category}_sentences" / context
                if context_path.exists():
                    fallback_path = context_path
        
        # Check if Laura's path exists with audio files
        if fallback_path.exists() and (list(fallback_path.glob('*.mp3')) or list(fallback_path.glob('*.wav'))):
            return fallback_path
    
    # Return original path even if empty (will be handled by caller)
    return primary_path

def get_calendar_service():
    """Helper function to build and return calendar service for Google Calendar API."""
    try:
        global creds  # Ensures we refer to the creds object in the main script if imported
        if creds is None:
            print("ERROR: No credentials available for Google Calendar API.")
            return None
        if not hasattr(creds, "valid") or not creds.valid:
            print("ERROR: Credentials are invalid or expired for Google Calendar API.")
            return None
        if DEBUG_CALENDAR:
            print("DEBUG: Attempting to build calendar service")
        service = build("calendar", "v3", credentials=creds)
        if DEBUG_CALENDAR:
            print("DEBUG: Calendar service built successfully")
        return service
    except NameError as ne:
        print("ERROR: 'creds' is not defined in get_calendar_service(). Make sure it is imported or defined globally.")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"ERROR: Exception occurred while building Google Calendar service: {e}")
        traceback.print_exc()
        return None
