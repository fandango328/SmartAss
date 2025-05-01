import os
import traceback
import webbrowser

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from requests_oauthlib import OAuth2Session

# Define the SCOPES needed for your app
SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/gmail.settings.basic",
    "https://www.googleapis.com/auth/gmail.settings.sharing",
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/contacts",
    "https://www.googleapis.com/auth/contacts.readonly",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/tasks"
]

def get_google_creds(debug=False, use_google=True):
    """
    Returns Google API credentials, ensuring all required scopes.
    If credentials are not valid or missing, runs the OAuth2 flow.
    """
    creds = None
    try:
        # Register chromium for OAuth, if available
        try:
            webbrowser.register('chromium', None, webbrowser.Chrome('/usr/bin/chromium'))
        except Exception as e:
            if debug:
                print(f"Warning: Could not register Chromium browser: {e}")

        if not use_google:
            if debug:
                print("USE_GOOGLE is False, skipping credential setup.")
            return None

        # Load token.json if it exists
        if os.path.exists("token.json"):
            try:
                creds = Credentials.from_authorized_user_file("token.json", SCOPES)
                if debug:
                    print("Loaded existing credentials from token.json")
            except Exception as e:
                print(f"Error loading credentials: {e}")
                if os.path.exists("token.json"):
                    os.remove("token.json")
                creds = None

        # If no valid creds, go through the OAuth flow
        if not creds or not creds.valid:
            try:
                if creds and creds.expired and creds.refresh_token:
                    if debug:
                        print("Refreshing expired credentials")
                    creds.refresh(Request())
                else:
                    if debug:
                        print("Initiating new OAuth2 flow")
                    flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes=SCOPES)
                    session = OAuth2Session(
                        client_id=flow.client_config['client_id'],
                        scope=SCOPES,
                        redirect_uri='http://localhost:8080/'
                    )
                    flow.oauth2session = session
                    auth_url, _ = flow.authorization_url(
                        access_type='offline',
                        include_granted_scopes='true',
                        prompt='consent'
                    )
                    print("Opening browser for authentication...")
                    creds = flow.run_local_server(
                        host='localhost',
                        port=8080,
                        access_type='offline',
                        prompt='consent',
                        authorization_prompt_message="Please complete authentication in the opened browser window",
                        success_message="Authentication completed successfully. You may close this window."
                    )

                    if debug:
                        print("\nDetailed Token Information:")
                        print(f"Valid: {creds.valid}")
                        print(f"Expired: {creds.expired}")
                        print(f"Has refresh_token attribute: {hasattr(creds, 'refresh_token')}")
                        if hasattr(creds, 'refresh_token'):
                            print(f"Refresh token value present: {bool(creds.refresh_token)}")
                            print(f"Token expiry: {creds.expiry}")

                if creds and creds.valid:
                    if not hasattr(creds, 'refresh_token') or not creds.refresh_token:
                        print("\nWARNING: No refresh token received!")
                        print("This might require re-authentication on next run.")
                        print("Try revoking access at https://myaccount.google.com/permissions")
                        print("Then delete token.json and run again.")
                    else:
                        with open("token.json", "w") as token:
                            token.write(creds.to_json())
                        if debug:
                            print("Credentials saved successfully")
                else:
                    print("Warning: Invalid credentials state")

            except Exception as e:
                print(f"Error during Google authentication: {e}")
                traceback.print_exc()
                if os.path.exists("token.json"):
                    os.remove("token.json")
                creds = None

    except Exception as e:
        print(f"Error setting up Google integration: {e}")
        traceback.print_exc()
        return None

    return creds

# Optional: Provide a default creds singleton for simple usage
creds = get_google_creds(debug=False, use_google=True)
