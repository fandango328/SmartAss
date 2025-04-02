#!/usr/bin/env python3

"""
Tool Contexts Definition Module for LAURA Voice Assistant
Last Updated: 2025-04-02 17:06:57
Author: fandango328
"""

from typing import Dict, List, Tuple, Optional, Union

# Calendar Interaction Contexts
CALENDAR_CONTEXTS = {
    "verbs": [
        # Core scheduling actions
        "schedule", "book", "plan", "create", "add",
        "move", "reschedule", "shift", "change", "switch",
        "update", "modify", "edit", "revise", "adjust",
        "cancel", "delete", "remove", "clear", "drop",
        
        # Query actions
        "check", "show", "see", "find", "look",
        "tell", "get", "give", "display", "list",
        
        # Time management
        "fit", "squeeze", "block", "set", "put",
        "push", "pull", "bump", "postpone", "delay",
        "advance", "bring", "move up", "push back",
        
        # Coordination
        "sync", "coordinate", "arrange", "organize",
        "align", "join", "attend", "participate",
        
        # Status changes
        "confirm", "decline", "accept", "reject",
        "tentative", "maybe", "pending"
    ],
    "contexts": [
        # Event types
        "meeting", "call", "appointment", "event",
        "conference", "lunch", "dinner", "session",
        "interview", "review", "sync", "1:1", "one-on-one",
        "team", "standup", "check-in", "huddle", "catch up",
        "presentation", "workshop", "training", "seminar",
        "break", "holiday", "vacation", "time off", "pto",
        
        # Time contexts
        "schedule", "calendar", "availability", "free time",
        "slot", "time", "date", "week", "month", "day",
        "morning", "afternoon", "evening", "tonight",
        "today", "tomorrow", "yesterday", "next week",
        "weekend", "weekday", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday",
        
        # Status contexts
        "busy", "free", "available", "conflict",
        "overlap", "double-booked", "recurring", "repeat",
        "series", "weekly", "monthly", "daily", "bi-weekly",
        
        # Location contexts
        "room", "office", "conference room", "virtual",
        "online", "in-person", "remote", "zoom", "teams",
        "meet", "hangout", "location", "place"
    ],
    "tools": [
        "create_calendar_event",
        "update_calendar_event",
        "calendar_query",
        "cancel_calendar_event"
    ]
}

# Email Management Contexts
EMAIL_CONTEXTS = {
    "verbs": [
        # Composition actions
        "send", "write", "draft", "compose", "prepare",
        "reply", "respond", "forward", "cc", "bcc",
        "attach", "include", "add",
        
        # Management actions
        "check", "read", "show", "see", "find",
        "search", "look", "get", "fetch", "pull up",
        "archive", "file", "move", "delete", "trash",
        "flag", "star", "mark", "label", "categorize",
        
        # Status actions
        "unread", "read", "important", "urgent",
        "follow up", "remind", "snooze", "mute",
        
        # Organization actions
        "organize", "sort", "filter", "clean", "clear",
        "folder", "label", "categorize", "group",
        
        # Common email phrases
        "follow up", "circle back", "touch base",
        "reach out", "ping", "drop a line", "shoot",
        "send over", "pass along", "forward"
    ],
    "contexts": [
        # Message types
        "email", "message", "mail", "reply", "response",
        "thread", "chain", "conversation", "discussion",
        "newsletter", "notification", "alert", "update",
        
        # Location contexts
        "inbox", "folder", "archive", "trash",
        "sent", "drafts", "spam", "junk", "box",
        "primary", "social", "promotions", "updates",
        
        # Status contexts
        "unread", "new", "recent", "old", "earlier",
        "important", "urgent", "priority", "critical",
        "starred", "flagged", "labeled", "marked",
        
        # Content contexts
        "attachment", "file", "document", "image",
        "picture", "pdf", "doc", "spreadsheet",
        "signature", "subject", "cc", "bcc", "recipient",
        
        # Time contexts
        "today", "yesterday", "this week", "last week",
        "month", "received", "sent", "dated", "before",
        "after", "between", "recent", "old", "latest"
    ],
    "tools": [
        "draft_email",
        "read_emails",
        "email_action"
    ]
}

# Task Management Contexts
TASK_CONTEXTS = {
    "verbs": [
        # Creation actions
        "create", "add", "make", "start", "set up",
        "new", "begin", "establish", "initiate",
        
        # Management actions
        "track", "list", "show", "check", "mark",
        "complete", "finish", "done", "update", "edit",
        "modify", "change", "revise", "adjust",
        
        # Organization actions
        "organize", "prioritize", "sort", "group",
        "categorize", "tag", "label", "classify",
        
        # Status actions
        "pending", "progress", "assign", "delegate",
        "transfer", "move", "shift", "escalate",
        
        # Follow-up actions
        "follow up", "chase", "remind", "ping",
        "check on", "status", "update"
    ],
    "contexts": [
        # Item types
        "task", "todo", "item", "action item", "action",
        "reminder", "checklist", "to-do", "to do",
        "assignment", "work item", "ticket", "issue",
        "project", "initiative", "goal", "objective",
        
        # Status contexts
        "pending", "in progress", "done", "completed",
        "overdue", "due", "upcoming", "scheduled",
        "blocked", "waiting", "hold", "delayed",
        "priority", "urgent", "important", "critical",
        
        # Time contexts
        "today", "tomorrow", "this week", "next week",
        "deadline", "due date", "by", "until",
        "morning", "afternoon", "evening", "tonight",
        
        # Organization contexts
        "list", "board", "project", "category",
        "priority", "tag", "label", "group",
        "personal", "work", "home", "errands"
    ],
    "tools": [
        "manage_tasks",
        "create_task_from_email",
        "create_task_for_event"
    ]
}

# System Calibration Contexts (Highest Priority)
SYSTEM_CALIBRATION = {
    "verbs": [
        # Direct calibration actions
        "calibrate", "adjust", "configure", "setup", "tune",
        "optimize", "fix", "correct", "improve", "tweak",
        
        # System check actions
        "test", "check", "verify", "diagnose", "analyze",
        "monitor", "assess", "evaluate", "measure",
        
        # Audio-specific actions
        "normalize", "balance", "level", "equalize",
        "filter", "clean", "clear", "enhance",
        
        # Adjustment requests
        "increase", "decrease", "lower", "raise",
        "reduce", "boost", "modify", "change",
        "turn up", "turn down", "adjust"
    ],
    "contexts": [
        # Core system terms
        "voice", "audio", "sound", "volume",
        "microphone", "mic", "input", "detection",
        "sensitivity", "recognition", "pickup",
        
        # Quality indicators
        "quality", "clarity", "accuracy", "precision",
        "performance", "response", "feedback", "output",
        
        # Issue indicators
        "noise", "echo", "background", "static",
        "interference", "static", "distortion", "feedback",
        "cutting out", "breaking up", "choppy",
        
        # Settings references
        "settings", "configuration", "setup", "system",
        "levels", "threshold", "baseline", "default",
        "preferences", "options", "parameters"
    ],
    "tools": ["calibrate_voice_detection"]
}

# Location Services Contexts
LOCATION_CONTEXTS = {
    "verbs": [
        # Direct location queries
        "where", "find", "locate", "get", "tell",
        "show", "give", "share", "display",
        
        # Navigation related
        "navigate", "direct", "guide", "lead",
        "take", "bring", "move", "go",
        
        # Position queries
        "position", "pinpoint", "spot", "mark",
        "identify", "determine", "establish",
        
        # Status queries
        "check", "verify", "confirm", "update",
        "track", "monitor", "watch"
    ],
    "contexts": [
        # Location terms
        "location", "place", "spot", "position",
        "address", "coordinates", "whereabouts",
        "site", "venue", "destination",
        
        # Geographic terms
        "latitude", "longitude", "coords",
        "map", "direction", "distance", "route",
        "path", "way", "street", "road",
        
        # Relative location
        "here", "current", "present", "local",
        "nearby", "around", "area", "vicinity",
        "zone", "region", "district", "neighborhood",
        
        # Building/Place types
        "building", "office", "home", "work",
        "restaurant", "store", "shop", "mall",
        "park", "station", "airport"
    ],
    "tools": ["get_location"]
}

# Time Query Contexts
TIME_CONTEXTS = {
    "verbs": [
        # Query actions
        "tell", "check", "what", "show", "give",
        "display", "see", "know", "get", "confirm",
        
        # Time-specific
        "time", "date", "schedule", "plan",
        "track", "monitor", "record", "log"
    ],
    "contexts": [
        # Time units
        "time", "hour", "minute", "second",
        "date", "day", "month", "year",
        "morning", "afternoon", "evening", "night",
        
        # Time queries
        "current", "now", "today", "tonight",
        "present", "moment", "instant", "currently",
        
        # Time zones
        "local", "zone", "timezone", "utc",
        "gmt", "est", "pst", "cst", "mst",
        
        # Relative time
        "right now", "at the moment", "presently",
        "currently", "exactly", "precise"
    ],
    "tools": ["get_current_time"]
}

# Master context dictionary
TOOL_CONTEXTS = {
    "system": SYSTEM_CALIBRATION,      # Highest priority
    "calendar": CALENDAR_CONTEXTS,
    "email": EMAIL_CONTEXTS,
    "tasks": TASK_CONTEXTS,
    "location": LOCATION_CONTEXTS,
    "time": TIME_CONTEXTS
}


# System State Command Patterns
# Central source of truth for all system-changing commands
SYSTEM_STATE_COMMANDS = {
    "tool": {
        "enable": [
            "activate tools", "launch toolkit", "begin assistance",
            "enable tool use", "start tools", "enable assistant",
            "tools online", "enable tools", 
            "toolkit online", "utilities on", 
            "activate functions", "tools ready", "tools on", 
            "toolkit on", "functions on", "start using tools", "start tool use", 
            "prepare toolkit", "let's use tools", "bring tools online"
        ],
        "disable": [
            "tools offline", "end toolkit", "close assistant",
            "stop tools", "disable assistant", "conversation only",
            "assistant power down", "toolkit offline", 
            "helper mode inactive", "utilities off",
            "deactivate functions", "tools away", "tools off", 
            "toolkit off", "functions off", "disable tools", 
            "disable tool use", "sleep tools", "dismiss toolkit", 
            "take tools offline"
        ]
    },
    "document": {
        "load": [
            "load file", "load files", "load documents",
            "load my file", "read my file", "read my files", "load my files", "load all files"
        ],
        "unload": [
            "offload file", "offload files", "remove files",
            "clear files", "clear documents", "unload files",
            "remove my files", "clear my files", "offload my files", 
            "offload all files", "remove all files", "clear all files"
        ]
    },
    "calibration": {
        "run": [
            "calibrate voice", "run calibration",
            "start voice calibration",
            "run voice calibration", "voice detection calibration",
            "calibrate voice detection"
        ]
    }
}

def match_system_command(transcript: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Check if transcript matches any system state command.
    
    Args:
        transcript (str): The voice input to check
        
    Returns:
        Tuple[Optional[str], Optional[str], Optional[str]]: A tuple containing:
            - category (str or None): The system command category (tool/document/calibration)
            - action (str or None): The specific action (enable/disable/load/unload/run)
            - matched_command (str or None): The exact command that matched
            
    Example:
        >>> match_system_command("enable tools")
        ('tool', 'enable', 'enable tools')
        >>> match_system_command("hello")
        (None, None, None)
    """
    transcript_lower = transcript.lower()
    
    for category, actions in SYSTEM_STATE_COMMANDS.items():
        for action, commands in actions.items():
            for command in commands:
                if command in transcript_lower:
                    return category, action, command
                    
    return None, None, None
