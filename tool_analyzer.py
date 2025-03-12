#!/usr/bin/env python3

"""
Tool Analysis Module for LAURA Voice Assistant

Provides intelligent analysis of user queries against the tool context taxonomy.
Handles detection, scoring, and resolution of tool requirements.

Change Classification: S (Small)
Component Type: Core Logic
Dependencies: tool_contexts.py
"""

from typing import Tuple, Set, Dict, Optional, NamedTuple
from dataclasses import dataclass
import re
from tool_context import TOOL_CONTEXTS

@dataclass
class ToolMatch:
    """Represents a matched tool with confidence and context"""
    category: str
    tools: Set[str]
    confidence: float
    matched_verbs: Set[str]
    matched_contexts: Set[str]
    primary_action: Optional[str]

class ToolAnalysis:
    """Analyzes user queries for tool requirements using context matching"""
    
    def __init__(self):
        self.context_cache = {}  # Cache for frequently used patterns
    
    def analyze_query(self, query: str) -> ToolMatch:
        """
        Analyzes a query for tool requirements with confidence scoring.
        
        Args:
            query: The user's input query
            
        Returns:
            ToolMatch: Contains matched tools, confidence, and context
            
        Example:
            >>> analyzer.analyze_query("Can you schedule a meeting for tomorrow?")
            ToolMatch(
                category='calendar',
                tools={'create_calendar_event'},
                confidence=0.85,
                matched_verbs={'schedule'},
                matched_contexts={'meeting'},
                primary_action='create'
            )
        """
        query_lower = query.lower()
        best_match = None
        highest_confidence = 0.0
        
        # Pre-process query for better matching
        query_terms = set(query_lower.split())
        
        for category, matchers in TOOL_CONTEXTS.items():
            # Find all matching verbs and contexts
            matched_verbs = {verb for verb in matchers["verbs"] 
                           if verb in query_lower}
            matched_contexts = {ctx for ctx in matchers["contexts"] 
                              if ctx in query_lower}
            
            if matched_verbs and matched_contexts:
                # Calculate confidence based on:
                # 1. Number of matches
                # 2. Specificity of matches
                # 3. Position of matches in query
                confidence = self._calculate_confidence(
                    query_lower,
                    matched_verbs,
                    matched_contexts
                )
                
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_match = ToolMatch(
                        category=category,
                        tools=set(matchers["tools"]),
                        confidence=confidence,
                        matched_verbs=matched_verbs,
                        matched_contexts=matched_contexts,
                        primary_action=self._determine_primary_action(
                            matched_verbs,
                            category
                        )
                    )
        
        return best_match if best_match else ToolMatch(
            category="none",
            tools=set(),
            confidence=0.0,
            matched_verbs=set(),
            matched_contexts=set(),
            primary_action=None
        )
    
    def _calculate_confidence(self, 
                            query: str, 
                            verbs: Set[str], 
                            contexts: Set[str]) -> float:
        """
        Calculates confidence score for matches.
        
        Scoring factors:
        - Number and quality of matches
        - Position of matches in query
        - Completeness of action-context pairs
        """
        base_score = 0.0
        
        # Weight verb-context pairs more heavily
        paired_score = len(verbs) * len(contexts) * 0.3
        
        # Position scoring - earlier matches weighted more heavily
        position_score = 0.0
        words = query.split()
        for i, word in enumerate(words):
            position_weight = 1.0 - (i / len(words))
            if word in verbs:
                position_score += position_weight * 0.2
            if word in contexts:
                position_score += position_weight * 0.15
                
        # Combine scores with weights
        confidence = (
            paired_score * 0.5 +
            position_score * 0.3 +
            (len(verbs) + len(contexts)) * 0.1
        )
        
        return min(confidence, 1.0)
    
    def _determine_primary_action(self, 
                                verbs: Set[str], 
                                category: str) -> Optional[str]:
        """Maps matched verbs to primary actions for the tool category"""
        action_maps = {
            "calendar": {
                "create": {"schedule", "book", "plan", "create", "add"},
                "update": {"move", "reschedule", "change", "modify"},
                "delete": {"cancel", "remove", "delete"},
                "query": {"check", "show", "find", "list"}
            },
            "email": {
                "create": {"send", "write", "draft", "compose"},
                "read": {"check", "read", "show", "get"},
                "modify": {"archive", "flag", "mark", "move"}
            }
            # Add maps for other categories
        }
        
        if category in action_maps:
            for action, action_verbs in action_maps[category].items():
                if verbs & action_verbs:  # If any verbs match
                    return action
                    
        return None

def get_tools_for_query(query: str) -> Tuple[bool, Set[str], float]:
    """
    Main interface function for query analysis.
    Uses strict 90% threshold for tool detection.
    
    Args:
        query: User's text query
        
    Returns:
        Tuple[bool, Set[str], float]: (needs_tools, tool_names, confidence)
            - needs_tools will only be True if confidence >= 0.9
    """
    analyzer = ToolAnalysis()
    result = analyzer.analyze_query(query)
    
    # Use strict 90% threshold for determining if tools are needed
    needs_tools = result.confidence >= 0.9 and bool(result.tools)
    
    return (
        needs_tools,
        result.tools,
        result.confidence
    )