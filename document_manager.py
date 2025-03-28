#!/usr/bin/env python3

# =============================================================================
# Standard Library Imports
# =============================================================================
import os
import glob
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

class DocumentManager:
    """
    Manages document loading and offloading for LAURA's query processing.
    
    Last Updated: 2025-03-27 20:00:48 UTC
    Author: fandango328
    """
    
    def __init__(self):
        """Initialize the DocumentManager with default settings."""
        # Base configuration
        self.query_files_dir = '/home/user/LAURA/query_files/'
        self.max_file_size = 40_000_000  # 40MB per file
        self.total_memory_limit = 100_000_000  # 100MB total
        
        # State tracking
        self.loaded_files: Dict[str, str] = {}  # {filename: content}
        self.current_memory_usage: int = 0
        self.files_loaded: bool = False
        
        # Supported file types and their MIME types
        self.supported_extensions = {
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.json': 'application/json',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml',
        }
        
        # Ensure query_files directory exists
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the query_files directory exists."""
        os.makedirs(self.query_files_dir, exist_ok=True)

    async def load_all_files(self) -> bool:
        """Load all supported files from the query_files directory."""
        try:
            # Clear any previously loaded files
            await self.offload_all_files()  # Note: now awaiting this call
            
            # Get all files in directory
            for ext in self.supported_extensions:
                files = glob.glob(os.path.join(self.query_files_dir, f'*{ext}'))
                for filepath in files:
                    try:
                        # Basic validation
                        if not self._validate_file(filepath):
                            continue
                        
                        # Check memory limits
                        file_size = os.path.getsize(filepath)
                        if not self._check_memory_limit(file_size):
                            print(f"Warning: Loading {filepath} would exceed memory limit")
                            continue
                        
                        # Add slight delay to prevent blocking
                        await asyncio.sleep(0.01)
                        
                        # Read and store file content
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        filename = os.path.basename(filepath)
                        self.loaded_files[filename] = content
                        self.current_memory_usage += file_size
                        
                    except Exception as e:
                        print(f"Error loading file {filepath}: {e}")
                        continue
            
            self.files_loaded = len(self.loaded_files) > 0
            return self.files_loaded
            
        except Exception as e:
            print(f"Error in load_all_files: {e}")
            return False

    async def offload_all_files(self) -> None:
        """Clear all loaded files from memory."""
        try:
            # Add small delay to prevent blocking
            await asyncio.sleep(0.01)
            self.loaded_files.clear()
            self.current_memory_usage = 0
            self.files_loaded = False
        except Exception as e:
            print(f"Error in offload_all_files: {e}")

    def get_loaded_content(self) -> str:
        """
        Get the content of all loaded files in a format suitable for context.
        
        Returns:
            str: Formatted string containing all loaded file contents.
        """
        if not self.files_loaded:
            return ""
            
        context = []
        for filename, content in self.loaded_files.items():
            # Add file separator and metadata
            context.append(f"\n### File: {filename} ###")
            # Add truncated content (first 1000 chars) with ellipsis if needed
            truncated_content = content[:1000] + ('...' if len(content) > 1000 else '')
            context.append(truncated_content)
            context.append("### End File ###\n")
            
        return "\n".join(context)

    def _validate_file(self, filepath: str) -> bool:
        """
        Validate a file for loading.
        
        Args:
            filepath: Path to the file to validate.
            
        Returns:
            bool: True if file is valid, False otherwise.
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                return False
                
            # Check file extension
            ext = os.path.splitext(filepath)[1].lower()
            if ext not in self.supported_extensions:
                return False
                
            # Check file size
            if os.path.getsize(filepath) > self.max_file_size:
                return False
                
            return True
            
        except Exception as e:
            print(f"Error validating file {filepath}: {e}")
            return False

    def _check_memory_limit(self, additional_size: int) -> bool:
        """
        Check if loading additional content would exceed memory limits.
        
        Args:
            additional_size: Size in bytes of new content to be loaded.
            
        Returns:
            bool: True if within limits, False if would exceed limits.
        """
        return (self.current_memory_usage + additional_size) <= self.total_memory_limit

    def get_status(self) -> dict:
        """
        Get current status of document manager.
        
        Returns:
            dict: Status information including loaded files and memory usage.
        """
        return {
            "files_loaded": len(self.loaded_files),
            "memory_usage": self.current_memory_usage,
            "memory_limit": self.total_memory_limit,
            "loaded_files": list(self.loaded_files.keys())
        }
