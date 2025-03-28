#!/usr/bin/env python3

# =============================================================================
# Standard Library Imports
# =============================================================================
import os
import glob
import asyncio
import base64
import io
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from PIL import Image 


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
            # Text formats
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.json': 'application/json',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml',
            
            # Image formats
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.ico': 'image/x-icon',
            
            # Document formats
            '.pdf': 'application/pdf',
            
            # Vector formats
            '.svg': 'image/svg+xml',
            
            # Additional web formats
            '.avif': 'image/avif',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff'
        }
        
        # Ensure query_files directory exists
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the query_files directory exists."""
        os.makedirs(self.query_files_dir, exist_ok=True)

    async def offload_all_files(self) -> None:
        """Clear all loaded files from memory."""
        self.loaded_files.clear()
        self.current_memory_usage = 0
        self.files_loaded = False
        print("All files have been offloaded from memory")

    async def load_all_files(self, clear_existing: bool = False) -> bool:
        """
        Load all supported files from the query_files directory.
        
        Args:
            clear_existing: If True, clear existing files before loading new ones
        """
        try:
            if clear_existing:
                await self.offload_all_files()
            
            for ext in self.supported_extensions:
                files = glob.glob(os.path.join(self.query_files_dir, f'*{ext}'))
                for filepath in files:
                    try:
                        if not self._validate_file(filepath):
                            continue
                        
                        file_size = os.path.getsize(filepath)
                        if not self._check_memory_limit(file_size):
                            print(f"Warning: Loading {filepath} would exceed memory limit")
                            continue
                        
                        await asyncio.sleep(0.01)  # Prevent blocking
                        
                        filename = os.path.basename(filepath)
                        mime_type = self.supported_extensions[ext]
                        
                        if mime_type.startswith('image/'):
                            try:
                                # Open and process image
                                with Image.open(filepath) as img:
                                    # Convert to RGB if necessary
                                    if img.mode in ('RGBA', 'LA'):
                                        img = img.convert('RGB')
                                    
                                    # Get image details
                                    width, height = img.size
                                    
                                    # Convert to base64
                                    buffered = io.BytesIO()
                                    img.save(buffered, format="JPEG", quality=85)
                                    img_str = base64.b64encode(buffered.getvalue()).decode()
                                    
                                    # Store image info and base64 data
                                    self.loaded_files[filename] = {
                                        'type': 'image',
                                        'mime_type': mime_type,
                                        'dimensions': f"{width}x{height}",
                                        'size': f"{file_size/1024:.1f}KB",
                                        'base64': img_str,
                                        'description': f"[Image File: {filename}, Dimensions: {width}x{height}, Size: {file_size/1024:.1f}KB]"
                                    }
                            except Exception as img_err:
                                print(f"Error processing image {filepath}: {img_err}")
                                continue
                                
                        else:
                            # Handle text-based files as before
                            try:
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    self.loaded_files[filename] = {
                                        'type': 'text',
                                        'mime_type': mime_type,
                                        'content': content,
                                        'size': f"{file_size/1024:.1f}KB"
                                    }
                            except UnicodeDecodeError:
                                # If text reading fails, try binary + base64
                                with open(filepath, 'rb') as f:
                                    binary_content = f.read()
                                    b64_content = base64.b64encode(binary_content).decode('utf-8')
                                    self.loaded_files[filename] = {
                                        'type': 'binary',
                                        'mime_type': mime_type,
                                        'base64': b64_content,
                                        'size': f"{file_size/1024:.1f}KB"
                                    }
                        
                        self.current_memory_usage += file_size
                        print(f"Successfully loaded: {filename}")
                        
                    except Exception as e:
                        print(f"Error loading file {filepath}: {e}")
                        continue
            
            self.files_loaded = len(self.loaded_files) > 0
            return self.files_loaded
            
        except Exception as e:
            print(f"Error in load_all_files: {e}")
            return False

    def get_loaded_content(self) -> str:
        """
        Get the content of all loaded files in a format suitable for context.
        
        Returns:
            str: Formatted string containing all loaded file contents.
        """
        if not self.files_loaded:
            return ""
            
        context = []
        for filename, file_data in self.loaded_files.items():
            context.append(f"\n### File: {filename} ###")
            
            if file_data['type'] == 'image':
                # For images, just add the description
                context.append(file_data['description'])
            elif file_data['type'] == 'text':
                # For text files, add truncated content
                content = file_data['content']
                truncated_content = content[:1000] + ('...' if len(content) > 1000 else '')
                context.append(truncated_content)
            elif file_data['type'] == 'binary':
                # For binary files, add file info
                context.append(f"[Binary File: {filename}, Size: {file_data['size']}, Type: {file_data['mime_type']}]")
                
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
