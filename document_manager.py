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
from config_cl import ANTHROPIC_MODEL
from anthropic import Anthropic
from secret import ANTHROPIC_API_KEY

class DocumentManager:
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
        
        # Initialize Anthropic client
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        
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

    def optimize_image_dimensions(self, img):
        """
        Optimize image dimensions according to Claude Vision requirements:
        - Max dimension: 1568px on longest side
        - Min dimension: 200px on shortest side
        - Maintains aspect ratio
        """
        width, height = img.size
        aspect_ratio = width / height
        
        # Check if image needs resizing
        needs_resize = False
        
        # Handle too large (max 1568px on longest side)
        if width > 1568 or height > 1568:
            needs_resize = True
            if width > height:
                new_width = 1568
                new_height = int(1568 / aspect_ratio)
            else:
                new_height = 1568
                new_width = int(1568 * aspect_ratio)
                
        # Handle too small (min 200px on shortest side)
        elif width < 200 or height < 200:
            needs_resize = True
            if width < height:
                new_width = 200
                new_height = int(200 * aspect_ratio)
            else:
                new_height = 200
                new_width = int(200 / aspect_ratio)
        
        if needs_resize:
            return img.resize((new_width, new_height), Image.LANCZOS)
        return img

    async def process_image(self, filepath, filename):
        """Process image file and get Claude's description"""
        try:
            with Image.open(filepath) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')
                
                # Optimize image dimensions
                img = self.optimize_image_dimensions(img)
                
                # Get image details
                width, height = img.size
                file_size = os.path.getsize(filepath)
                
                # Calculate approximate tokens (width * height / 750)
                image_tokens = (width * height) / 750
                
                # Get original file extension and mime type
                ext = os.path.splitext(filepath)[1].lower()
                mime_type = self.supported_extensions[ext]
                
                # Convert to base64, maintaining original format
                buffered = io.BytesIO()
                img_format = mime_type.split('/')[-1].upper()
                img.save(buffered, format=img_format)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Prepare message for Claude Vision API
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model=ANTHROPIC_MODEL,
                        max_tokens=1000,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": mime_type,
                                            "data": img_str
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": "Describe this image in detail, including any text, visual elements, colors, and their relationships."
                                    }
                                ]
                            }
                        ]
                    )
                )
                
                image_description = response.content[0].text
                
                # Store both the image data and description
                self.loaded_files[filename] = {
                    'type': 'image',
                    'mime_type': mime_type,  # Use original mime_type
                    'dimensions': f"{width}x{height}",
                    'size': f"{file_size/1024:.1f}KB",
                    'tokens': int(image_tokens),
                    'base64': img_str,
                    'description': image_description
                }
                
                return True
                
        except Exception as e:
            print(f"Error processing image {filename}: {str(e)}")
            return False

    async def load_all_files(self, clear_existing: bool = False) -> bool:
        """Load all supported files from the query_files directory."""
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
                            # Create a new event loop for the synchronous API call
                            success = await asyncio.get_event_loop().run_in_executor(
                                None, 
                                self.client.messages.create,
                                ANTHROPIC_MODEL,
                                max_tokens=1000,
                                messages=[...]  # message structure as before
                            )
                            
                            if success:
                                print(f"Successfully loaded: {filename}")
                            else:
                                print(f"Failed to load: {filename}")
                                continue
                        else:
                            # Handle text-based files
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
        """Get the content of all loaded files in a format suitable for context."""
        if not self.files_loaded:
            return ""
            
        context = []
        for filename, file_data in self.loaded_files.items():
            context.append(f"\n### File: {filename} ###")
            
            if file_data['type'] == 'image':
                # For images, include the description
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
        """Validate a file for loading."""
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
        """Check if loading additional content would exceed memory limits."""
        return (self.current_memory_usage + additional_size) <= self.total_memory_limit

    def get_status(self) -> dict:
        """Get current status of document manager."""
        return {
            "files_loaded": len(self.loaded_files),
            "memory_usage": self.current_memory_usage,
            "memory_limit": self.total_memory_limit,
            "loaded_files": list(self.loaded_files.keys())
        }
