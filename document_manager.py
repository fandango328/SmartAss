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
from config import ANTHROPIC_MODEL
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
            '.png': 'image/jpeg',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.JPG': 'image/jpeg',  # iPhone uppercase variant
            '.JPEG': 'image/jpeg', # Other possible uppercase variants
            '.PNG': 'image/jpeg',
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

    async def load_all_files(self, clear_existing: bool = False) -> bool:
        """Load all supported files from the query_files directory with case-insensitive extension matching."""
        try:
            if clear_existing:
                await self.offload_all_files()
            
            # Get all files in the directory first
            all_files = os.listdir(self.query_files_dir)
            print(f"Found {len(all_files)} total files in directory")
            
            files_processed = 0
            
            # Process each file regardless of case
            for filename in all_files:
                try:
                    filepath = os.path.join(self.query_files_dir, filename)
                    
                    # Skip if not a file
                    if not os.path.isfile(filepath):
                        continue
                    
                    # Get extension and convert to lowercase for matching
                    _, ext_with_case = os.path.splitext(filename)
                    ext_lower = ext_with_case.lower()

                    # Skip if extension not supported
                    if ext_lower not in self.supported_extensions:
                        continue

                    # Get mime type FIRST before using it
                    mime_type = self.supported_extensions[ext_lower]

                    # Validate and check memory limits
                    if not self._validate_file(filepath):
                        print(f"Skipping invalid file: {filepath}")
                        continue

                    file_size = os.path.getsize(filepath)
                    if not self._check_memory_limit(file_size):
                        print(f"Warning: Loading {filepath} would exceed memory limit")
                        continue

                    await asyncio.sleep(0.01)  # Prevent blocking

                    # Now we can safely use mime_type
                    if mime_type.startswith('image/'):
                        print(f"Processing image file: {filepath}")
                        success = await self.process_image(filepath, filename)
                        
                        if success:
                            self.loaded_files[filename] = success  # Store the processed image data
                            self.current_memory_usage += file_size
                            print(f"Successfully loaded image: {filename}")
                            files_processed += 1
                        else:
                            print(f"Failed to process image: {filename}")
                    
                    elif mime_type.startswith('text/') or mime_type in ['application/json', 'application/x-yaml']:
                        # Handle text-based files
                        try:
                            print(f"Processing text file: {filepath}")
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                                print(f"Successfully read {len(content)} bytes from {filename}")
                                
                                # Store both text content and base64 for consistency
                                b64_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
                                self.loaded_files[filename] = {
                                    'type': 'text',
                                    'mime_type': mime_type,
                                    'content': content,
                                    'base64': b64_content,
                                    'size': f"{file_size/1024:.1f}KB",
                                    'description': f"[Text File: {filename}, Size: {file_size/1024:.1f}KB, Type: {mime_type}]"
                                }
                                self.current_memory_usage += file_size
                                print(f"Successfully loaded text file: {filename}")
                                files_processed += 1
                        except UnicodeDecodeError:
                            # If text decoding fails, fall through to binary handling
                            print(f"Text decoding failed for {filename}, handling as binary")
                            raise
                            
                    else:
                        # Handle binary files (including PDFs and other non-text files)
                        print(f"Processing binary file: {filepath}")
                        with open(filepath, 'rb') as f:
                            binary_content = f.read()
                            b64_content = base64.b64encode(binary_content).decode('utf-8')
                            self.loaded_files[filename] = {
                                'type': 'binary',
                                'mime_type': mime_type,
                                'base64': b64_content,
                                'size': f"{file_size/1024:.1f}KB",
                                'description': f"[Binary File: {filename}, Size: {file_size/1024:.1f}KB, Type: {mime_type}]"
                            }
                            self.current_memory_usage += file_size
                            print(f"Successfully loaded binary file: {filename}")
                            files_processed += 1
                    
                except Exception as e:
                    print(f"Error loading file {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            self.files_loaded = len(self.loaded_files) > 0
            print(f"Files processed: {files_processed}, Files loaded: {len(self.loaded_files)}, Status: {self.files_loaded}")
            return self.files_loaded
            
        except Exception as e:
            print(f"Error in load_all_files: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_claude_message_blocks(self) -> list:
        """
        Return loaded files in the format required for Anthropic Claude Vision API.
        Each image is included as a separate message block with correct media_type and base64 data.
        Each text file is included as a "type": "text" block.
        """
        blocks = []
        for filename, file_data in self.loaded_files.items():
            if file_data['type'] == 'image':
                # Add a "text" announcement for image (optional, helps with context)
                blocks.append({
                    "type": "text",
                    "text": f"Image: {filename}"
                })
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": file_data['mime_type'],
                        "data": file_data['base64']
                    }
                })
            elif file_data['type'] == 'text':
                blocks.append({
                    "type": "text",
                    "text": f"File: {filename}\n{file_data['content']}"
                })
            elif file_data['type'] == 'binary':
                blocks.append({
                    "type": "text",
                    "text": f"[Binary file {filename} of type {file_data['mime_type']}, size: {file_data['size']}]"
                })
        return blocks

    async def process_image(self, filepath, filename):
        """Process image file and optimize for Claude Vision, keeping correct format and media_type."""
        try:
            file_size = os.path.getsize(filepath)
            ext = os.path.splitext(filepath)[1].lower()
            mime_type = self.supported_extensions[ext]

            def process_image_sync():
                with Image.open(filepath) as img:
                    # Convert to appropriate mode
                    if img.mode in ('RGBA', 'LA') and mime_type == 'image/png':
                        # Keep alpha for PNG
                        pass
                    elif img.mode in ('RGBA', 'LA'):
                        img = img.convert('RGB')
                    # Resize to Claude requirements (min 200px, max 1568px)
                    img = self.optimize_image_dimensions(img)
                    width, height = img.size
                    # Always use PNG if alpha, JPEG otherwise
                    if img.mode in ('RGBA', 'LA') or ext == '.png':
                        out_format = "PNG"
                        out_mime = "image/png"
                    else:
                        out_format = "JPEG"
                        out_mime = "image/jpeg"
                    buffered = io.BytesIO()
                    img.save(buffered, format=out_format, quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    return {
                        'type': 'image',
                        'mime_type': out_mime,
                        'dimensions': f"{width}x{height}",
                        'size': f"{file_size/1024:.1f}KB",
                        'base64': img_str,
                        'description': f"[Image File: {filename}, Dimensions: {width}x{height}, Size: {file_size/1024:.1f}KB]"
                    }
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, process_image_sync)
        except Exception as img_err:
            print(f"Error processing image {filepath}: {img_err}")
            import traceback
            traceback.print_exc()
            return None

    def get_loaded_content(self) -> dict:
        """Get the content of all loaded files separated by type."""
        if not self.files_loaded:
            return {"system_content": "", "image_content": []}
                
        text_context = []
        image_blocks = []
            
        for filename, file_data in self.loaded_files.items():
            if file_data['type'] == 'image':
                # Format image for Claude Vision API
                image_blocks.extend([
                    {
                        "type": "text",
                        "text": f"Image {filename}:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": file_data['mime_type'],
                            "data": file_data['base64']
                        }
                    }
                ])
            elif file_data['type'] == 'text':
                # Add text content to system context
                text_context.append(f"\n### File: {filename} ###")
                text_context.append(file_data['content'])
                text_context.append("### End File ###\n")
            elif file_data['type'] == 'binary':
                # Add binary file info to system context
                text_context.append(f"\n### File: {filename} ###")
                text_context.append(f"[Binary File: {filename}, Size: {file_data['size']}, Type: {file_data['mime_type']}]")
                text_context.append("### End File ###\n")
                    
        return {
            "system_content": "\n".join(text_context),
            "image_content": image_blocks
        }
    
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
