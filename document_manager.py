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
from typing import Dict, Optional, List, Any
from datetime import datetime
from PIL import Image
from config import ANTHROPIC_MODEL
from anthropic import Anthropic #type: ignore
from secret import ANTHROPIC_API_KEY

class DocumentManager:
    def __init__(self):
        """Initialize the DocumentManager with default settings."""
        # Base configuration
        self.query_files_dir = '/home/user/LAURA/query_files/'
        self.max_file_size = 40_000_000  # 40MB per file
        self.total_memory_limit = 100_000_000  # 100MB total
        
        # State tracking
        self.loaded_files: Dict[str, Dict[str, Any]] = {}  # {filename: file_data_dict}
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
            '.JPG': 'image/jpeg',  # iPhone uppercase variant
            '.JPEG': 'image/jpeg', # Other possible uppercase variants
            '.PNG': 'image/png',
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
            print(f"[DocumentManager] Found {len(all_files)} total items in {self.query_files_dir}")
            
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
                        print(f"[DocumentManager] Skipping unsupported extension: {filename}")
                        continue

                    # Get mime type
                    mime_type = self.supported_extensions[ext_lower]

                    # Validate and check memory limits
                    if not self._validate_file(filepath):
                        print(f"[DocumentManager] Skipping invalid file: {filename}")
                        continue

                    file_size = os.path.getsize(filepath)
                    if not self._check_memory_limit(file_size):
                        print(f"[WARN DocumentManager] Not loading {filename}, would exceed memory limit.")
                        continue

                    await asyncio.sleep(0.01)  # Prevent blocking

                    # Read file as binary
                    with open(filepath, 'rb') as f:
                        binary_content = f.read()
                    
                    base64_content = base64.b64encode(binary_content).decode('utf-8')
                    
                    # Create base file entry
                    file_entry = {
                        'filename': filename,
                        'mime_type': mime_type,
                        'base64': base64_content,
                        'size': file_size,
                        'size_display': f"{file_size/1024:.1f}KB",
                    }

                    # Process based on file type
                    if mime_type.startswith('image/'):
                        print(f"[DocumentManager] Processing image file: {filename}")
                        processed_image = await self.process_image(binary_data=binary_content, filename=filename)
                        
                        if processed_image:
                            file_entry.update({
                                'type': 'image',
                                'base64': processed_image['base64'],
                                'mime_type': 'image/jpeg',
                                'dimensions': processed_image['dimensions'],
                                'description': f"[Image File: {filename}, Dimensions: {processed_image['dimensions']}, Size: {file_entry['size_display']}]"
                            })
                        else:
                            print(f"[DocumentManager] Failed to process image: {filename}, treating as binary")
                            file_entry.update({
                                'type': 'binary',
                                'description': f"[Binary File: {filename}, Size: {file_entry['size_display']}, Type: {mime_type}]"
                            })
                            
                    elif mime_type == 'application/pdf':
                        print(f"[DocumentManager] Processing PDF file: {filename}")
                        file_entry.update({
                            'type': 'pdf',
                            'description': f"[PDF File: {filename}, Size: {file_entry['size_display']}]"
                        })
                        
                    elif mime_type.startswith('text/') or mime_type in ['application/json', 'application/x-yaml']:
                        print(f"[DocumentManager] Processing text file: {filename}")
                        try:
                            # Try to decode as UTF-8 text
                            content_text = binary_content.decode('utf-8')
                            file_entry.update({
                                'type': 'text',
                                'content': content_text,
                                'description': f"[Text File: {filename}, Size: {file_entry['size_display']}, Type: {mime_type}]"
                            })
                        except UnicodeDecodeError:
                            print(f"[DocumentManager] Text decoding failed for {filename}, treating as binary")
                            file_entry.update({
                                'type': 'binary',
                                'description': f"[Binary File: {filename}, Size: {file_entry['size_display']}, Type: {mime_type}]"
                            })
                    else:
                        # Handle as generic binary file
                        print(f"[DocumentManager] Processing binary file: {filename}")
                        file_entry.update({
                            'type': 'binary',
                            'description': f"[Binary File: {filename}, Size: {file_entry['size_display']}, Type: {mime_type}]"
                        })

                    # Store the file entry
                    self.loaded_files[filename] = file_entry
                    self.current_memory_usage += file_size
                    files_processed += 1
                    print(f"[DocumentManager] Loaded {file_entry['type']} file: {filename}")
                    
                except Exception as e:
                    print(f"[ERROR DocumentManager] Error loading file {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            self.files_loaded = len(self.loaded_files) > 0
            print(f"[DocumentManager] Load all files complete. Total loaded: {len(self.loaded_files)}. Status: {self.files_loaded}")
            return self.files_loaded
            
        except Exception as e:
            print(f"[ERROR DocumentManager] Error in load_all_files: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_document_content_for_claude(self, filename: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get properly formatted Anthropic content blocks for a specific document.
        Returns blocks in the format expected by Anthropic's API.
        """
        file_data = self.loaded_files.get(filename)
        if not file_data:
            return None

        blocks = []
        file_type = file_data['type']
        base64_content = file_data['base64']
        original_filename = file_data['filename']
        mime_type = file_data['mime_type']

        if file_type == 'image':
            # Add text announcement for context
            blocks.append({"type": "text", "text": f"Image content from file: {original_filename}"})
            # Add properly formatted image block
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64_content
                }
            })
            
        elif file_type == 'pdf':
            # Add text announcement for context
            blocks.append({"type": "text", "text": f"PDF document content from file: {original_filename}"})
            # Add properly formatted document block for PDF
            blocks.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "name": original_filename,
                    "data": base64_content
                }
            })
            
        elif file_type == 'text' and 'content' in file_data:
            # Add text content directly
            blocks.append({
                "type": "text", 
                "text": f"Text file content from '{original_filename}':\n\n{file_data['content']}"
            })
            
        elif file_type == 'binary':
            # Announce binary file presence
            blocks.append({
                "type": "text", 
                "text": f"[A binary file named '{original_filename}' (type: {mime_type}) was provided but cannot be directly processed.]"
            })
            
        else:
            # Fallback for unknown types
            blocks.append({
                "type": "text", 
                "text": f"[File: {original_filename} of unknown type '{file_type}']"
            })

        return blocks

    def get_all_loaded_document_blocks_for_claude(self) -> List[Dict[str, Any]]:
        """
        Get properly formatted Anthropic content blocks for ALL currently loaded documents.
        Returns blocks in the format expected by Anthropic's API.
        """
        all_blocks = []
        
        if not self.files_loaded:
            return []
            
        for filename in self.loaded_files.keys():
            doc_blocks = self.get_document_content_for_claude(filename)
            if doc_blocks:
                all_blocks.extend(doc_blocks)
                
        return all_blocks

    def get_claude_message_blocks(self) -> List[Dict[str, Any]]:
        """
        Legacy method for backward compatibility.
        Returns the same format as get_all_loaded_document_blocks_for_claude.
        """
        return self.get_all_loaded_document_blocks_for_claude()

    async def process_image(self, filepath=None, filename=None, binary_data=None):
        """
        Process image: always convert to JPEG, enforce dimension limits (min 200px, max 1568px on sides),
        and output as base64 JPEG for Claude Vision compatibility.
        Can accept either filepath or binary_data to avoid double file reads.
        """
        try:
            if filepath and not binary_data:
                file_size = os.path.getsize(filepath)
                with open(filepath, 'rb') as f:
                    binary_data = f.read()
            elif binary_data:
                file_size = len(binary_data)
            else:
                raise ValueError("Must provide either filepath or binary_data")

            def process_image_sync():
                img_io = io.BytesIO(binary_data)
                with Image.open(img_io) as img:
                    # Always convert to RGB (JPEG does not support alpha)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    width, height = img.size
                    
                    # Resize if outside allowed size range
                    min_dim = min(width, height)
                    max_dim = max(width, height)
                    if max_dim > 1568:
                        # Scale down
                        scale = 1568.0 / max_dim
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                    elif min_dim < 200:
                        # Scale up
                        scale = 200.0 / min_dim
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                    # Else: already within limits, leave alone
                    
                    final_width, final_height = img.size
                    
                    # Save as JPEG
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    return {
                        'type': 'image',
                        'mime_type': 'image/jpeg',
                        'dimensions': f"{final_width}x{final_height}",
                        'size': f"{file_size/1024:.1f}KB",
                        'base64': img_str,
                        'description': f"[Image File: {filename}, Dimensions: {final_width}x{final_height}, Size: {file_size/1024:.1f}KB]"
                    }
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, process_image_sync)
            
        except Exception as img_err:
            print(f"[ERROR DocumentManager] Error processing image {filename}: {img_err}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_loaded_content(self) -> dict:
        """Get the content of all loaded files separated by type (legacy compatibility method)."""
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
            elif file_data['type'] in ['binary', 'pdf']:
                # Add file info to system context
                text_context.append(f"\n### File: {filename} ###")
                text_context.append(file_data['description'])
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
                print(f"[DocumentManager] File {filepath} exceeds max size ({self.max_file_size} bytes)")
                return False
                
            return True
            
        except Exception as e:
            print(f"[ERROR DocumentManager] Error validating file {filepath}: {e}")
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
            "memory_usage_mb": f"{self.current_memory_usage / 1024 / 1024:.1f}MB",
            "memory_limit_mb": f"{self.total_memory_limit / 1024 / 1024:.1f}MB",
            "loaded_files": list(self.loaded_files.keys()),
            "files_by_type": {
                file_type: [name for name, data in self.loaded_files.items() if data['type'] == file_type]
                for file_type in set(data['type'] for data in self.loaded_files.values())
            }
        }
