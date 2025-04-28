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
