import hashlib
import os
from loguru import logger
from typing import Optional
import uuid

def get_md5_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_file_md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()

def generate_unique_id() -> str:
    return str(uuid.uuid4())

def get_file_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()

def sanitize_filename(filename: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def format_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"
