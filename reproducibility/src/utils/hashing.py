import hashlib
import os


def compute_sha256(file_path: str, block_size: int = 65536) -> str:
    """Compute SHA256 checksum of a file.
    
    Args:
        file_path: Absolute or relative path to file.
        block_size: Buffer block size for reading.
        
    Returns:
        Hexadecimal SHA256 string.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()
