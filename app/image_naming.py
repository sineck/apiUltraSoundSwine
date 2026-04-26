from datetime import datetime
import re
import uuid


def build_image_filename(kind: str, page_number: int | None = None, extension: str = ".png") -> str:
    """Build a generated image filename without using the uploaded source name."""
    safe_kind = re.sub(r"[^a-zA-Z0-9]+", "_", kind).strip("_").lower() or "image"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_id = uuid.uuid4().hex[:8]
    suffix = extension if extension.startswith(".") else f".{extension}"
    if page_number is not None:
        return f"{safe_kind}_{timestamp}_p{page_number:03d}_{unique_id}{suffix}"
    return f"{safe_kind}_{timestamp}_{unique_id}{suffix}"
