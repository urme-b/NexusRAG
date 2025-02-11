import re

def clean_text(raw_text):
    """
    Removes excessive whitespace, non-ASCII characters, etc.
    Returns cleaned string.
    """
    # Replace multiple whitespace/newlines with a single space
    text = re.sub(r'\s+', ' ', raw_text)
    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()
    # Trim leading/trailing
    text = text.strip()
    return text