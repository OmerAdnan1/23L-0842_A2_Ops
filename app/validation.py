"""
Task 4.2 - Validate an uploaded file entirely in memory (nothing is written to disk).

Rejects: missing/empty files, extensions other than jpg/jpeg/png, and files that
claim to be images but cannot be decoded or are not really JPEG/PNG.
The size limit is enforced by Flask's MAX_CONTENT_LENGTH (HTTP 413).
"""
from io import BytesIO

from flask import Request
from PIL import Image
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}


class UploadError(ValueError):
    pass


class InMemoryRequest(Request):
    """Werkzeug spools uploads over 500 KB to a temp file; keep them in RAM instead
    (bounded by MAX_CONTENT_LENGTH) so no uploaded image ever touches the disk."""

    def _get_file_stream(self, total_content_length, content_type, filename=None,
                         content_length=None):
        return BytesIO()


def check_decodable(raw):
    try:
        with Image.open(BytesIO(raw)) as im:
            im.verify()
        with Image.open(BytesIO(raw)) as im:
            fmt = im.format
            im.load()
    except Exception as e:
        raise UploadError("The file looks like an image but could not be decoded.") from e
    if fmt not in ALLOWED_FORMATS:
        raise UploadError(f"The file content is {fmt}, not a JPEG or PNG image.")


def read_upload(file):
    """Return (raw_bytes, safe_filename) or raise UploadError."""
    if file is None or not file.filename:
        raise UploadError("No file was selected.")
    name = secure_filename(file.filename) or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(f"Unsupported file type '.{ext}'. Upload a JPG or PNG image.")
    raw = file.read()
    if not raw:
        raise UploadError("The uploaded file is empty.")
    check_decodable(raw)
    return raw, name
