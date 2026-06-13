"""Shared helpers for accepting input drawing images (CLI + web).

Drawings are JPEG/PNG only. Media type is inferred from the file suffix with a
magic-byte fallback, so no image-decoding dependency (Pillow) is needed —
PydanticAI's ``BinaryContent`` only needs raw bytes plus a media type.
"""

from __future__ import annotations

ALLOWED_MEDIA_TYPES = ("image/png", "image/jpeg")
MAX_DRAWING_BYTES = 10 * 1024 * 1024  # 10 MB per drawing
MAX_DRAWINGS = 5

_SUFFIX_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def media_type_for(filename: str | None, data: bytes = b"") -> str | None:
    """Best-effort media type from a filename suffix, falling back to magic bytes.

    Returns one of ALLOWED_MEDIA_TYPES, or ``None`` if it is neither PNG nor JPEG.
    """
    if filename:
        suffix = filename.lower().rsplit(".", 1)
        if len(suffix) == 2:
            media = _SUFFIX_MEDIA.get("." + suffix[1])
            if media is not None:
                return media
    if data[:8] == _PNG_MAGIC:
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":  # JPEG SOI + marker
        return "image/jpeg"
    return None


def drawing_filename(index: int, media_type: str) -> str:
    """Deterministic on-disk/served name for the `index`-th input drawing (1-based)."""
    suffix = ".png" if media_type == "image/png" else ".jpg"
    return f"drawing_{index:02d}{suffix}"
