"""Symmetric encryption for OAuth tokens stored in the database.

Tokens are encrypted with Fernet (AES-128-CBC + HMAC) using a key held in the
environment, never in the database or in source control. A stolen copy of
db.sqlite3 alone is therefore not enough to call the Google Health API.
"""

import logging
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    try:
        return Fernet(settings.FERNET_KEY.encode())
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            "FERNET_KEY in .env is not a valid Fernet key. Generate one with:\n"
            '  .venv/Scripts/python.exe -c "from cryptography.fernet import '
            'Fernet; print(Fernet.generate_key().decode())"'
        ) from exc


def encrypt(plaintext: str) -> str:
    """Encrypt a token for storage. Returns URL-safe base64 text."""
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str | None:
    """Decrypt a stored token.

    Returns None if the value cannot be decrypted, which in practice means
    FERNET_KEY was rotated. Callers treat a missing token the same as an
    expired one: the user is asked to reconnect their Google account.
    """
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        logger.warning(
            "Stored token could not be decrypted - FERNET_KEY probably changed. "
            "The affected user must reconnect their Google account."
        )
        return None
