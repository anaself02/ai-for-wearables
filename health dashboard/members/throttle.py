"""A login throttle: 5 failed attempts per email, then a 5-minute lock.

Counted in Django's cache, keyed by a hash of the lowercased email, so the email
itself is never stored in the cache or written to a log. Unknown emails are
counted like known ones, so the throttle (and the error message) cannot be used
to find out which emails have accounts.

While an email is locked every attempt is refused WITHOUT checking the password,
so a correct password cannot be used to probe the lock.
"""

import hashlib

from django.core.cache import cache

MAX_ATTEMPTS = 5
LOCK_SECONDS = 5 * 60


def _digest(email):
    return hashlib.sha256((email or "").strip().lower().encode()).hexdigest()


def _failures_key(email):
    return "login-failures:" + _digest(email)


def _lock_key(email):
    return "login-lock:" + _digest(email)


def is_locked(email):
    return cache.get(_lock_key(email)) is not None


def record_failure(email):
    """Count one failed attempt; lock the email on the fifth. Returns True if locked."""
    key = _failures_key(email)
    cache.add(key, 0, timeout=LOCK_SECONDS)
    count = cache.incr(key)
    if count >= MAX_ATTEMPTS:
        cache.set(_lock_key(email), True, timeout=LOCK_SECONDS)
        cache.delete(key)
        return True
    return False


def reset(email):
    cache.delete(_failures_key(email))
    cache.delete(_lock_key(email))
