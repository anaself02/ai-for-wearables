"""A short-lived holding area between "here is the validation report" and
"yes, load it".

Uploading is two requests so nothing is saved before it has been shown. The
files from the first request have to survive until the second, so they are
written to a spool directory under a random token and deleted as soon as the
upload is confirmed, cancelled, or an hour old.

A token belongs to the person it was made for. Confirming it under a different
person's URL is refused, so a report for one person can never be loaded into
another.
"""

import json
import os
import re
import secrets
import shutil
import time

from django.conf import settings

from .sources import SourceFile

TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
MAX_AGE_SECONDS = 3600


class BadToken(Exception):
    pass


def _root():
    root = str(settings.UPLOAD_SPOOL_DIR)
    os.makedirs(root, exist_ok=True)
    return root


def _dir(token):
    if not TOKEN_RE.match(token or ""):
        raise BadToken("That upload reference is not valid.")
    return os.path.join(_root(), token)


def cleanup(max_age=MAX_AGE_SECONDS):
    """Delete spool directories older than `max_age` seconds."""
    root = _root()
    now = time.time()
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and now - os.path.getmtime(path) > max_age:
            shutil.rmtree(path, ignore_errors=True)


def save(person, source_files, skipped_names):
    """Write the accepted files for one person; return the token."""
    cleanup()
    token = secrets.token_urlsafe(18)
    path = _dir(token)
    os.makedirs(path)
    names = []
    for index, source_file in enumerate(source_files):
        with open(os.path.join(path, "{:05d}.bin".format(index)), "wb") as handle:
            handle.write(source_file.data)
        names.append(source_file.name)
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"person": person.slug, "names": names,
                   "skipped_names": skipped_names}, handle)
    return token


def _manifest(person, token):
    path = _dir(token)
    try:
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (FileNotFoundError, ValueError):
        raise BadToken("That upload has expired or was already used. "
                       "Upload the files again.")
    if manifest.get("person") != person.slug:
        raise BadToken("That upload was prepared for a different person and "
                       "cannot be loaded here.")
    return path, manifest


def load(person, token):
    """-> (list of SourceFile, skipped names) for this person's token."""
    path, manifest = _manifest(person, token)
    files = []
    for index, name in enumerate(manifest["names"]):
        with open(os.path.join(path, "{:05d}.bin".format(index)), "rb") as handle:
            files.append(SourceFile(name, handle.read()))
    return files, manifest.get("skipped_names", [])


def discard(person, token):
    try:
        path, _ = _manifest(person, token)
    except BadToken:
        return
    shutil.rmtree(path, ignore_errors=True)
