"""A short-lived holding area between "here is the validation report" and
"yes, load it".

Importing is two requests so nothing is saved before it has been shown. What
waits between them is NOT the uploaded files: those are parsed in the first
request and discarded when it ends. Only the parsed readings (time, bpm, source
code) are parked, under a random token, and they are deleted as soon as the import
is confirmed or cancelled, or when they are an hour old.

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
from datetime import datetime, timezone

import numpy as np
from django.conf import settings

from .sources.base import FileReport, ParsedFile

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
        raise BadToken("That import reference is not valid.")
    return os.path.join(_root(), token)


def cleanup(max_age=MAX_AGE_SECONDS):
    """Delete spool directories older than `max_age` seconds."""
    root = _root()
    now = time.time()
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and now - os.path.getmtime(path) > max_age:
            shutil.rmtree(path, ignore_errors=True)


def _epoch(moment):
    return None if moment is None else int(moment.timestamp())


def _moment(epoch):
    return None if epoch is None else datetime.fromtimestamp(epoch, timezone.utc)


def save(person, loads, skipped_names):
    """Park PARSED files for one person; return the token.

    `loads` is [(sha256, file name, ParsedFile)]. No original bytes are written.
    """
    cleanup()
    token = secrets.token_urlsafe(18)
    path = _dir(token)
    os.makedirs(path)
    files = []
    for index, (digest, name, parsed) in enumerate(loads):
        np.savez(os.path.join(path, "{:05d}.npz".format(index)),
                 epochs=parsed.epochs, bpm=parsed.bpm, codes=parsed.source_codes)
        r = parsed.report
        files.append({"sha256": digest, "name": name, "source_names": parsed.source_names,
                      "rows": r.rows, "first": _epoch(r.first_utc),
                      "last": _epoch(r.last_utc),
                      "dropped_unparseable": r.dropped_unparseable,
                      "dropped_duplicate": r.dropped_duplicate})
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"person": person.slug, "files": files,
                   "skipped_names": skipped_names}, handle)
    return token


def _manifest(person, token):
    path = _dir(token)
    try:
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (FileNotFoundError, ValueError):
        raise BadToken("That import has expired or was already used. "
                       "Choose the files again.")
    if manifest.get("person") != person.slug:
        raise BadToken("That import was prepared for a different person and "
                       "cannot be loaded here.")
    return path, manifest


def load(person, token):
    """-> ([(sha256, file name, ParsedFile)], skipped names) for this person's token."""
    path, manifest = _manifest(person, token)
    items = []
    for index, entry in enumerate(manifest["files"]):
        with np.load(os.path.join(path, "{:05d}.npz".format(index))) as arrays:
            epochs, bpm, codes = arrays["epochs"], arrays["bpm"], arrays["codes"]
        report = FileReport(
            name=entry["name"], status="parsed", rows=entry["rows"],
            first_utc=_moment(entry["first"]), last_utc=_moment(entry["last"]),
            dropped_unparseable=entry["dropped_unparseable"],
            dropped_duplicate=entry["dropped_duplicate"])
        items.append((entry["sha256"], entry["name"], ParsedFile(
            epochs=epochs, bpm=bpm, source_names=entry["source_names"],
            source_codes=codes, report=report)))
    return items, manifest.get("skipped_names", [])


def discard(person, token):
    try:
        path, _ = _manifest(person, token)
    except BadToken:
        return
    shutil.rmtree(path, ignore_errors=True)
