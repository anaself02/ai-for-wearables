"""Checks on template source that rendering tests cannot catch.

A template can be wrong in a way that still returns HTTP 200 with all the
expected content in it. The comment case below shipped for exactly that
reason: every assertion about the sleep page passed while the page carried a
paragraph of source commentary at the top of it.
"""

import pathlib
import re

from django.conf import settings
from django.test import SimpleTestCase

TEMPLATE_DIRS = [pathlib.Path(directory) for directory in settings.TEMPLATES[0]["DIRS"]]


def template_files():
    for directory in TEMPLATE_DIRS:
        for path in sorted(directory.rglob("*.html")):
            yield path, path.read_text(encoding="utf-8")


class TemplateSourceTests(SimpleTestCase):
    def test_templates_are_found(self):
        """Guard the guard: a bad path here would make the checks below vacuous."""
        self.assertGreater(len(list(template_files())), 10)

    def test_no_multiline_hash_comments(self):
        """Django's {# #} comment is single-line only.

        Opened on one line and closed on another, it is not a comment at all:
        the parser never recognises it, and every word of it is rendered into
        the page. Multi-line commentary needs {% comment %}.
        """
        offenders = []
        for path, source in template_files():
            for match in re.finditer(r"\{#", source):
                tail = source[match.start():]
                newline = tail.find("\n")
                closed = tail.find("#}")
                if closed == -1 or (newline != -1 and newline < closed):
                    line = source[: match.start()].count("\n") + 1
                    offenders.append(f"{path.name}:{line} {tail[:60]!r}")

        self.assertEqual(
            offenders,
            [],
            "these open {# on one line and close it on another, so they render "
            "as page text - use {% comment %} instead: " + ", ".join(offenders),
        )
