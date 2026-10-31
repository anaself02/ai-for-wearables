"""The app needs no Google sign-in, no token encryption and no secrets but one.

These tests guard the removal of the Google connection: that the app starts
with an environment holding only the Django secret key, that nothing still
imports the Google or encryption libraries, and that the tables left behind by
the old sign-in are dropped.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

from django.db import connection
from django.test import TestCase

BASE_DIR = Path(__file__).resolve().parent.parent

START_WITHOUT_ANY_GOOGLE_VALUES = """
import os
import dotenv
dotenv.load_dotenv = lambda *args, **kwargs: False    # do not read the real .env
os.environ["DJANGO_SECRET_KEY"] = "only-value-the-app-needs"
os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
for name in list(os.environ):
    assert not name.startswith(("GOOGLE_", "FERNET")), name
import django
django.setup()
from django.conf import settings
from django.core.management import call_command
call_command("check")
for name in dir(settings):
    assert not name.startswith(("GOOGLE_", "FERNET")), name
assert settings.ALLOWED_HOSTS == ["localhost", "127.0.0.1"], settings.ALLOWED_HOSTS
print("started")
"""


class StartsWithoutGoogleTests(TestCase):
    def test_the_app_starts_with_only_the_django_secret_key(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("GOOGLE_", "FERNET", "DJANGO_"))}
        result = subprocess.run(
            [sys.executable, "-c", START_WITHOUT_ANY_GOOGLE_VALUES],
            cwd=BASE_DIR, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        self.assertIn("started", result.stdout)


class NothingImportsGoogleOrEncryptionTests(TestCase):
    def test_no_application_module_imports_the_removed_libraries(self):
        banned = re.compile(
            r"^\s*(import|from)\s+(google|google_auth_oauthlib|oauthlib|requests_oauthlib|"
            r"requests|cryptography|accounts)\b", re.M)
        offenders = []
        for top in ("config", "health", "people"):
            for path in (BASE_DIR / top).rglob("*.py"):
                if "migrations" in path.parts:
                    continue
                text = path.read_text(encoding="utf-8")
                for match in banned.finditer(text):
                    offenders.append("{}: {}".format(path.relative_to(BASE_DIR),
                                                     match.group(0).strip()))
        self.assertEqual(offenders, [])

    def test_requirements_no_longer_list_the_google_or_encryption_packages(self):
        text = (BASE_DIR / "requirements.txt").read_text(encoding="utf-8")
        for package in ("google-auth", "google-auth-oauthlib", "requests", "cryptography"):
            self.assertNotRegex(text, r"(?m)^{}\b".format(re.escape(package)))

    def test_the_env_example_asks_for_no_google_or_fernet_values(self):
        text = (BASE_DIR / ".env.example").read_text(encoding="utf-8")
        values = [line for line in text.splitlines()
                  if line and not line.lstrip().startswith("#")]
        for line in values:
            self.assertNotRegex(line, r"^(GOOGLE_|FERNET)")

    def test_the_old_routes_are_gone(self):
        for url in ("/oauth/login/", "/oauth/callback/", "/oauth/logout/", "/settings/"):
            self.assertEqual(self.client.get(url).status_code, 404, url)


class DropOldTablesMigrationTests(TestCase):
    def sql(self):
        from importlib import import_module
        module = import_module("people.migrations.0003_drop_oauth_tables")
        return module.Migration.operations[0].sql

    def test_it_drops_the_token_and_profile_tables_and_is_safe_to_repeat(self):
        with connection.cursor() as cursor:
            cursor.execute("CREATE TABLE accounts_googlecredential (id integer)")
            cursor.execute("CREATE TABLE accounts_profile (id integer)")
            self.assertIn("accounts_googlecredential", connection.introspection.table_names(cursor))
            for statement in self.sql():
                cursor.execute(statement)
            names = connection.introspection.table_names(cursor)
            self.assertNotIn("accounts_googlecredential", names)
            self.assertNotIn("accounts_profile", names)
            for statement in self.sql():          # nothing to drop: must not raise
                cursor.execute(statement)
