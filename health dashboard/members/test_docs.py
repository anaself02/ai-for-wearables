"""The ethics note says what the account system is for, and nothing in the
repository creates an account or a password on its own."""

import re
from pathlib import Path

from django.test import SimpleTestCase

BASE = Path(__file__).resolve().parent.parent


class EthicsNoteTests(SimpleTestCase):
    def setUp(self):
        self.text = " ".join((BASE / "docs" / "ETHICS_AND_PRIVACY.md")
                             .read_text(encoding="utf-8").split())

    def test_it_says_the_accounts_exist_only_to_separate_the_two_profiles(self):
        self.assertIn("The account system exists only to separate two people's "
                      "profiles on one machine.", self.text)

    def test_it_covers_emails_as_personal_data_and_the_admin_rules(self):
        for phrase in ("Emails are personal data", "never written to a log",
                       "never shown to another account", "pending claim",
                       "createsuperuser", "does not display password hashes",
                       "The uploaded files are not kept"):
            self.assertIn(phrase, self.text, phrase)


class NoAccountsAreCreatedByCodeTests(SimpleTestCase):
    def test_no_application_code_migration_or_command_creates_users_or_passwords(self):
        offenders = []
        pattern = re.compile(r"create_user\(|create_superuser\(|set_password\(|"
                             r"make_password\(|User\.objects\.create\(")
        for top in ("config", "health", "people", "members"):
            for path in (BASE / top).rglob("*.py"):
                if (path.name.startswith("test") or path.name in ("testing.py", "fitguard.py")
                        or "__pycache__" in path.parts):
                    continue
                text = path.read_text(encoding="utf-8")
                for number, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line) and not line.lstrip().startswith("#"):
                        offenders.append("{}:{}".format(path.relative_to(BASE), number))
        # The only place an account is created is the sign-up form, deliberately.
        self.assertTrue(offenders)
        for offender in offenders:
            self.assertTrue(offender.replace("\\", "/").startswith("members/forms.py"),
                            offender)

    def test_there_are_no_fixtures_with_accounts(self):
        fixtures = [p for p in BASE.rglob("*.json")
                    if "fixtures" in p.parts and ".venv" not in p.parts]
        self.assertEqual(fixtures, [])
