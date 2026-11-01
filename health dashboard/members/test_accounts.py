"""Sign-up, login, logout, the throttle, cookies, privacy of emails.

Every account here is synthetic. Passwords are random per run (or obviously weak
test strings for the validator tests) and are never stored.
"""

import logging
import secrets
from unittest import mock

from django.contrib.auth import get_user_model
from django.core import mail
from django.core.cache import cache
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from members import throttle
from people import links
from people.models import Person, ProfileClaim
from people.testing import make_user

User = get_user_model()


def strong():
    return secrets.token_urlsafe(18)


class CleanCache(TestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)


class SignupTests(CleanCache):
    def post(self, email="Someone@Example.com", password=None, confirm=None,
             person=None, client=None):
        password = password or strong()
        data = {"email": email, "password1": password,
                "password2": password if confirm is None else confirm,
                "person": (person if person is not None
                           else Person.objects.get(slug="anas").pk)}
        return (client or self.client).post(reverse("members:signup"), data)

    def test_the_page_says_no_email_is_sent_and_an_admin_must_approve(self):
        html = " ".join(self.client.get(reverse("members:signup")).content.decode().split())
        self.assertIn("No email is sent or verified", html)
        self.assertIn("An administrator must approve", html)
        self.assertIn("before any data is visible", html)

    def test_the_profile_choice_lists_names_only(self):
        html = self.client.get(reverse("members:signup")).content.decode()
        self.assertIn(">Anas<", html)
        self.assertIn(">Rahil<", html)
        self.assertNotIn("@", html.split("<main")[1].split("</main>")[0].replace(
            "autocomplete", "").replace("email", ""))

    def test_a_linked_profile_is_not_offered(self):
        make_user("owner@example.com", linked_to=Person.objects.get(slug="anas"))
        html = self.client.get(reverse("members:signup")).content.decode()
        self.assertNotIn(">Anas<", html)
        self.assertIn(">Rahil<", html)

    def test_signing_up_creates_a_lowercased_account_and_a_pending_claim_only(self):
        response = self.post(email="  Someone@Example.COM ")
        self.assertRedirects(response, reverse("members:login"))
        user = User.objects.get(email="someone@example.com")
        self.assertEqual(user.username, "someone@example.com")
        self.assertFalse(user.is_staff)
        self.assertTrue(user.has_usable_password())
        claim = user.claims.get()
        self.assertEqual((claim.status, claim.person.slug), ("pending", "anas"))
        self.assertIsNone(Person.objects.get(slug="anas").owner)       # never auto-linked
        self.assertIsNone(links.linked_person(user))

    def test_no_email_is_ever_sent(self):
        self.post()
        self.assertEqual(mail.outbox, [])
        # The test runner swaps in its own backend, so read the configured one.
        from pathlib import Path
        source = (Path(__file__).resolve().parent.parent / "config" / "settings.py"
                  ).read_text(encoding="utf-8")
        self.assertIn('EMAIL_BACKEND = "django.core.mail.backends.dummy.EmailBackend"',
                      source)

    def test_signing_up_does_not_log_the_user_in(self):
        self.post()
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_a_duplicate_email_is_refused_case_insensitively(self):
        self.post(email="dup@example.com")
        response = self.post(email="DUP@Example.com", person=Person.objects.get(slug="rahil").pk)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "An account with this email already exists.")
        self.assertEqual(User.objects.filter(email__iexact="dup@example.com").count(), 1)

    def test_a_duplicate_of_an_existing_username_is_refused_too(self):
        User.objects.create_user(username="Mixed@Example.com", password=strong())
        response = self.post(email="mixed@example.com")
        self.assertContains(response, "already exists")

    def test_the_two_passwords_must_match(self):
        response = self.post(password=strong(), confirm=strong())
        self.assertContains(response, "The two passwords do not match.")
        self.assertFalse(User.objects.exists())

    def test_weak_passwords_are_refused_by_djangos_validators(self):
        for weak in ("short1", "password", "12345678901", "someone@example.com"):
            response = self.post(email="someone@example.com", password=weak)
            self.assertEqual(response.status_code, 200, weak)
            self.assertContains(response, "field-error", msg_prefix=weak)
        self.assertFalse(User.objects.exists())

    def test_an_invalid_email_is_refused(self):
        response = self.post(email="not-an-email")
        self.assertContains(response, "Enter a valid email address.")

    def test_a_profile_choice_is_required_and_must_be_available(self):
        response = self.post(person="")
        self.assertContains(response, "Choose which profile you are.")
        anas = Person.objects.get(slug="anas")
        make_user("owner@example.com", linked_to=anas)
        response = self.post(person=anas.pk)
        self.assertContains(response, "not available")
        self.assertEqual(User.objects.filter(email="someone@example.com").count(), 0)

    def test_a_password_is_never_echoed_back_on_an_error(self):
        secret = strong()
        response = self.post(password=secret, confirm=secret + "x")
        self.assertNotContains(response, secret)

    def test_the_form_needs_a_csrf_token(self):
        strict = Client(enforce_csrf_checks=True)
        response = self.post(client=strict)
        self.assertEqual(response.status_code, 403)
        self.assertFalse(User.objects.exists())


class LoginLogoutTests(CleanCache):
    def setUp(self):
        super().setUp()
        self.user, self.password = make_user("person@example.com")

    def login(self, email="person@example.com", password=None, **extra):
        return self.client.post(reverse("members:login"),
                                {"email": email, "password": password or self.password,
                                 **extra})

    def test_a_correct_login_signs_in_and_redirects_home(self):
        response = self.login()
        self.assertRedirects(response, "/", fetch_redirect_response=False)
        self.assertEqual(int(self.client.session["_auth_user_id"]), self.user.pk)

    def test_the_email_is_case_insensitive_at_login(self):
        self.assertEqual(self.login(email="  PERSON@Example.COM ").status_code, 302)

    def test_a_safe_next_is_followed_and_an_unsafe_one_is_not(self):
        self.assertEqual(self.login(next="/p/anas/")["Location"], "/p/anas/")
        self.client.logout()
        self.assertEqual(self.login(next="https://evil.example/steal")["Location"], "/")
        self.client.logout()
        self.assertEqual(self.login(next="//evil.example")["Location"], "/")

    def test_wrong_password_and_unknown_email_give_the_same_message(self):
        wrong = self.login(password="not-the-password")
        unknown = self.login(email="nobody@example.com")
        self.assertContains(wrong, "Email or password is incorrect.")
        self.assertContains(unknown, "Email or password is incorrect.")
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_an_inactive_account_cannot_log_in(self):
        self.user.is_active = False
        self.user.save()
        self.assertContains(self.login(), "Email or password is incorrect.")

    def test_logout_is_post_only(self):
        self.login()
        self.assertEqual(self.client.get(reverse("members:logout")).status_code, 405)
        self.assertIn("_auth_user_id", self.client.session)
        response = self.client.post(reverse("members:logout"))
        self.assertRedirects(response, reverse("members:login"))
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_login_and_logout_need_a_csrf_token(self):
        strict = Client(enforce_csrf_checks=True)
        self.assertEqual(strict.post(reverse("members:login"),
                                     {"email": "person@example.com",
                                      "password": self.password}).status_code, 403)
        strict.force_login(self.user)                 # logout is only reachable signed in
        self.assertEqual(strict.post(reverse("members:logout")).status_code, 403)
        self.assertIn("_auth_user_id", strict.session)


class ThrottleTests(CleanCache):
    def setUp(self):
        super().setUp()
        self.user, self.password = make_user("target@example.com")

    def attempt(self, email="target@example.com", password="wrong-password"):
        return self.client.post(reverse("members:login"),
                                {"email": email, "password": password})

    def test_five_failures_lock_the_email_for_five_minutes(self):
        for _ in range(4):
            self.assertContains(self.attempt(), "Email or password is incorrect.")
        locked = self.attempt()                                   # the fifth failure
        self.assertContains(locked, "Too many failed attempts")
        self.assertTrue(throttle.is_locked("target@example.com"))

    def test_a_correct_password_is_refused_while_locked(self):
        for _ in range(5):
            self.attempt()
        response = self.attempt(password=self.password)
        self.assertContains(response, "Too many failed attempts")
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_the_lock_expires_after_five_minutes(self):
        for _ in range(5):
            self.attempt()
        with mock.patch("time.time", return_value=__import__("time").time() + 299):
            self.assertTrue(throttle.is_locked("target@example.com"))
        with mock.patch("time.time", return_value=__import__("time").time() + 301):
            self.assertFalse(throttle.is_locked("target@example.com"))
            response = self.attempt(password=self.password)
            self.assertEqual(response.status_code, 302)

    def test_a_success_clears_the_count(self):
        for _ in range(4):
            self.attempt()
        self.assertEqual(self.attempt(password=self.password).status_code, 302)
        self.client.logout()
        for _ in range(4):
            self.assertContains(self.attempt(), "Email or password is incorrect.")

    def test_unknown_emails_are_throttled_with_the_same_message(self):
        for _ in range(4):
            self.assertContains(self.attempt(email="ghost@example.com"),
                                "Email or password is incorrect.")
        self.assertContains(self.attempt(email="ghost@example.com"),
                            "Too many failed attempts")

    def test_the_throttle_is_per_email(self):
        other, other_password = make_user("other@example.com")
        for _ in range(5):
            self.attempt()
        self.assertEqual(self.attempt(email="other@example.com",
                                      password=other_password).status_code, 302)

    def test_the_email_is_case_insensitive_for_the_count(self):
        for email in ("Target@Example.com", "TARGET@example.com", "target@example.com",
                      " target@example.com", "target@EXAMPLE.com"):
            self.attempt(email=email)
        self.assertTrue(throttle.is_locked("target@example.com"))

    def test_the_cache_never_holds_the_email_itself(self):
        for _ in range(5):
            self.attempt()
        from django.core.cache import caches
        keys = list(caches["default"]._cache.keys())
        self.assertTrue(keys)
        for key in keys:
            self.assertNotIn("target@example.com", key)
            self.assertNotIn("example.com", key)


class NothingSensitiveIsLoggedTests(CleanCache):
    def test_emails_and_passwords_never_reach_the_logs(self):
        records = []

        class Collect(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = Collect(level=logging.DEBUG)
        root = logging.getLogger()
        previous = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        self.addCleanup(lambda: (root.removeHandler(handler), root.setLevel(previous)))

        password = strong()
        email = "logged.person@example.com"
        self.client.post(reverse("members:signup"), {
            "email": email, "password1": password, "password2": password,
            "person": Person.objects.get(slug="anas").pk})
        for _ in range(6):
            self.client.post(reverse("members:login"),
                             {"email": email, "password": "wrong-password-value"})
        self.client.post(reverse("members:login"), {"email": email, "password": password})

        text = "\n".join(records)
        for secret in (email, password, "wrong-password-value", "logged.person"):
            self.assertNotIn(secret, text)

    def test_the_members_source_has_no_logging_of_any_kind(self):
        import pathlib
        import re
        offenders = []
        for path in pathlib.Path(__file__).resolve().parent.glob("*.py"):
            if path.name.startswith("test"):
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r"\blogger\b|logging\.|print\(", line):
                    offenders.append("{}:{}".format(path.name, number))
        self.assertEqual(offenders, [])


class CookieAndHeaderTests(CleanCache):
    def test_the_session_cookie_is_httponly_and_samesite_lax(self):
        user, password = make_user("cookie@example.com")
        response = self.client.post(reverse("members:login"),
                                    {"email": "cookie@example.com", "password": password})
        morsel = response.cookies["sessionid"]
        self.assertTrue(morsel["httponly"])
        self.assertEqual(morsel["samesite"], "Lax")

    def test_the_csrf_cookie_is_httponly_and_samesite_lax(self):
        response = self.client.get(reverse("members:login"))
        morsel = response.cookies["csrftoken"]
        self.assertTrue(morsel["httponly"])
        self.assertEqual(morsel["samesite"], "Lax")

    def test_signed_in_responses_are_no_store(self):
        user, password = make_user("store@example.com")
        self.client.post(reverse("members:login"),
                         {"email": "store@example.com", "password": password})
        response = self.client.get(reverse("members:waiting"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("no-store", response["Cache-Control"])

    def test_the_server_only_answers_to_localhost_names(self):
        import subprocess
        import sys
        from pathlib import Path
        base = Path(__file__).resolve().parent.parent
        code = ("import os,dotenv;dotenv.load_dotenv=lambda *a,**k:False;"
                "os.environ['DJANGO_SECRET_KEY']='x';"
                "os.environ['DJANGO_SETTINGS_MODULE']='config.settings';"
                "import django;django.setup();from django.conf import settings;"
                "print(settings.ALLOWED_HOSTS)")
        out = subprocess.run([sys.executable, "-c", code], cwd=base, capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.stdout.strip(), "['localhost', '127.0.0.1']")


class WaitingPageTests(CleanCache):
    def setUp(self):
        super().setUp()
        self.user, self.password = make_user("wait@example.com")
        self.client.force_login(self.user)
        self.anas = Person.objects.get(slug="anas")

    def test_a_pending_claim_is_shown_by_profile_name_with_no_data(self):
        links.create_claim(self.user, self.anas)
        html = self.client.get(reverse("members:waiting")).content.decode()
        self.assertIn("waiting for an administrator", html)
        self.assertIn("Anas", html)
        self.assertNotIn("wait@example.com", html)
        self.assertNotIn("<canvas", html)
        self.assertNotIn("<table", html)

    def test_after_a_rejection_the_account_can_ask_again(self):
        claim = links.create_claim(self.user, self.anas)
        links.reject_claim(claim)
        html = self.client.get(reverse("members:waiting")).content.decode()
        self.assertIn("was not approved", html)
        response = self.client.post(reverse("members:claim"),
                                    {"person": Person.objects.get(slug="rahil").pk})
        self.assertRedirects(response, reverse("members:waiting"))
        self.assertEqual(links.pending_claim(self.user).person.slug, "rahil")
        self.assertIsNone(Person.objects.get(slug="rahil").owner)

    def test_a_claim_on_a_linked_profile_is_refused(self):
        make_user("owner@example.com", linked_to=self.anas)
        response = self.client.post(reverse("members:claim"), {"person": self.anas.pk},
                                    follow=True)
        self.assertEqual(ProfileClaim.objects.count(), 0)
        self.assertContains(response, "not available")

    def test_a_linked_account_is_sent_away_from_the_waiting_page(self):
        links.create_claim(self.user, self.anas)
        links.approve_claim(self.user.claims.get())
        response = self.client.get(reverse("members:waiting"))
        self.assertRedirects(response, "/", fetch_redirect_response=False)

    def test_waiting_needs_a_login(self):
        self.client.logout()
        response = self.client.get(reverse("members:waiting"))
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response["Location"].startswith("/login/"))
