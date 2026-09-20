"""Signing out, and what the browser is allowed to keep afterwards."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from accounts.middleware import NO_STORE

User = get_user_model()


class SignOutTests(TestCase):
    def setUp(self):
        self.user = User.objects.create(username="tester@example.com")
        self.client.force_login(self.user)

    def test_signing_out_ends_the_session(self):
        self.client.post(reverse("accounts:sign_out"))
        self.assertIsNone(self.client.session.session_key)

    def test_a_protected_page_redirects_once_signed_out(self):
        self.client.post(reverse("accounts:sign_out"))
        response = self.client.get(reverse("health:fitness"))
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response["Location"].startswith("/?next="))

    def test_signed_in_pages_forbid_being_stored(self):
        """The back button after sign-out replays whatever the browser kept."""
        for name in ("health:index", "health:fitness", "health:sleep_calculator"):
            response = self.client.get(reverse(name))
            self.assertEqual(response.headers.get("Cache-Control"), NO_STORE, name)
            self.assertIn("no-store", response.headers["Cache-Control"], name)

    def test_the_front_door_is_not_needlessly_uncacheable(self):
        """Nothing private is on it, so it is left alone."""
        self.client.logout()
        response = self.client.get(reverse("accounts:landing"))
        self.assertNotIn("no-store", response.headers.get("Cache-Control", ""))
