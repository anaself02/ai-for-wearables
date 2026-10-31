"""Signing out, and what the browser is allowed to keep afterwards."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


User = get_user_model()


class SignOutTests(TestCase):
    def setUp(self):
        self.user = User.objects.create(username="tester@example.com")
        self.client.force_login(self.user)

    def test_signing_out_ends_the_session(self):
        self.client.post(reverse("accounts:sign_out"))
        self.assertIsNone(self.client.session.session_key)
