import hashlib
from base64 import urlsafe_b64encode
from datetime import datetime, timedelta, timezone as dt_timezone
from urllib.parse import parse_qs, urlparse

from django.contrib.auth import get_user_model
from django.db import connection
from django.test import TestCase
from google.oauth2.credentials import Credentials

from .models import GoogleCredential
from .views import SESSION_STATE_KEY, SESSION_VERIFIER_KEY

User = get_user_model()


class EncryptedTokenStorageTests(TestCase):
    def setUp(self):
        self.user = User.objects.create(username="tester@example.com")
        self.credential = GoogleCredential.objects.create(
            user=self.user,
            google_sub="1234567890",
            access_token="ya29.access-token-value",
            refresh_token="1//refresh-token-value",
        )

    def test_tokens_round_trip_through_the_orm(self):
        reloaded = GoogleCredential.objects.get(pk=self.credential.pk)
        self.assertEqual(reloaded.access_token, "ya29.access-token-value")
        self.assertEqual(reloaded.refresh_token, "1//refresh-token-value")

    def test_tokens_are_ciphertext_in_the_database(self):
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT access_token, refresh_token "
                "FROM accounts_googlecredential WHERE id = %s",
                [self.credential.pk],
            )
            access, refresh = cursor.fetchone()

        self.assertNotIn("ya29", access)
        self.assertNotIn("refresh-token-value", refresh)
        # Fernet tokens are versioned and start with 0x80, encoding to "gAAAAA".
        self.assertTrue(access.startswith("gAAAAA"))
        self.assertTrue(refresh.startswith("gAAAAA"))

    def test_blank_token_is_not_encrypted(self):
        self.credential.access_token = ""
        self.credential.save()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT access_token FROM accounts_googlecredential WHERE id = %s",
                [self.credential.pk],
            )
            self.assertEqual(cursor.fetchone()[0], "")


class CredentialConversionTests(TestCase):
    def setUp(self):
        self.user = User.objects.create(username="tester@example.com")
        self.credential = GoogleCredential.objects.create(
            user=self.user, google_sub="sub-1", refresh_token="1//refresh"
        )

    def test_update_from_credentials_stores_aware_expiry(self):
        naive_expiry = datetime.now(dt_timezone.utc).replace(tzinfo=None) + timedelta(hours=1)
        creds = Credentials(
            token="new-access",
            refresh_token="1//refresh",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="cid",
            client_secret="secret",
            scopes=["openid"],
            expiry=naive_expiry,
        )
        self.credential.update_from_credentials(creds)

        self.credential.refresh_from_db()
        self.assertEqual(self.credential.access_token, "new-access")
        self.assertIsNotNone(self.credential.expires_at.tzinfo)
        self.assertFalse(self.credential.needs_reconnect)

    def test_refresh_response_without_refresh_token_keeps_the_stored_one(self):
        # Google omits refresh_token when it renews an access token.
        creds = Credentials(
            token="new-access",
            refresh_token=None,
            token_uri="https://oauth2.googleapis.com/token",
            client_id="cid",
            client_secret="secret",
        )
        self.credential.update_from_credentials(creds)

        self.credential.refresh_from_db()
        self.assertEqual(self.credential.refresh_token, "1//refresh")

    def test_to_credentials_converts_expiry_back_to_naive_utc(self):
        self.credential.expires_at = datetime(2026, 9, 19, 12, 0, tzinfo=dt_timezone.utc)
        creds = self.credential.to_credentials()
        self.assertIsNone(creds.expiry.tzinfo)
        self.assertEqual(creds.expiry, datetime(2026, 9, 19, 12, 0))

    def test_mark_needs_reconnect_clears_the_access_token(self):
        self.credential.access_token = "ya29.something"
        self.credential.save()
        self.credential.mark_needs_reconnect()

        self.credential.refresh_from_db()
        self.assertTrue(self.credential.needs_reconnect)
        self.assertEqual(self.credential.access_token, "")
        self.assertFalse(self.credential.is_usable)


class SignInFlowTests(TestCase):
    def test_signed_out_landing_offers_google_sign_in(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Sign in with Google")

    def test_login_start_redirects_to_google(self):
        response = self.client.get("/oauth/login/")
        self.assertEqual(response.status_code, 302)
        location = response["Location"]
        self.assertTrue(location.startswith("https://accounts.google.com/"))
        # access_type=offline is what makes Google issue a refresh token at all.
        self.assertIn("access_type=offline", location)
        self.assertIn("prompt=consent", location)

    def test_login_start_requests_all_three_health_scopes(self):
        location = self.client.get("/oauth/login/")["Location"]
        for fragment in (
            "googlehealth.activity_and_fitness.readonly",
            "googlehealth.health_metrics_and_measurements.readonly",
            "googlehealth.sleep.readonly",
        ):
            self.assertIn(fragment, location)

    def test_callback_rejects_a_mismatched_state(self):
        response = self.client.get("/oauth/callback/?code=abc&state=not-the-one")
        self.assertEqual(response.status_code, 400)

    def test_callback_without_a_stored_verifier_fails_cleanly(self):
        session = self.client.session
        session[SESSION_STATE_KEY] = "some-state"
        # Deliberately no SESSION_VERIFIER_KEY.
        session.save()

        response = self.client.get("/oauth/callback/?code=abc&state=some-state")
        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "sign-in session expired", status_code=400)

    def test_callback_surfaces_a_denied_consent(self):
        response = self.client.get("/oauth/callback/?error=access_denied")
        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "access_denied", status_code=400)


class PkceTests(TestCase):
    """Regression cover for 'invalid_grant: Missing code verifier'.

    google-auth-oauthlib does PKCE by default. authorization_url() generates the
    code_verifier on the Flow object and sends its S256 challenge; fetch_token()
    must replay that same verifier. The callback runs on a different request with
    a different Flow, so the verifier has to travel through the session.
    """

    @staticmethod
    def _challenge_for(verifier: str) -> str:
        digest = hashlib.sha256(verifier.encode()).digest()
        return urlsafe_b64encode(digest).decode().split("=")[0]

    def test_authorization_request_uses_pkce(self):
        response = self.client.get("/oauth/login/")
        params = parse_qs(urlparse(response["Location"]).query)
        self.assertIn("code_challenge", params)
        self.assertEqual(params["code_challenge_method"], ["S256"])

    def test_code_verifier_is_persisted_in_the_session(self):
        self.client.get("/oauth/login/")
        verifier = self.client.session.get(SESSION_VERIFIER_KEY)
        self.assertIsNotNone(verifier)
        # PKCE requires 43-128 characters.
        self.assertGreaterEqual(len(verifier), 43)
        self.assertLessEqual(len(verifier), 128)

    def test_stored_verifier_matches_the_challenge_sent_to_google(self):
        response = self.client.get("/oauth/login/")
        params = parse_qs(urlparse(response["Location"]).query)

        verifier = self.client.session[SESSION_VERIFIER_KEY]
        self.assertEqual(self._challenge_for(verifier), params["code_challenge"][0])

    def test_callback_flow_is_rebuilt_with_the_stored_verifier(self):
        from .oauth import build_flow

        self.client.get("/oauth/login/")
        session = self.client.session
        flow = build_flow(
            state=session[SESSION_STATE_KEY],
            code_verifier=session[SESSION_VERIFIER_KEY],
        )
        self.assertEqual(flow.code_verifier, session[SESSION_VERIFIER_KEY])

    def test_state_and_verifier_are_consumed_once(self):
        self.client.get("/oauth/login/")
        self.client.get("/oauth/callback/?code=abc&state=wrong")

        session = self.client.session
        self.assertNotIn(SESSION_STATE_KEY, session)
        self.assertNotIn(SESSION_VERIFIER_KEY, session)
