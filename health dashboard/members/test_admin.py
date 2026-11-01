"""The admin: who is linked to which profile, approve, reject, unlink.

Never shows password hashes or heart-rate data. All accounts are synthetic.
"""

import re
from datetime import date

from django.contrib.auth import get_user_model
from django.contrib.messages import get_messages
from django.core.cache import cache
from django.test import Client, TestCase
from django.urls import reverse

from people import links
from people.models import HeartRateDay, Person, ProfileClaim
from people.test_views import make_person
from people.testing import make_user

User = get_user_model()


class AdminCase(TestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)
        self.admin_user, _ = make_user("boss@example.com", superuser=True)
        self.client = Client()
        self.client.force_login(self.admin_user)
        self.anas = Person.objects.get(slug="anas")
        self.rahil = Person.objects.get(slug="rahil")

    def run_action(self, url_name, action, ids, **kw):
        return self.client.post(reverse(url_name), {
            "action": action, "_selected_action": [str(i) for i in ids]}, follow=True, **kw)

    def messages_of(self, response):
        return [str(m) for m in response.context["messages"]]


class PersonListTests(AdminCase):
    def test_lists_name_device_status_readings_and_linked_account(self):
        make_person("alpha", "Alpha", "Fitbit Air", [date(2026, 7, 15)], 77)
        make_user("owner@example.com", linked_to=Person.objects.get(slug="alpha"))
        html = self.client.get(reverse("admin:people_person_changelist")).content.decode()
        for expected in ("Anas", "Fitbit Air", "Rahil", "EmotiBit", "awaiting data",
                         "data loaded", "28,800" if False else "28800", "owner@example.com",
                         "not linked"):
            self.assertIn(expected, html, expected)

    def test_a_profiles_link_cannot_be_edited_directly(self):
        html = self.client.get(
            reverse("admin:people_person_change", args=[self.anas.pk])).content.decode()
        self.assertNotIn('name="owner"', html)
        self.assertIn("not linked", html)

    def test_a_profile_cannot_be_deleted_from_the_admin(self):
        response = self.client.get(reverse("admin:people_person_delete", args=[self.anas.pk]))
        self.assertEqual(response.status_code, 403)


class AccountListTests(AdminCase):
    def test_lists_email_linked_profile_pending_claim_active_and_last_login(self):
        linked, _ = make_user("linked@example.com", linked_to=self.anas)
        waiting, _ = make_user("waiting@example.com")
        links.create_claim(waiting, self.rahil)
        html = self.client.get(reverse("admin:auth_user_changelist")).content.decode()
        row = lambda email: re.search(r"<tr[^>]*>(?:(?!</tr>).)*%s.*?</tr>" % re.escape(email),
                                      html, re.S).group(0)
        self.assertIn("Anas", row("linked@example.com"))
        self.assertIn("not linked", row("waiting@example.com"))
        self.assertIn("Rahil", row("waiting@example.com"))        # the pending claim
        self.assertIn("none", row("linked@example.com"))

    def test_the_pending_claims_filter(self):
        linked, _ = make_user("linked@example.com", linked_to=self.anas)
        waiting, _ = make_user("waiting@example.com")
        links.create_claim(waiting, self.rahil)
        url = reverse("admin:auth_user_changelist")
        yes = self.client.get(url + "?pending=yes").content.decode()
        self.assertIn("waiting@example.com", yes)
        self.assertNotIn("linked@example.com", yes)
        no = self.client.get(url + "?pending=no").content.decode()
        self.assertIn("linked@example.com", no)
        self.assertNotIn("waiting@example.com", no)

    def test_no_password_hash_is_ever_shown(self):
        user, _ = make_user("hash@example.com")
        for url in (reverse("admin:auth_user_changelist"),
                    reverse("admin:auth_user_change", args=[user.pk]),
                    reverse("admin:index")):
            html = self.client.get(url).content.decode()
            self.assertNotIn(user.password, html)
            self.assertNotIn(user.password.split("$")[0] + "$", html)
            self.assertNotIn('name="password"', html)
            self.assertNotIn("algorithm", html.lower().replace("algorithms", ""))

    def test_accounts_cannot_be_created_or_given_passwords_from_the_admin(self):
        response = self.client.get(reverse("admin:auth_user_add"))
        self.assertEqual(response.status_code, 403)
        user, _ = make_user("x@example.com")
        html = self.client.get(reverse("admin:auth_user_change", args=[user.pk])).content.decode()
        self.assertNotIn("id_password", html)                       # no password field
        self.assertNotIn("/password/", html.replace("/admin/password_change/", ""))
        self.assertNotIn("Raw passwords are not stored", html)

    def test_the_stock_groups_page_is_gone(self):
        self.assertEqual(self.client.get("/admin/auth/group/").status_code, 404)

    def test_heart_rate_data_is_not_in_the_admin(self):
        make_person("alpha", "Alpha", "Fitbit Air", [date(2026, 7, 15)], 77)
        index = self.client.get(reverse("admin:index")).content.decode()
        self.assertNotIn("Heart rate days", index)
        self.assertNotIn("Recordings", index)
        self.assertEqual(self.client.get("/admin/people/heartrateday/").status_code, 404)
        self.assertEqual(self.client.get("/admin/people/recording/").status_code, 404)
        alpha = Person.objects.get(slug="alpha")
        html = self.client.get(reverse("admin:people_person_change", args=[alpha.pk])).content.decode()
        self.assertNotRegex(re.sub(r"<[^>]+>", " ", html), r"\b77(\.0)?\b")
        self.assertTrue(HeartRateDay.objects.filter(person=alpha).exists())


class ApproveRejectUnlinkTests(AdminCase):
    def setUp(self):
        super().setUp()
        self.user, _ = make_user("claimant@example.com")
        self.claim = links.create_claim(self.user, self.anas)

    def test_approving_from_the_account_list_links_the_profile(self):
        response = self.run_action("admin:auth_user_changelist", "approve_claims", [self.user.pk])
        self.anas.refresh_from_db()
        self.assertEqual(self.anas.owner, self.user)
        self.assertIn("Linked Anas.", self.messages_of(response))
        self.assertNotIn("claimant@example.com", " ".join(self.messages_of(response)))

    def test_approving_from_the_claims_list_works_too(self):
        self.run_action("admin:people_profileclaim_changelist", "approve", [self.claim.pk])
        self.anas.refresh_from_db()
        self.assertEqual(self.anas.owner, self.user)

    def test_approving_is_refused_when_the_profile_is_already_linked(self):
        make_user("first@example.com", linked_to=self.anas)
        response = self.run_action("admin:auth_user_changelist", "approve_claims", [self.user.pk])
        self.anas.refresh_from_db()
        self.assertEqual(self.anas.owner.email, "first@example.com")      # not replaced
        self.assertTrue(any("already linked" in m for m in self.messages_of(response)))
        self.claim.refresh_from_db()
        self.assertEqual(self.claim.status, "pending")

    def test_approving_an_account_that_has_no_pending_claim_says_so(self):
        other, _ = make_user("noclaim@example.com")
        response = self.run_action("admin:auth_user_changelist", "approve_claims", [other.pk])
        self.assertTrue(any("no pending claim" in m for m in self.messages_of(response)))
        self.assertFalse(Person.objects.filter(owner=other).exists())

    def test_rejecting_marks_the_claim_and_links_nothing(self):
        self.run_action("admin:auth_user_changelist", "reject_claims", [self.user.pk])
        self.claim.refresh_from_db()
        self.anas.refresh_from_db()
        self.assertEqual(self.claim.status, "rejected")
        self.assertIsNone(self.anas.owner)

    def test_unlinking_from_the_account_list(self):
        links.approve_claim(self.claim)
        response = self.run_action("admin:auth_user_changelist", "unlink_accounts", [self.user.pk])
        self.anas.refresh_from_db()
        self.assertIsNone(self.anas.owner)
        self.assertIn("Unlinked Anas.", self.messages_of(response))

    def test_unlinking_from_the_profile_list(self):
        links.approve_claim(self.claim)
        self.run_action("admin:people_person_changelist", "unlink_account", [self.anas.pk])
        self.anas.refresh_from_db()
        self.assertIsNone(self.anas.owner)

    def test_unlinking_an_unlinked_account_is_refused_with_a_message(self):
        response = self.run_action("admin:auth_user_changelist", "unlink_accounts", [self.user.pk])
        self.assertTrue(any("not linked" in m for m in self.messages_of(response)))

    def test_a_bulk_approval_reports_each_outcome_and_keeps_the_good_ones(self):
        second, _ = make_user("second@example.com")
        links.create_claim(second, self.rahil)
        third, _ = make_user("third@example.com")           # no claim
        response = self.run_action("admin:auth_user_changelist", "approve_claims",
                                   [self.user.pk, second.pk, third.pk])
        self.anas.refresh_from_db()
        self.rahil.refresh_from_db()
        self.assertEqual((self.anas.owner, self.rahil.owner), (self.user, second))
        self.assertEqual(len([m for m in self.messages_of(response) if "no pending claim" in m]), 1)

    def test_the_approved_account_then_sees_its_profile(self):
        self.run_action("admin:auth_user_changelist", "approve_claims", [self.user.pk])
        member = Client()
        member.force_login(self.user)
        self.assertRedirects(member.get("/"), "/p/anas/", fetch_redirect_response=False)


class AdminAccessTests(AdminCase):
    def test_a_non_staff_account_gets_a_404_not_a_login_form(self):
        user, _ = make_user("plain@example.com", linked_to=self.anas)
        client = Client()
        client.force_login(user)
        self.assertEqual(client.get("/admin/").status_code, 404)

    def test_an_anonymous_visitor_is_sent_to_the_normal_login(self):
        response = Client().get("/admin/")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response["Location"].startswith("/login/"))
