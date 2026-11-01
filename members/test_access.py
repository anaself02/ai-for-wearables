"""Who may open what: anonymous, owner, unlinked, pending, and staff.

Two profiles with data (Alpha, Beta) and the seeded Anas and Rahil. Every account
is synthetic. Heart rate is a constant marker (77 for Alpha, 133 for Beta) so a
page can be searched for the other profile's number.
"""

import re
from datetime import date

from django.core.cache import cache
from django.test import Client, TestCase
from django.urls import reverse

from people import links
from people.models import HeartRateDay, Person
from people.test_views import make_person
from people.testing import make_user


def person_urls(slug):
    day = date(2026, 7, 15)
    return {
        "overview": reverse("people:person", args=[slug]),
        "heart rate": reverse("people:heart_rate", args=[slug]),
        "a day": reverse("people:heart_rate_day", args=[slug, day]),
        "raw": reverse("people:raw_index", args=[slug]),
        "raw day": reverse("people:raw_day", args=[slug, day]),
        "arousal": reverse("people:arousal", args=[slug]),
        "arousal how": reverse("people:arousal_methodology", args=[slug]),
        "upload": reverse("people:upload", args=[slug]),
    }


def post_urls(slug):
    return [reverse("people:upload", args=[slug]),
            reverse("people:upload_confirm", args=[slug]),
            reverse("people:upload_cancel", args=[slug])]


class AccessCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.alpha = make_person("alpha", "Alpha", "Fitbit Air",
                                [date(2026, 7, 15), date(2026, 7, 16)], 77)
        cls.beta = make_person("beta", "Beta", "Fitbit Charge",
                               [date(2026, 8, 1), date(2026, 8, 2)], 133)

    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)
        self.user_a, _ = make_user("user.a@example.com", linked_to=self.alpha)
        self.user_b, _ = make_user("user.b@example.com", linked_to=self.beta)
        self.staff, _ = make_user("admin.person@example.com", staff=True)

    def as_user(self, user):
        client = Client()
        client.force_login(user)
        return client


class AnonymousTests(AccessCase):
    def test_everything_but_signup_and_login_redirects_to_login(self):
        urls = ["/", "/waiting/", "/admin/", "/admin/login/"]
        urls += list(person_urls("alpha").values()) + list(person_urls("anas").values())
        for url in urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 302, url)
            self.assertTrue(response["Location"].startswith("/login/"), url)

    def test_the_login_redirect_remembers_where_you_were_going(self):
        response = self.client.get("/p/alpha/heart-rate/")
        self.assertEqual(response["Location"], "/login/?next=%2Fp%2Falpha%2Fheart-rate%2F")

    def test_posts_are_redirected_too_and_change_nothing(self):
        before = HeartRateDay.objects.count()
        for url in post_urls("alpha") + ["/claim/", "/logout/"]:
            response = self.client.post(url)
            self.assertEqual(response.status_code, 302, url)
            self.assertTrue(response["Location"].startswith("/login/"), url)
        self.assertEqual(HeartRateDay.objects.count(), before)

    def test_signup_and_login_stay_open(self):
        self.assertEqual(self.client.get("/signup/").status_code, 200)
        self.assertEqual(self.client.get("/login/").status_code, 200)

    def test_static_files_are_not_redirected(self):
        self.assertNotEqual(self.client.get("/static/app.css").status_code, 302)

    def test_an_anonymous_page_shows_no_data_and_no_emails(self):
        html = self.client.get("/login/").content.decode()
        for secret in ("77", "133", "Alpha", "Beta", "example.com"):
            self.assertNotIn(secret, html)


class OwnerTests(AccessCase):
    def test_home_sends_an_owner_to_their_own_profile(self):
        response = self.as_user(self.user_a).get("/")
        self.assertRedirects(response, "/p/alpha/", fetch_redirect_response=False)
        response = self.as_user(self.user_b).get("/")
        self.assertRedirects(response, "/p/beta/", fetch_redirect_response=False)

    def test_an_owner_opens_every_page_of_their_own_profile(self):
        client = self.as_user(self.user_a)
        for name, url in person_urls("alpha").items():
            self.assertEqual(client.get(url).status_code, 200, name)

    def test_user_a_cannot_read_user_bs_profile_by_url_arousal_or_upload(self):
        client = self.as_user(self.user_a)
        for name, url in person_urls("beta").items():
            self.assertEqual(client.get(url).status_code, 404, name)
        for url in post_urls("beta"):
            self.assertEqual(client.post(url, {"token": "x" * 24}).status_code, 404, url)

    def test_user_b_cannot_read_user_as_profile_either(self):
        client = self.as_user(self.user_b)
        for name, url in person_urls("alpha").items():
            self.assertEqual(client.get(url).status_code, 404, name)
        for url in post_urls("alpha"):
            self.assertEqual(client.post(url).status_code, 404, url)

    def test_the_other_profiles_refusal_looks_like_a_missing_page(self):
        client = self.as_user(self.user_a)
        refused = client.get("/p/beta/").status_code
        missing = client.get("/p/does-not-exist/").status_code
        self.assertEqual((refused, missing), (404, 404))

    def test_an_owner_never_sees_the_chooser_or_the_other_profiles_data(self):
        client = self.as_user(self.user_a)
        for url in list(person_urls("alpha").values()) + ["/p/alpha/"]:
            html = client.get(url, follow=True).content.decode()
            main = re.search(r"<main.*?</main>", html, re.S).group(0)
            self.assertNotIn("Beta", html)
            self.assertNotIn("Fitbit Charge", html)
            self.assertNotRegex(main, r"\b133(\.0)?\b")
            self.assertNotIn("Whose data?", html)

    def test_the_bar_has_no_switch_link_for_an_owner_but_has_log_out(self):
        html = self.as_user(self.user_a).get("/p/alpha/").content.decode()
        bar = re.search(r'<header class="topbar">.*?</header>', html, re.S).group(0)
        self.assertNotIn(">Switch<", bar)
        self.assertIn(">Log out<", bar)
        self.assertIn("Alpha", bar)
        self.assertIn("Fitbit Air", bar)

    def test_an_owner_cannot_open_admin_or_unknown_pages(self):
        client = self.as_user(self.user_a)
        for url in ("/admin/", "/admin/people/person/", "/nowhere/", "/signup/x"):
            self.assertEqual(client.get(url).status_code, 404, url)

    def test_an_owner_cannot_confirm_another_profiles_parked_upload(self):
        response = self.as_user(self.user_a).post(
            reverse("people:upload_confirm", args=["beta"]), {"token": "a" * 24})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(HeartRateDay.objects.filter(person=self.beta).count(), 2)

    def test_no_account_ever_sees_another_accounts_email(self):
        emails = ["user.a@example.com", "user.b@example.com", "admin.person@example.com"]
        for user, urls in ((self.user_a, person_urls("alpha")),
                           (self.user_b, person_urls("beta"))):
            client = self.as_user(user)
            pages = list(urls.values()) + ["/waiting/", "/login/", "/signup/"]
            for url in pages:
                html = client.get(url, follow=True).content.decode()
                for email in emails:
                    if email != user.email:
                        self.assertNotIn(email, html, (user.email, url))


class UnlinkedAndPendingTests(AccessCase):
    def setUp(self):
        super().setUp()
        self.newcomer, _ = make_user("newcomer@example.com")
        self.client = self.as_user(self.newcomer)

    def test_an_unlinked_account_is_sent_to_the_waiting_page_and_sees_no_data(self):
        for url in ["/", "/p/alpha/", "/p/anas/", "/p/beta/heart-rate/"]:
            response = self.client.get(url)
            self.assertRedirects(response, "/waiting/", fetch_redirect_response=False)
        html = self.client.get("/waiting/").content.decode()
        main = re.search(r"<main.*?</main>", html, re.S).group(0)
        self.assertIn("not linked to a profile yet", html)
        self.assertNotRegex(main, r"(77|133)(\.0)?")
        for secret in ("Fitbit Air", "Fitbit Charge", "<canvas", "<table"):
            self.assertNotIn(secret, html)

    def test_other_routes_are_a_404_for_an_unlinked_account(self):
        self.assertEqual(self.client.get("/admin/").status_code, 404)
        self.assertEqual(self.client.get("/nowhere/").status_code, 404)

    def test_a_pending_claim_sees_no_data_until_it_is_approved(self):
        free = Person.objects.get(slug="anas")
        claim = links.create_claim(self.newcomer, free)
        response = self.client.get("/p/anas/")
        self.assertRedirects(response, "/waiting/", fetch_redirect_response=False)
        self.assertIn("waiting for an administrator",
                      self.client.get("/waiting/").content.decode())

        links.approve_claim(claim)
        self.assertRedirects(self.client.get("/"), "/p/anas/", fetch_redirect_response=False)
        self.assertEqual(self.client.get("/p/anas/").status_code, 200)
        self.assertEqual(self.client.get("/p/alpha/").status_code, 404)

    def test_a_rejected_claim_still_sees_no_data(self):
        claim = links.create_claim(self.newcomer, Person.objects.get(slug="anas"))
        links.reject_claim(claim)
        self.assertRedirects(self.client.get("/p/anas/"), "/waiting/",
                             fetch_redirect_response=False)

    def test_unlinking_takes_the_access_away_again(self):
        anas = Person.objects.get(slug="anas")
        links.approve_claim(links.create_claim(self.newcomer, anas))
        self.assertEqual(self.client.get("/p/anas/").status_code, 200)
        links.unlink_person(anas)
        self.assertRedirects(self.client.get("/p/anas/"), "/waiting/",
                             fetch_redirect_response=False)


class StaffTests(AccessCase):
    def test_staff_see_the_chooser_with_both_seeded_people(self):
        response = self.as_user(self.staff).get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Anas")
        self.assertContains(response, "Rahil")
        self.assertContains(response, "Whose data?")

    def test_staff_can_open_any_profile_and_every_page(self):
        client = self.as_user(self.staff)
        for slug in ("alpha", "beta", "anas", "rahil"):
            for name, url in person_urls(slug).items():
                self.assertEqual(client.get(url).status_code, 200, (slug, name))

    def test_staff_keep_the_switch_link(self):
        html = self.as_user(self.staff).get("/p/alpha/").content.decode()
        bar = re.search(r'<header class="topbar">.*?</header>', html, re.S).group(0)
        self.assertIn(">Switch<", bar)

    def test_staff_can_reach_the_admin(self):
        self.assertEqual(self.as_user(self.staff).get("/admin/").status_code, 200)

    def test_each_profile_page_still_shows_only_that_profiles_data(self):
        client = self.as_user(self.staff)
        html = client.get("/p/alpha/").content.decode()
        self.assertIn("77.0", html)
        self.assertNotIn("133", html)
        self.assertNotIn("Beta", html)
