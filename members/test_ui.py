"""The chooser layout and the navigation, checked from the markup and stylesheet.

Layout is not rendered here (there is no browser in the test run), so these tests
pin what the markup and CSS promise: two cards, in order, each complete, in a grid
that is two columns wide and one column on a narrow screen, drawn only from the
existing design tokens.
"""

import re
from pathlib import Path

from django.test import Client, TestCase

from people.testing import make_user

CSS = (Path(__file__).resolve().parent.parent / "static" / "app.css").read_text(encoding="utf-8")


def block(selector):
    """The declarations of the first rule whose selector is exactly `selector`."""
    match = re.search(r"(?m)^%s\s*\{([^}]*)\}" % re.escape(selector), CSS)
    assert match, selector
    return match.group(1)


def people_css():
    start = CSS.index("/* Two cards side by side")
    end = CSS.index(".person-badge {")
    return CSS[start:end]


class ChooserMarkupTests(TestCase):
    def setUp(self):
        staff, _ = make_user("staff@example.com", staff=True)
        self.client = Client()
        self.client.force_login(staff)
        self.html = self.client.get("/").content.decode()

    def cards(self):
        return re.findall(r'<section class="person-card".*?</section>', self.html, re.S)

    def test_there_are_exactly_two_cards_anas_then_rahil(self):
        cards = self.cards()
        self.assertEqual(len(cards), 2)
        self.assertIn("Anas", cards[0])
        self.assertIn("Rahil", cards[1])

    def test_each_card_is_complete_so_rahils_is_not_a_stub(self):
        for card, name, device in zip(self.cards(), ("Anas", "Rahil"),
                                      ("Fitbit Air", "EmotiBit")):
            self.assertIn(device, card)
            self.assertIn("Status", card)
            self.assertIn("Local date range", card)
            self.assertIn("Readings", card)
            self.assertIn("Open {}".format(name), card)

    def test_rahils_card_carries_the_honest_empty_state_and_no_numbers(self):
        rahil = self.cards()[1]
        self.assertIn("No data uploaded yet. Expected after the mid-project presentation.", rahil)
        self.assertNotRegex(re.sub(r'href="[^"]*"', "", rahil), r"\d{4}-\d{2}-\d{2}")

    def test_both_cards_sit_in_one_grid(self):
        self.assertEqual(self.html.count('<div class="people">'), 1)
        grid = self.html.split('<div class="people">')[1].split("</section>\n  </div>")[0]
        self.assertEqual(grid.count('class="person-card"'), 2)


class ChooserStyleTests(TestCase):
    def test_the_grid_is_two_columns_that_can_shrink(self):
        rule = block(".people")
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", rule)

    def test_it_stacks_on_a_narrow_screen(self):
        match = re.search(r"@media \(max-width: (\d+)px\)\s*\{\s*\.people\s*\{([^}]*)\}", CSS)
        self.assertIsNotNone(match)
        self.assertLessEqual(int(match.group(1)), 720)
        self.assertIn("grid-template-columns: minmax(0, 1fr)", match.group(2))

    def test_a_card_can_shrink_and_wraps_text_instead_of_being_cut_off(self):
        rule = block(".person-card")
        self.assertIn("min-width: 0", rule)
        self.assertIn("overflow-wrap: anywhere", rule)
        self.assertNotIn("overflow: hidden", rule)
        self.assertNotIn("nowrap", block(".device-chip"))

    def test_only_existing_design_tokens_are_used_no_new_colours(self):
        css = people_css()
        self.assertNotRegex(css, r"#[0-9a-fA-F]{3,8}\b")
        self.assertNotRegex(css, r"\b(rgb|rgba|hsl|hsla)\(")
        for token in set(re.findall(r"var\((--[\w-]+)\)", css)):
            self.assertRegex(CSS, r"(?m)^\s*%s\s*:" % re.escape(token), token)


class NavigationTests(TestCase):
    def test_import_data_is_in_the_person_navigation(self):
        staff, _ = make_user("staff@example.com", staff=True)
        client = Client()
        client.force_login(staff)
        for slug in ("anas", "rahil"):
            html = client.get("/p/{}/".format(slug)).content.decode()
            bar = re.search(r'<header class="topbar">.*?</header>', html, re.S).group(0)
            self.assertIn(">Import data<", bar)
            self.assertIn("/p/{}/import/".format(slug), bar)
            for tab in ("Overview", "Arousal", "Raw rows"):
                self.assertIn(">{}<".format(tab), bar)


def topbar(html):
    return re.search(r'<header class="topbar">.*?</header>', html, re.S).group(0)


class AdminLinkTests(TestCase):
    """The Admin button is in the markup for staff only, not merely hidden."""

    def test_staff_see_admin_in_the_bar_on_the_chooser_and_a_profile(self):
        staff, _ = make_user("staff@example.com", staff=True)
        client = Client()
        client.force_login(staff)
        for url in ("/", "/p/anas/"):
            bar = topbar(client.get(url).content.decode())
            self.assertIn('href="/admin/"', bar, url)
            self.assertIn(">Admin<", bar, url)

    def test_an_owner_has_no_admin_link(self):
        from people.models import Person
        owner, _ = make_user("owner@example.com", linked_to=Person.objects.get(slug="anas"))
        client = Client()
        client.force_login(owner)
        html = client.get("/p/anas/").content.decode()
        self.assertNotIn("/admin/", html)
        self.assertNotIn(">Admin<", html)

    def test_anonymous_pages_have_no_admin_link(self):
        for url in ("/login/", "/signup/"):
            html = Client().get(url).content.decode()
            self.assertNotIn("/admin/", html, url)
            self.assertNotIn(">Admin<", html, url)


class LoginPageTests(TestCase):
    def test_wrong_password_and_unknown_email_get_one_generic_message(self):
        make_user("someone@example.com")
        wrong = self.client.post("/login/", {"email": "someone@example.com",
                                             "password": "not-the-password"})
        unknown = self.client.post("/login/", {"email": "nobody@example.com",
                                               "password": "not-the-password"})
        for response in (wrong, unknown):
            html = response.content.decode()
            self.assertIn('role="alert">Email or password is incorrect.<', html)
            self.assertIn('aria-invalid="true"', html)
            self.assertNotIn("nobody@example.com is", html)

    def test_the_password_toggle_is_a_button_with_an_accessible_name(self):
        html = self.client.get("/login/").content.decode()
        button = re.search(r"<button[^>]*data-reveal=\"id_password\"[^>]*>", html, re.S).group(0)
        self.assertIn('type="button"', button)
        self.assertIn('aria-label="Show password"', button)
        self.assertIn('aria-pressed="false"', button)
