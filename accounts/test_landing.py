"""The front door, and the third-party mark on it.

The Google "G" is somebody else's trademark, used under branding guidelines
that require it to be the published asset, unmodified. Nothing about that is
enforceable by review alone: a well-meaning tidy-up of "redundant" SVG, a
reformatter, or a search-and-replace on hex codes would all change it silently
and still render something that looks approximately right.

So the published path data and colours are pinned here, character for
character, in the order Google publishes them.
"""

from django.test import TestCase
from django.urls import reverse

#: Verbatim from Google's Sign in with Google branding guidelines.
OFFICIAL_G_PATHS = [
    (
        "#EA4335",
        "M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 "
        "6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z",
    ),
    (
        "#4285F4",
        "M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 "
        "7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z",
    ),
    (
        "#FBBC05",
        "M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 "
        "16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z",
    ),
    (
        "#34A853",
        "M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 "
        "0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z",
    ),
    ("none", "M0 0h48v48H0z"),
]


class LandingPageTests(TestCase):
    def setUp(self):
        self.response = self.client.get(reverse("accounts:landing"))

    def test_a_logged_out_visitor_gets_the_front_door(self):
        self.assertEqual(self.response.status_code, 200)
        self.assertTemplateUsed(self.response, "accounts/landing.html")

    def test_the_front_door_has_no_navigation(self):
        """There is nowhere to go until you are signed in."""
        self.assertNotContains(self.response, 'class="tabs"')
        self.assertNotContains(self.response, "Raw data")

    def test_it_does_not_oversell_itself(self):
        """A personal dashboard, not a product with customers."""
        for word in ("trusted by", "testimonial", "users worldwide", "free trial"):
            self.assertNotContains(self.response, word, status_code=200)


class GoogleButtonTests(TestCase):
    def setUp(self):
        self.response = self.client.get(reverse("accounts:landing"))

    def test_the_button_uses_a_permitted_string(self):
        self.assertContains(self.response, "Sign in with Google")

    def test_the_mark_is_the_published_asset_unmodified(self):
        for colour, path in OFFICIAL_G_PATHS:
            self.assertContains(
                self.response,
                f'<path fill="{colour}" d="{path}"',
                msg_prefix=(
                    f"The {colour} path of the Google mark does not match the "
                    "published asset. It must not be redrawn, simplified or "
                    "recoloured"
                ),
            )

    def test_the_mark_keeps_its_own_coordinate_system(self):
        """A different viewBox means the paths were rescaled by hand."""
        self.assertContains(self.response, 'viewBox="0 0 48 48"')

    def test_the_mark_is_never_monochrome(self):
        """Explicitly prohibited: no single-colour version of the G."""
        self.assertNotContains(self.response, '<path fill="currentColor" d="M24 9.5')
