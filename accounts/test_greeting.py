"""The greeting, its boundaries, and the screen that shows it.

The boundaries are pinned to the minute on both sides. They are the kind of
thing that gets "tidied" into a different comparison operator by somebody
refactoring in good faith, and the only way anyone would notice is by signing
in at 11:59 and being told good afternoon.
"""

from datetime import datetime

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from accounts.greeting import display_name, greeting_for

User = get_user_model()


def at(hour, minute=0):
    return datetime(2026, 9, 20, hour, minute)


class GreetingBoundaryTests(TestCase):
    def test_morning_starts_at_five(self):
        self.assertEqual(greeting_for(at(4, 59)), "Good evening")
        self.assertEqual(greeting_for(at(5, 0)), "Good morning")

    def test_afternoon_starts_at_noon(self):
        self.assertEqual(greeting_for(at(11, 59)), "Good morning")
        self.assertEqual(greeting_for(at(12, 0)), "Good afternoon")

    def test_evening_starts_at_six(self):
        self.assertEqual(greeting_for(at(17, 59)), "Good afternoon")
        self.assertEqual(greeting_for(at(18, 0)), "Good evening")

    def test_the_small_hours_are_still_evening(self):
        """Awake at 02:00 is the end of an evening, not the start of a day."""
        self.assertEqual(greeting_for(at(0, 0)), "Good evening")
        self.assertEqual(greeting_for(at(2, 30)), "Good evening")
        self.assertEqual(greeting_for(at(23, 59)), "Good evening")

    def test_every_hour_of_the_day_gets_a_greeting(self):
        for hour in range(24):
            self.assertIn(
                greeting_for(at(hour)),
                {"Good morning", "Good afternoon", "Good evening"},
                f"hour {hour}",
            )


class DisplayNameTests(TestCase):
    def _user(self, first_name):
        user = User(username="x@example.com")
        user.first_name = first_name
        return user

    def test_a_shouted_name_is_calmed_down(self):
        """Google stores what was typed, which is very often all capitals."""
        self.assertEqual(display_name(self._user("ANAS")), "Anas")

    def test_an_all_lowercase_name_is_capitalised(self):
        self.assertEqual(display_name(self._user("anas")), "Anas")

    def test_a_name_that_carries_its_own_case_is_left_alone(self):
        self.assertEqual(display_name(self._user("McDonald")), "McDonald")
        self.assertEqual(display_name(self._user("van Dijk")), "van Dijk")

    def test_surrounding_whitespace_is_dropped(self):
        self.assertEqual(display_name(self._user("  ANAS  ")), "Anas")

    def test_no_name_is_an_empty_string_not_a_guess(self):
        self.assertEqual(display_name(self._user("")), "")
        self.assertEqual(display_name(User(username="x@example.com")), "")
