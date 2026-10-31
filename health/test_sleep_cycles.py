"""Cycle length measured from stage transitions, and the bedtimes it feeds."""

from datetime import time
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase
from django.urls import reverse

from accounts.models import GoogleCredential
from health import sleep_cycles

User = get_user_model()


def segment(start, end, stage):
    return {
        "type": stage,
        "startTime": f"2026-09-19T{start}:00Z",
        "endTime": f"2026-09-19T{end}:00Z",
    }


def night(stages):
    return {
        "sleep": {
            "type": "STAGES",
            "interval": {
                "startTime": "2026-09-19T00:00:00Z",
                "endTime": "2026-09-19T08:00:00Z",
            },
            "stages": stages,
        }
    }


#: REM onsets at 01:30, 03:00 and 04:40: cycles of 90 and 100 minutes.
THREE_CYCLES = night(
    [
        segment("00:00", "01:30", "LIGHT"),
        segment("01:30", "01:50", "REM"),
        segment("01:50", "03:00", "DEEP"),
        segment("03:00", "03:25", "REM"),
        segment("03:25", "04:40", "LIGHT"),
        segment("04:40", "05:10", "REM"),
    ]
)


class NightCycleTests(TestCase):
    def test_measures_rem_onset_to_rem_onset(self):
        result = sleep_cycles.night_cycles(THREE_CYCLES)
        self.assertEqual(result["cycles"], [90.0, 100.0])
        self.assertEqual(result["rem_periods"], 3)

    def test_a_night_without_stages_contributes_nothing(self):
        result = sleep_cycles.night_cycles({"sleep": {"type": "CLASSIC"}})
        self.assertEqual(result["cycles"], [])
        self.assertFalse(result["has_stages"])

    def test_fragmented_rem_is_one_period_not_several(self):
        """Two REM bouts four minutes apart are one REM period, not two cycles."""
        fragmented = night(
            [
                segment("00:00", "01:30", "LIGHT"),
                segment("01:30", "01:40", "REM"),
                segment("01:40", "01:44", "LIGHT"),
                segment("01:44", "02:00", "REM"),
                segment("02:00", "03:10", "DEEP"),
                segment("03:10", "03:40", "REM"),
            ]
        )
        result = sleep_cycles.night_cycles(fragmented)
        self.assertEqual(result["rem_periods"], 2)
        self.assertEqual(result["cycles"], [100.0])

    def test_a_very_short_rem_bout_is_treated_as_noise(self):
        noisy = night(
            [
                segment("00:00", "01:30", "LIGHT"),
                segment("01:30", "01:32", "REM"),
                segment("01:32", "03:00", "DEEP"),
                segment("03:00", "03:30", "REM"),
            ]
        )
        self.assertEqual(sleep_cycles.night_cycles(noisy)["rem_periods"], 1)

    def test_an_implausible_interval_is_dropped_and_counted(self):
        implausible = night(
            [
                segment("00:00", "00:30", "REM"),
                segment("00:30", "05:00", "LIGHT"),
                segment("05:00", "05:30", "REM"),
            ]
        )
        result = sleep_cycles.night_cycles(implausible)
        self.assertEqual(result["cycles"], [])
        self.assertEqual(result["dropped"], 1)


#: REM onsets at 00:30 and 05:00: a 270-minute gap, which is not a cycle but
#: two or three with the REM between them missed.
MISSED_REM = night(
    [
        segment("00:00", "00:30", "REM"),
        segment("00:30", "05:00", "LIGHT"),
        segment("05:00", "05:30", "REM"),
    ]
)


class DiscardTests(TestCase):
    def test_short_and_long_discards_are_counted_separately(self):
        """They mean different things, so one counter cannot serve both."""
        result = sleep_cycles.night_cycles(MISSED_REM)
        self.assertEqual(result["dropped_long"], 1)
        self.assertEqual(result["dropped_short"], 0)

        # Two REM periods 30 minutes apart onset to onset: far enough apart to
        # escape the merge window, too close together to be a cycle.
        crowded = night(
            [
                segment("00:00", "00:10", "REM"),
                segment("00:10", "00:30", "LIGHT"),
                segment("00:30", "00:50", "REM"),
            ]
        )
        result = sleep_cycles.night_cycles(crowded)
        self.assertEqual(result["rem_periods"], 2)
        self.assertEqual(result["dropped_short"], 1)
        self.assertEqual(result["dropped_long"], 0)

    def test_an_overlong_interval_is_evidence_rem_was_missed(self):
        """The count is surfaced because it bounds nothing but points at bias."""
        result = sleep_cycles.estimate([THREE_CYCLES] * 6 + [MISSED_REM])
        self.assertEqual(result.dropped_long, 1)
        self.assertTrue(result.reliable)

    def test_first_and_later_cycles_are_compared_not_assumed(self):
        result = sleep_cycles.estimate([THREE_CYCLES] * 6)
        # THREE_CYCLES is 90 then 100, so the first is the shorter one.
        self.assertEqual(result.first_cycle_minutes, 90)
        self.assertEqual(result.later_cycle_minutes, 100)


class EstimateTests(TestCase):
    def test_one_night_is_reported_but_not_called_reliable(self):
        result = sleep_cycles.estimate([THREE_CYCLES])
        self.assertFalse(result.reliable)
        self.assertEqual(result.cycles_measured, 2)
        # The median is still shown rather than hidden.
        self.assertEqual(result.cycle_minutes, 95)
        self.assertIn("below the", result.reason)

    def test_enough_nights_clears_the_gate(self):
        result = sleep_cycles.estimate([THREE_CYCLES] * 6)
        self.assertTrue(result.reliable)
        self.assertEqual(result.nights_contributing, 6)
        self.assertEqual(result.cycles_measured, 12)
        self.assertEqual(result.cycle_minutes, 95)

    def test_no_usable_night_says_why(self):
        result = sleep_cycles.estimate([{"sleep": {"type": "CLASSIC"}}])
        self.assertFalse(result.reliable)
        self.assertIsNone(result.cycle_minutes)
        self.assertIn("two separate REM periods", result.reason)

    def test_spread_is_reported_alongside_the_median(self):
        result = sleep_cycles.estimate([THREE_CYCLES] * 6)
        self.assertIsNotNone(result.spread_minutes)
        self.assertEqual(result.shortest, 90.0)
        self.assertEqual(result.longest, 100.0)


class BedtimeTests(TestCase):
    def test_counts_back_whole_cycles_plus_time_to_fall_asleep(self):
        options = sleep_cycles.bedtimes(time(7, 0), 90, counts=(5,))
        # 5 * 90 = 450 minutes, plus 15 to fall asleep, back from 07:00.
        self.assertEqual(options[0]["bedtime"], "23:15")
        self.assertEqual(options[0]["asleep_label"], "7h 30m")

    def test_more_cycles_means_an_earlier_bedtime(self):
        options = sleep_cycles.bedtimes(time(7, 0), 90, counts=(4, 5, 6))
        clocks = [option["bedtime"] for option in options]
        self.assertEqual(clocks, ["00:45", "23:15", "21:45"])

    def test_option_counts_follow_the_cycle_length(self):
        """A fixed (4, 5, 6) offers eleven hours in bed at a 110-minute cycle."""
        self.assertEqual(sleep_cycles.cycle_counts(90), (4, 5, 6))
        self.assertEqual(sleep_cycles.cycle_counts(110), (3, 4, 5))
        for length in (80, 90, 100, 110, 120, 140):
            for option in sleep_cycles.bedtimes(time(7, 0), length):
                self.assertGreaterEqual(option["asleep_minutes"], sleep_cycles.MIN_SENSIBLE_SLEEP)
                self.assertLessEqual(option["asleep_minutes"], sleep_cycles.MAX_SENSIBLE_SLEEP)

    def test_at_the_textbook_length_the_options_are_unchanged(self):
        clocks = [o["bedtime"] for o in sleep_cycles.bedtimes(time(7, 0), 90)]
        self.assertEqual(clocks, ["00:45", "23:15", "21:45"])

    def test_a_measured_cycle_length_changes_the_answer(self):
        standard = sleep_cycles.bedtimes(time(7, 0), 90, counts=(5,))[0]
        measured = sleep_cycles.bedtimes(time(7, 0), 100, counts=(5,))[0]
        self.assertNotEqual(standard["bedtime"], measured["bedtime"])
