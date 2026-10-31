"""Tests for the Person / Recording / HeartRateDay models and the seed."""

from datetime import date, datetime, timezone

import numpy as np
from django.db import IntegrityError, transaction
from django.test import TestCase

from people.models import HeartRateDay, Person, Recording


def make_day(person, utc_date, epochs, bpm):
    offsets, packed = HeartRateDay.pack(epochs, bpm)
    return HeartRateDay.objects.create(
        person=person, utc_date=utc_date, n=len(epochs),
        first_utc=datetime.fromtimestamp(int(epochs[0]), timezone.utc),
        last_utc=datetime.fromtimestamp(int(epochs[-1]), timezone.utc),
        bpm_min=float(np.min(bpm)), bpm_max=float(np.max(bpm)),
        bpm_mean=float(np.mean(bpm)), offsets=offsets, bpm_x10=packed)


class SeedTests(TestCase):
    def test_the_two_participants_exist_with_their_devices(self):
        anas = Person.objects.get(slug="anas")
        rahil = Person.objects.get(slug="rahil")
        self.assertEqual((anas.display_name, anas.device_label, anas.adapter_key),
                         ("Anas", "Fitbit Air", "fitbit-takeout"))
        self.assertEqual((rahil.display_name, rahil.device_label, rahil.adapter_key),
                         ("Rahil", "EmotiBit", "emotibit"))
        self.assertEqual(anas.timezone, "America/Toronto")

    def test_nobody_has_data_until_something_is_loaded(self):
        for person in Person.objects.all():
            self.assertFalse(person.has_data)
            self.assertEqual(person.status, "awaiting data")
            self.assertIsNone(person.utc_range())
            self.assertIsNone(person.local_range())

    def test_exactly_two_people_are_seeded(self):
        self.assertEqual(Person.objects.count(), 2)


class PackingTests(TestCase):
    def setUp(self):
        self.person = Person.objects.get(slug="anas")
        self.midnight = int(datetime(2026, 7, 16, tzinfo=timezone.utc).timestamp())

    def test_round_trip_preserves_times_and_values(self):
        epochs = self.midnight + np.array([0, 3, 6, 86399])
        bpm = np.array([61.0, 62.0, 71.5, 190.0])
        day = make_day(self.person, date(2026, 7, 16), epochs, bpm)
        day.refresh_from_db()
        got_epochs, got_bpm = day.unpack()
        self.assertEqual(list(got_epochs), list(epochs))
        self.assertEqual(list(got_bpm), list(bpm))

    def test_decimals_to_a_tenth_are_held_exactly(self):
        _, packed = HeartRateDay.pack([self.midnight], [71.5])
        self.assertEqual(int(np.frombuffer(packed, dtype="<u2")[0]), 715)

    def test_readings_from_two_utc_days_are_refused(self):
        with self.assertRaises(ValueError):
            HeartRateDay.pack([self.midnight, self.midnight + 86400], [60.0, 60.0])

    def test_a_day_is_unique_per_person(self):
        epochs = self.midnight + np.array([0, 3])
        make_day(self.person, date(2026, 7, 16), epochs, np.array([60.0, 61.0]))
        with self.assertRaises(IntegrityError), transaction.atomic():
            make_day(self.person, date(2026, 7, 16), epochs, np.array([60.0, 61.0]))

    def test_the_same_day_may_exist_for_two_people(self):
        epochs = self.midnight + np.array([0, 3])
        make_day(self.person, date(2026, 7, 16), epochs, np.array([60.0, 61.0]))
        make_day(Person.objects.get(slug="rahil"), date(2026, 7, 16), epochs,
                 np.array([90.0, 91.0]))
        self.assertEqual(HeartRateDay.objects.count(), 2)


class RangeTests(TestCase):
    def test_local_range_is_in_the_persons_own_timezone(self):
        person = Person.objects.get(slug="anas")
        # 03:00Z on the 16th is 23:00 on the 15th in Toronto (UTC-4).
        start = int(datetime(2026, 7, 16, 3, tzinfo=timezone.utc).timestamp())
        make_day(person, date(2026, 7, 16), np.array([start, start + 3]),
                 np.array([60.0, 61.0]))
        self.assertEqual(person.local_range(), (date(2026, 7, 15), date(2026, 7, 15)))
        self.assertEqual(person.status, "data loaded")


class RecordingTests(TestCase):
    def test_same_hash_cannot_be_recorded_twice_for_one_person(self):
        person = Person.objects.get(slug="anas")
        fields = dict(person=person, original_filename="a.csv", sha256="ab" * 32,
                      row_count=2, first_utc=datetime(2026, 7, 16, tzinfo=timezone.utc),
                      last_utc=datetime(2026, 7, 16, tzinfo=timezone.utc),
                      timezone_assumed="America/Toronto")
        Recording.objects.create(**fields)
        with self.assertRaises(IntegrityError), transaction.atomic():
            Recording.objects.create(**fields)
