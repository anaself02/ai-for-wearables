"""Tests for the derived-insight layer.

Every fixture here is constructed so the expected answer is known before the
function runs: a series with an exact 3 bpm drift, a set of nights whose deficit
against the target sums to a round number, a correlation whose p-value is in
standard statistical tables.
"""

from datetime import date, timedelta
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase

from accounts.models import GoogleCredential
from health import insights, metrics

User = get_user_model()

START = date(2026, 8, 21)


def series(values, start=START, step=1):
    """[(date, value)] starting at `start`, one point every `step` days."""
    return [(start + timedelta(days=i * step), v) for i, v in enumerate(values)]


def linear(n, first, per_day):
    return [first + per_day * i for i in range(n)]


# ---------------------------------------------------------------------------
# statistics primitives
# ---------------------------------------------------------------------------


class StatisticsTests(TestCase):
    def test_slope_of_a_known_line(self):
        xs = list(range(10))
        ys = [3 + 2 * x for x in xs]
        self.assertAlmostEqual(insights.linear_slope(xs, ys), 2.0)

    def test_slope_is_none_without_variation_in_x(self):
        self.assertIsNone(insights.linear_slope([5, 5, 5], [1, 2, 3]))

    def test_pearson_of_a_perfect_positive_relationship(self):
        self.assertAlmostEqual(insights.pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)

    def test_pearson_of_a_perfect_negative_relationship(self):
        self.assertAlmostEqual(insights.pearson([1, 2, 3, 4], [8, 6, 4, 2]), -1.0)

    def test_pearson_is_none_when_a_series_is_constant(self):
        self.assertIsNone(insights.pearson([1, 2, 3], [5, 5, 5]))

    def test_p_value_matches_a_standard_table(self):
        # r = 0.5 with n = 20 gives t = 2.4495 on 18 df -> two-tailed p = 0.0248.
        self.assertAlmostEqual(insights.p_value_for_r(0.5, 20), 0.0248, places=4)

    def test_p_value_matches_a_second_table_entry(self):
        # r = 0.3, n = 30 -> t = 1.6641 on 28 df -> p = 0.10725.
        self.assertAlmostEqual(insights.p_value_for_r(0.3, 30), 0.10725, places=5)

    def test_p_value_is_symmetric_in_the_sign_of_r(self):
        self.assertAlmostEqual(
            insights.p_value_for_r(0.42, 25), insights.p_value_for_r(-0.42, 25)
        )

    def test_strength_labels(self):
        self.assertEqual(insights.describe_strength(0.85), "strong")
        self.assertEqual(insights.describe_strength(-0.55), "moderate")
        self.assertEqual(insights.describe_strength(0.35), "weak")
        self.assertEqual(insights.describe_strength(0.1), "very weak")

    def test_align_pairs_only_shared_dates(self):
        left = series([1, 2, 3])
        right = [(START + timedelta(days=1), 10), (START + timedelta(days=5), 20)]
        xs, ys, days = insights.align(left, right)
        self.assertEqual(days, [START + timedelta(days=1)])
        self.assertEqual((xs, ys), ([2.0], [10.0]))

    def test_clean_drops_nones_and_sorts(self):
        messy = [(START + timedelta(days=2), 3), (START, None), (START + timedelta(days=1), 1)]
        self.assertEqual(
            insights.clean(messy),
            [(START + timedelta(days=1), 1.0), (START + timedelta(days=2), 3.0)],
        )


# ---------------------------------------------------------------------------
# individual insights
# ---------------------------------------------------------------------------


class TrendTests(TestCase):
    def _trend(self, values, **kwargs):
        options = dict(key="t", title="T", label="Resting heart rate", unit="bpm")
        options.update(kwargs)
        return insights.trend(series(values), **options)

    def test_reports_a_known_three_bpm_drift(self):
        # 30 days rising exactly 3 bpm end to end: 29 gaps of 3/29 bpm.
        result = self._trend(linear(30, 50.0, 3 / 29))
        self.assertTrue(result.available)
        self.assertAlmostEqual(result.value, 3.0, places=2)
        self.assertEqual(result.direction, "up")
        self.assertIn("up by 3.0 bpm", result.detail)
        self.assertIn("29 days", result.detail)

    def test_reports_a_fall_as_down(self):
        result = self._trend(linear(30, 60.0, -5 / 29))
        self.assertAlmostEqual(result.value, -5.0, places=2)
        self.assertEqual(result.direction, "down")
        self.assertIn("down by 5.0 bpm", result.detail)

    def test_a_flat_series_is_reported_as_flat_not_as_a_tiny_trend(self):
        result = self._trend([55.0] * 30)
        self.assertEqual(result.direction, "flat")
        self.assertIn("essentially flat", result.detail)

    def test_noise_around_a_known_slope_still_recovers_it(self):
        # A repeating [+1,-1,-1,+1] block has zero covariance with the index,
        # so the fitted slope is exactly the underlying 0.1/day over 27 days.
        noise = [1, -1, -1, 1]
        values = [50 + 0.1 * i + noise[i % 4] for i in range(28)]
        result = self._trend(values)
        self.assertAlmostEqual(result.value, 2.7, places=2)

    def test_not_enough_data_degrades_on_its_own(self):
        result = self._trend(linear(5, 50.0, 0.1))
        self.assertFalse(result.available)
        self.assertIn("Needs 14 days", result.note)
        self.assertEqual(result.detail, "")

    def test_the_minimum_is_configurable(self):
        result = self._trend(linear(5, 50.0, 0.25), min_days=5)
        self.assertTrue(result.available)


class SleepDebtTests(TestCase):
    def test_a_known_deficit(self):
        # 10 nights, each 30 minutes under the 450-minute target -> 5 hours down.
        result = insights.sleep_debt(series([420.0] * 10))
        self.assertTrue(result.available)
        self.assertEqual(result.value, -5.0)
        self.assertIn("deficit of 5.0 hours", result.detail)

    def test_a_known_surplus(self):
        result = insights.sleep_debt(series([480.0] * 12))
        self.assertEqual(result.value, 6.0)
        self.assertIn("surplus of 6.0 hours", result.detail)

    def test_deficits_and_surpluses_offset(self):
        result = insights.sleep_debt(series([390.0, 510.0] * 5))
        self.assertEqual(result.value, 0.0)

    def test_the_target_is_configurable(self):
        result = insights.sleep_debt(series([420.0] * 10), target_minutes=420)
        self.assertEqual(result.value, 0.0)

    def test_footnote_reports_the_personal_average_and_short_nights(self):
        result = insights.sleep_debt(series([420.0] * 5 + [480.0] * 5))
        self.assertIn("Average 7.50h", result.footnote)
        self.assertIn("5 of 10 nights", result.footnote)

    def test_not_enough_nights(self):
        result = insights.sleep_debt(series([420.0] * 3))
        self.assertFalse(result.available)
        self.assertIn("Needs 7 nights", result.note)


class RestingHrAfterSleepTests(TestCase):
    def test_detects_a_constructed_difference(self):
        # Alternating short and long nights; short nights carry a 4 bpm penalty.
        nights, beats = [], []
        for i in range(24):
            day = START + timedelta(days=i)
            short = i % 2 == 0
            nights.append((day, 300.0 if short else 540.0))
            beats.append((day, 56.0 if short else 52.0))

        result = insights.resting_hr_after_sleep(nights, beats)
        self.assertTrue(result.available)
        self.assertEqual(result.value, 4.0)
        self.assertIn("4.0 bpm higher", result.detail)
        self.assertEqual(result.sample_size, 24)

    def test_reports_no_meaningful_difference_when_there_is_none(self):
        nights = series([float(300 + 10 * i) for i in range(24)])
        beats = series([53.0] * 24)
        result = insights.resting_hr_after_sleep(nights, beats)
        self.assertIn("about the same", result.detail)

    def test_states_the_circularity_caveat(self):
        nights = series([float(300 + 10 * i) for i in range(24)])
        beats = series([float(60 - i * 0.2) for i in range(24)])
        result = insights.resting_hr_after_sleep(nights, beats)
        # The API derives resting HR from the same night's sleep, which has to
        # be said rather than presented as an independent finding.
        self.assertIn("WITH_SLEEP", result.footnote)
        self.assertIn("not independent", result.footnote)

    def test_only_counts_nights_with_a_matching_reading(self):
        nights = series([400.0] * 24)
        beats = series([53.0] * 5)
        result = insights.resting_hr_after_sleep(nights, beats)
        self.assertFalse(result.available)
        self.assertIn("have 5", result.note)


class PairCorrelationTests(TestCase):
    def _pair(self, left, right, **kwargs):
        options = dict(
            key="p", title="P", left_label="left thing", right_label="right thing"
        )
        options.update(kwargs)
        return insights.pair_correlation(left, right, **options)

    def test_perfect_positive_correlation(self):
        left = series(linear(20, 1.0, 1))
        right = series(linear(20, 10.0, 2))
        result = self._pair(left, right)
        self.assertEqual(result.value, 1.0)
        self.assertIn("higher right thing", result.detail)

    def test_perfect_negative_correlation(self):
        left = series(linear(20, 1.0, 1))
        right = series(linear(20, 100.0, -2))
        result = self._pair(left, right)
        self.assertEqual(result.value, -1.0)
        self.assertIn("lower right thing", result.detail)

    def test_wording_never_claims_causation(self):
        left = series(linear(20, 1.0, 1))
        right = series(linear(20, 10.0, 2))
        result = self._pair(left, right)
        for banned in ("causes", "caused", "because of", "leads to", "results in"):
            self.assertNotIn(banned, result.detail.lower())
            self.assertNotIn(banned, result.footnote.lower())
        self.assertIn("Association only", result.footnote)

    def test_not_enough_overlap(self):
        result = self._pair(series([1.0] * 5), series([2.0] * 5))
        self.assertFalse(result.available)
        self.assertIn("Needs 14 paired days", result.note)

    def test_a_constant_series_is_reported_not_crashed(self):
        result = self._pair(series(linear(20, 1.0, 1)), series([5.0] * 20))
        self.assertFalse(result.available)
        self.assertIn("does not vary", result.note)


class ConsistencyTests(TestCase):
    def test_a_constant_series_is_very_consistent(self):
        result = insights.consistency(
            series([450.0] * 14), key="c", title="C", label="Time asleep"
        )
        self.assertEqual(result.value, 0.0)
        self.assertIn("very consistent", result.detail)

    def test_a_wide_spread_is_reported_as_variable(self):
        result = insights.consistency(
            series([240.0, 660.0] * 7), key="c", title="C", label="Time asleep"
        )
        self.assertIn("variable", result.detail)
        self.assertEqual(result.value, 3.5)

    def test_not_enough_nights(self):
        result = insights.consistency(
            series([450.0] * 3), key="c", title="C", label="Time asleep"
        )
        self.assertFalse(result.available)


class StrongestRelationshipTests(TestCase):
    def test_picks_the_largest_absolute_correlation(self):
        base = linear(20, 1.0, 1)
        result = insights.strongest_relationship(
            {
                "a": series(base),
                "b": series([v * 2 for v in base]),  # r = +1 with a
                "c": series([(v * 7) % 5 for v in base]),  # noisy
            }
        )
        self.assertTrue(result.available)
        self.assertEqual(abs(result.value), 1.0)

    def test_skips_pairs_where_one_metric_is_part_of_the_other(self):
        # Deep sleep is a component of time asleep, so a strong correlation there
        # is arithmetic rather than a finding.
        total = linear(20, 400.0, 5)
        result = insights.strongest_relationship(
            {
                "time asleep": series(total),
                "deep sleep": series([v * 0.2 for v in total]),
                "steps": series([9000 + (i % 3) * 500 for i in range(20)]),
            }
        )
        both_present = "deep sleep" in result.detail and "time asleep" in result.detail
        self.assertFalse(both_present, result.detail)

    def test_corrects_the_threshold_for_multiple_comparisons(self):
        base = linear(20, 1.0, 1)
        result = insights.strongest_relationship(
            {
                "a": series(base),
                "b": series([v + (1 if i % 2 else -1) for i, v in enumerate(base)]),
                "c": series([(v * 3) % 7 for v in base]),
                "d": series([(v * 5) % 11 for v in base]),
            }
        )
        self.assertIn("Bonferroni", result.footnote)
        self.assertIn("comparisons inflates it", result.footnote)

    def test_nothing_to_compare(self):
        result = insights.strongest_relationship({"a": series([1.0] * 5)})
        self.assertFalse(result.available)


# ---------------------------------------------------------------------------
# assembly and view integration
# ---------------------------------------------------------------------------


class WordingAuditTests(TestCase):
    """No insight may read as a finding when its own test did not support one."""

    #: Phrases that assert a real effect. None may appear in a detail sentence
    #: whose significance is SIG_NONE.
    OVERCLAIMING = [
        "significant",
        "proves",
        "shows that",
        "demonstrates",
        "confirms",
        "causes",
        "caused",
        "leads to",
        "results in",
        "due to",
        "because of",
        "clear effect",
        "real effect",
    ]

    def _all_insights(self):
        """A realistic mixed bag: some strong, some pure noise."""
        strong = series(linear(28, 50.0, 0.3))
        noise = series([50 + (i % 5) - 2 for i in range(28)])
        nights = series([float(400 + (i % 7) * 25) for i in range(28)])
        return insights.build(
            {
                "resting_hr": noise,
                "hrv": strong,
                "spo2": series([97 + (i % 3) * 0.4 for i in range(28)]),
                "respiratory_rate": series([16 + (i % 4) * 0.3 for i in range(28)]),
                "sleep_minutes": nights,
                "deep_sleep_minutes": series([float(90 + (i % 5) * 6) for i in range(28)]),
                "steps": series([float(8000 + (i % 6) * 900) for i in range(28)]),
            }
        )

    def test_inconclusive_insights_never_claim_an_effect(self):
        for insight in insights.available(self._all_insights()):
            if insight.is_significant:
                continue
            text = insight.detail.lower()
            for phrase in self.OVERCLAIMING:
                # "not significant after correcting..." is allowed; a bare claim is not.
                if phrase == "significant" and "not significant" in text:
                    continue
                if phrase == "real effect" and "not a real effect" in text:
                    continue
                self.assertNotIn(phrase, text, f"{insight.key}: {insight.detail}")

    def test_no_insight_ever_uses_causal_language(self):
        for insight in insights.available(self._all_insights()):
            combined = f"{insight.detail} {insight.footnote}".lower()
            for phrase in ["causes", "caused by", "leads to", "results in"]:
                self.assertNotIn(phrase, combined, insight.key)

    def test_every_available_insight_declares_its_significance(self):
        for insight in insights.available(self._all_insights()):
            self.assertIn(
                insight.significance,
                {
                    insights.SIG_DESCRIPTIVE,
                    insights.SIG_NONE,
                    insights.SIG_SAMPLE,
                    insights.SIG_CORRECTED,
                },
                insight.key,
            )

    def test_a_non_significant_trend_says_so_in_the_sentence(self):
        # Pure noise around a flat mean: any fitted slope is meaningless.
        noisy = series([50 + (i % 5) - 2 for i in range(28)])
        result = insights.trend(
            noisy, key="t", title="T", label="Resting heart rate", unit="bpm"
        )
        self.assertEqual(result.significance, insights.SIG_NONE)
        self.assertIn("not distinguishable from no trend", result.detail)
        self.assertNotIn("trended", result.detail)

    def test_a_real_trend_is_stated_plainly(self):
        result = insights.trend(
            series(linear(28, 50.0, 0.3)),
            key="t",
            title="T",
            label="Resting heart rate",
            unit="bpm",
        )
        self.assertEqual(result.significance, insights.SIG_SAMPLE)
        self.assertIn("trended up", result.detail)

    def test_a_null_correlation_leads_with_the_null_result(self):
        left = series([float(i % 5) for i in range(28)])
        right = series([float((i * 7) % 11) for i in range(28)])
        result = insights.pair_correlation(
            left, right, key="p", title="P", left_label="alpha", right_label="beta"
        )
        self.assertEqual(result.significance, insights.SIG_NONE)
        self.assertTrue(result.detail.startswith("Tested, no clear relationship"))
        self.assertIn("within what random variation would produce", result.detail)

    def test_strongest_relationship_states_the_correction_in_the_sentence(self):
        base = linear(24, 1.0, 1)
        result = insights.strongest_relationship(
            {
                "a": series(base),
                "b": series([v + (1 if i % 2 else -1) for i, v in enumerate(base)]),
                "c": series([(v * 3) % 7 for v in base]),
                "d": series([(v * 5) % 11 for v in base]),
            }
        )
        # The count of tests must be in the headline sentence, not just the footnote.
        self.assertIn("relationships tested", result.detail)
        combined = result.detail + " " + result.footnote
        self.assertIn("correcting", combined)
        self.assertIn("Bonferroni", combined)

    def test_strongest_relationship_that_fails_correction_says_lead_not_effect(self):
        # Weak, noisy series: the winner will not clear a Bonferroni threshold.
        result = insights.strongest_relationship(
            {
                "a": series([float((i * 3) % 7) for i in range(24)]),
                "b": series([float((i * 5) % 11) for i in range(24)]),
                "c": series([float((i * 2) % 5) for i in range(24)]),
                "d": series([float((i * 7) % 13) for i in range(24)]),
            }
        )
        if not result.meta.get("survives_correction", False):
            self.assertIn("not significant after correcting", result.detail)
            self.assertIn("not a real effect", result.detail)


class MethodologyTests(TestCase):
    def _computed(self):
        return insights.build(
            {
                "resting_hr": series(linear(28, 50.0, 0.1)),
                "hrv": series(linear(28, 60.0, 0.2)),
                "spo2": series([97 + (i % 3) * 0.4 for i in range(28)]),
                "sleep_minutes": series([float(400 + (i % 7) * 25) for i in range(28)]),
                "deep_sleep_minutes": series([float(90 + (i % 5) * 6) for i in range(28)]),
                "steps": series([float(8000 + (i % 6) * 900) for i in range(28)]),
            }
        )

    def test_states_the_with_sleep_circularity(self):
        notes = insights.methodology(self._computed())
        body = " ".join(note["body"] for note in notes)
        self.assertIn("WITH_SLEEP", body)
        self.assertIn("not independent", body)
        self.assertIn("definitional", body)

    def test_states_the_multiple_comparison_correction_with_real_numbers(self):
        computed = self._computed()
        notes = insights.methodology(computed)
        note = next(n for n in notes if n["key"] == "multiple-comparisons")

        strongest = next(i for i in computed if i.key == "strongest-relationship")
        tested = strongest.meta["pairs_tested"]
        # The prose must quote what was actually computed, not a fixed number.
        self.assertIn(f"{tested} correlations tested", note["body"])
        self.assertIn("Bonferroni", note["body"])
        self.assertIn(f"{strongest.meta['bonferroni_threshold']:.4f}", note["body"])

    def test_states_why_nested_metrics_were_excluded(self):
        notes = insights.methodology(self._computed())
        note = next(n for n in notes if n["key"] == "nested-metrics")
        self.assertIn("deep sleep", note["body"])
        self.assertIn("time asleep", note["body"])
        self.assertIn("arithmetic", note["body"])

    def test_states_the_sample_limitation_and_autocorrelation(self):
        notes = insights.methodology(self._computed())
        body = " ".join(note["body"] for note in notes)
        self.assertIn("no control group", body.lower())
        self.assertIn("autocorrelated", body)
        self.assertIn("cannot show that one thing causes another", body)

    def test_notes_are_omitted_when_their_insight_is_unavailable(self):
        notes = insights.methodology(insights.build({}))
        keys = {note["key"] for note in notes}
        self.assertNotIn("multiple-comparisons", keys)
        self.assertNotIn("circularity", keys)
        # The general caveats always apply.
        self.assertIn("sample", keys)


class BuildTests(TestCase):
    def test_empty_input_yields_all_unavailable_rather_than_an_error(self):
        computed = insights.build({})
        self.assertTrue(computed)
        self.assertEqual(insights.available(computed), [])
        self.assertEqual(len(insights.pending(computed)), len(computed))
        for insight in computed:
            self.assertTrue(insight.note)

    def test_one_thin_metric_does_not_blank_the_others(self):
        computed = insights.build(
            {
                "resting_hr": series(linear(30, 50.0, 0.1)),
                "hrv": series([60.0, 61.0]),  # far too short
            }
        )
        keys = {i.key for i in insights.available(computed)}
        pending_keys = {i.key for i in insights.pending(computed)}
        self.assertIn("rhr-trend", keys)
        self.assertIn("hrv-trend", pending_keys)


class InsightSeriesTests(TestCase):
    """The bridge from API points to the series the insight layer consumes."""

    def _sleep_point(self, day, asleep, deep, nap=False):
        metadata = {"processed": True}
        if nap:
            metadata["nap"] = True
        else:
            metadata["mainSleep"] = True
        return {
            "sleep": {
                "interval": {
                    "endTime": f"{day}T11:00:00Z",
                    "civilEndTime": {
                        "date": {
                            "year": day.year,
                            "month": day.month,
                            "day": day.day,
                        }
                    },
                },
                "metadata": metadata,
                "summary": {
                    "minutesAsleep": str(asleep),
                    "minutesInSleepPeriod": str(asleep + 30),
                    "stagesSummary": [{"type": "DEEP", "minutes": str(deep)}],
                },
            }
        }

    def test_naps_are_excluded_from_the_nightly_series(self):
        # `nap` is the reliable flag; mainSleep is set on some naps too.
        points = [
            self._sleep_point(date(2026, 9, 1), 480, 100),
            self._sleep_point(date(2026, 9, 2), 95, 20, nap=True),
        ]
        asleep, _ = metrics.sleep_pairs(points)
        self.assertEqual(asleep, [(date(2026, 9, 1), 480.0)])

    def test_a_date_with_two_sessions_keeps_the_longest(self):
        points = [
            self._sleep_point(date(2026, 9, 1), 130, 20),
            self._sleep_point(date(2026, 9, 1), 470, 95),
        ]
        asleep, deep = metrics.sleep_pairs(points)
        self.assertEqual(asleep, [(date(2026, 9, 1), 470.0)])
        self.assertEqual(deep, [(date(2026, 9, 1), 95.0)])

    def test_as_daily_pairs_converts_iso_labels_to_dates(self):
        chart = metrics.chart_data(
            ["2026-09-01", "2026-09-02"],
            [{"label": "x", "data": [1, None]}],
        )
        self.assertEqual(metrics.as_daily_pairs(chart), [(date(2026, 9, 1), 1.0)])

    def test_as_daily_pairs_ignores_clock_labels(self):
        chart = metrics.chart_data(["16:13"], [{"label": "x", "data": [70]}])
        self.assertEqual(metrics.as_daily_pairs(chart), [])

    def test_tail_slices_every_dataset(self):
        chart = metrics.chart_data(
            ["a", "b", "c", "d"],
            [{"label": "x", "data": [1, 2, 3, 4]}, {"label": "y", "data": [5, 6, 7, 8]}],
        )
        sliced = metrics.tail(chart, 2)
        self.assertEqual(sliced["labels"], ["c", "d"])
        self.assertEqual(sliced["datasets"][0]["data"], [3, 4])
        self.assertEqual(sliced["datasets"][1]["data"], [7, 8])


class InsightsPageTests(TestCase):
    """The insights page, which is where these live now.

    They used to render at the bottom of the overview. They moved so that a
    derived claim is never read as one of the measurements above it.
    """

    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)
        self.user = User.objects.create(username="tester@example.com")
        GoogleCredential.objects.create(
            user=self.user, google_sub="sub-1", refresh_token="1//refresh"
        )
        self.client.force_login(self.user)

    def _get(self, url, rhr_points=None):
        fake = mock.MagicMock()
        fake.list_data_points.side_effect = lambda dt, **kw: (
            rhr_points or [] if dt.slug == "daily-resting-heart-rate" else []
        )
        fake.daily_roll_up.return_value = []
        with mock.patch("health.views.GoogleHealthClient", return_value=fake):
            return self.client.get(url)

    def _insights(self, rhr_points=None):
        return self._get("/insights/", rhr_points)

    def _dashboard(self, rhr_points=None):
        return self._get("/dashboard/", rhr_points)

    def test_requires_login(self):
        self.client.logout()
        response = self.client.get("/insights/")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response["Location"].startswith("/?next="))

    def test_the_overview_no_longer_derives_anything(self):
        """The split is the point: the overview carries measurements only."""
        response = self._dashboard()
        for derived in ("insights", "insights_pending", "methodology", "arousal"):
            self.assertNotIn(derived, response.context)

    def test_section_renders_with_no_data_without_erroring(self):
        response = self._insights()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["insights"], [])
        self.assertTrue(response.context["insights_pending"])
        self.assertContains(response, "waiting for more data")

    def test_methodology_notes_render_on_the_page_itself(self):
        points = [
            {
                "dailyRestingHeartRate": {
                    "date": {"year": 2026, "month": 9, "day": 1 + i},
                    "beatsPerMinute": str(50 + i // 5),
                }
            }
            for i in range(20)
        ]
        response = self._insights(rhr_points=points)
        self.assertContains(response, "Methodology notes")
        self.assertContains(response, "no control group")
        self.assertContains(response, "autocorrelated")

    def test_weak_insights_stay_visible_rather_than_being_hidden(self):
        points = [
            {
                "dailyRestingHeartRate": {
                    "date": {"year": 2026, "month": 9, "day": 1 + i},
                    "beatsPerMinute": str(50 + (i % 5) - 2),
                }
            }
            for i in range(20)
        ]
        response = self._insights(rhr_points=points)
        trend_insight = next(
            i for i in response.context["insights"] if i.key == "rhr-trend"
        )
        # Showing "tested, not distinguishable from chance" beats silence.
        self.assertFalse(trend_insight.is_significant)
        self.assertContains(response, "not distinguishable from chance")

    def test_a_real_trend_appears_in_the_section(self):
        points = [
            {
                "dailyRestingHeartRate": {
                    "date": {"year": 2026, "month": 9, "day": 1 + i},
                    "beatsPerMinute": str(50 + i // 5),
                }
            }
            for i in range(20)
        ]
        response = self._insights(rhr_points=points)
        keys = {i.key for i in response.context["insights"]}
        self.assertIn("rhr-trend", keys)
        self.assertContains(response, "Resting heart rate trend")
        self.assertContains(response, "Insights")

    def test_the_dashboard_fetches_a_month_but_sparklines_show_a_week(self):
        response = self._dashboard()
        self.assertEqual(response.context["days"], 30)
        points = [
            {
                "dailyRestingHeartRate": {
                    "date": {"year": 2026, "month": 9, "day": 1 + i},
                    "beatsPerMinute": "52",
                }
            }
            for i in range(20)
        ]
        cache.clear()
        response = self._dashboard(rhr_points=points)
        card = next(
            c for c in response.context["cards"]
            if c["metric"].slug == "daily-resting-heart-rate"
        )
        self.assertLessEqual(len(card["series"]["labels"]), 7)
