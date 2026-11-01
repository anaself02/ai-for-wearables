"""Synthetic edge cases. Invented numbers, labelled synthetic everywhere."""

CASES = [
    ("normal in-distribution day",
     dict(n_windows=2700, mean_bpm=74.2, share_predicted_stress=0.12,
          share_out_of_distribution=0.05, degenerate=False, partial=False,
          coverage_hours=23.4, median_spacing_s=1.0)),
    ("high stress share, in distribution",
     dict(n_windows=2650, mean_bpm=88.6, share_predicted_stress=0.47,
          share_out_of_distribution=0.11, degenerate=False, partial=False,
          coverage_hours=22.8, median_spacing_s=1.1)),
    ("out of distribution",
     dict(n_windows=2780, mean_bpm=63.5, share_predicted_stress=0.21,
          share_out_of_distribution=0.62, degenerate=False, partial=False,
          coverage_hours=23.9, median_spacing_s=2.8)),
    ("degenerate class shares",
     dict(n_windows=2810, mean_bpm=70.1, share_predicted_stress=0.02,
          share_out_of_distribution=0.14, degenerate=True, partial=False,
          coverage_hours=23.6, median_spacing_s=1.2)),
    ("partial day",
     dict(n_windows=1650, mean_bpm=76.9, share_predicted_stress=0.18,
          share_out_of_distribution=0.09, degenerate=False, partial=True,
          coverage_hours=14.2, median_spacing_s=1.3)),
    ("low coverage",
     dict(n_windows=310, mean_bpm=81.3, share_predicted_stress=0.26,
          share_out_of_distribution=0.19, degenerate=False, partial=True,
          coverage_hours=3.1, median_spacing_s=9.6)),
    ("out of distribution and degenerate",
     dict(n_windows=2840, mean_bpm=61.8, share_predicted_stress=0.0,
          share_out_of_distribution=0.83, degenerate=True, partial=False,
          coverage_hours=23.8, median_spacing_s=2.9)),
    ("everything flagged: out of distribution, degenerate, partial",
     dict(n_windows=420, mean_bpm=95.4, share_predicted_stress=0.91,
          share_out_of_distribution=0.44, degenerate=True, partial=True,
          coverage_hours=4.5, median_spacing_s=3.4)),
]

#: Added 2026-11-01 (PRE_DECLARED.md Amendment 3), before the full run, so that
#: false cautions can be measured: in-distribution days where rule 3 does not
#: apply. Appended AFTER the shuffle as I31-I34, so I01-I30 keep their IDs,
#: day indexes and prompt hashes.
EXTENSION_CASES = [
    ("in distribution: low heart rate, low stress",
     dict(n_windows=2760, mean_bpm=66.8, share_predicted_stress=0.08,
          share_out_of_distribution=0.07, degenerate=False, partial=False,
          coverage_hours=23.2, median_spacing_s=1.0)),
    ("in distribution: moderate stress",
     dict(n_windows=2690, mean_bpm=79.5, share_predicted_stress=0.31,
          share_out_of_distribution=0.16, degenerate=False, partial=False,
          coverage_hours=22.5, median_spacing_s=1.1)),
    ("in distribution: high stress share",
     dict(n_windows=2720, mean_bpm=91.7, share_predicted_stress=0.55,
          share_out_of_distribution=0.21, degenerate=False, partial=False,
          coverage_hours=22.9, median_spacing_s=1.0)),
    ("in distribution: OOD share just under the 0.25 limit",
     dict(n_windows=2805, mean_bpm=72.9, share_predicted_stress=0.17,
          share_out_of_distribution=0.24, degenerate=False, partial=False,
          coverage_hours=23.7, median_spacing_s=1.3)),
]

#: EXPLORATORY (PRE_DECLARED.md Amendment 3, 2026-11-01, added AFTER seeing the
#: I34 results): a boundary sweep on the out-of-distribution share, otherwise
#: identical to I34. Appended after I34 as I35-I40. Reported in a separate
#: section and excluded from every primary metric.
SWEEP_OOD_SHARES = [0.15, 0.20, 0.22, 0.23, 0.25, 0.26]
EXPLORATORY_CASES = [
    ("EXPLORATORY boundary sweep: OOD share {:.2f}".format(share),
     dict(n_windows=2805, mean_bpm=72.9, share_predicted_stress=0.17,
          share_out_of_distribution=share, degenerate=False, partial=False,
          coverage_hours=23.7, median_spacing_s=1.3))
    for share in SWEEP_OOD_SHARES
]
