"""
project_config.py
-----------------
All project-wide constants live here, in ONE place, so that every numbered
script (01_..., 02_..., ...) uses exactly the same settings and so that
changing a setting means editing a single line.

Nothing in this file does any work - it only defines values.

Why the name is "project_config" and not just "config": script 05 boots the
Django health-dashboard project in-process, and that project has its own
`config` PACKAGE (config/settings.py). Two different things called `config`
cannot both live in sys.modules, so ours carries a distinct name. Every
script imports it as `import project_config as config`, so the rest of the
code still reads `config.WINDOW_SECONDS`.
"""

import os

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------

# Folder that contains this file (wearable-ml/scripts)
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root (wearable-ml)
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)

# Where the extracted WESAD dataset lives (wearable-ml/data/WESAD)
WESAD_DIR = os.path.join(PROJECT_DIR, "data", "WESAD")

# Where intermediate outputs go (feature table, plots, metrics)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Where trained models and the saved feature configuration go
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# The feature table produced by 02_feature_extraction.py
FEATURES_CSV = os.path.join(RESULTS_DIR, "features.csv")

# --- personal (non-WESAD) data, used only by 05_apply_to_personal_data.py ---

# The Django health-dashboard project. 05 adds this to sys.path and boots it
# so it can reuse that project's stored Google OAuth credentials and its
# Google Health API client. We deliberately do NOT re-implement any OAuth or
# client-secret handling here - there must be exactly one copy of that code.
DASHBOARD_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "health dashboard")

# Outputs derived from personal wearable data are kept apart from the WESAD
# results, so the two can never be confused in the write-up.
PERSONAL_DIR = os.path.join(RESULTS_DIR, "personal")


# ---------------------------------------------------------------------------
# 2. Subjects
# ---------------------------------------------------------------------------
# WESAD ships subjects S2..S17. S1 and S12 were excluded by the dataset authors
# (sensor malfunction), so those folders simply do not exist. That is expected.

SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
            "S10", "S11", "S13", "S14", "S15", "S16", "S17"]


# ---------------------------------------------------------------------------
# 3. Signals and their sampling rates
# ---------------------------------------------------------------------------
# These rates were CONFIRMED by inspection, not taken from the paper:
# for every subject, len(signal) / (len(label) / 700) came out to exactly
# these numbers. See 01_load_and_inspect.py, which re-checks them.
#
# We deliberately use ONLY the wrist (Empatica E4) signals, because a consumer
# wearable (Fitbit-class) reports wrist data - not chest ECG/EMG/respiration.

FS_LABEL = 700          # Hz - the label array is sampled at the chest rate
FS_WRIST_ACC = 32       # Hz - 3-axis accelerometer, units of 1/64 g
FS_WRIST_BVP = 64       # Hz - blood volume pulse (photoplethysmogram)
FS_WRIST_EDA = 4        # Hz - electrodermal activity, microsiemens
FS_WRIST_TEMP = 4       # Hz - skin temperature, degrees Celsius


# ---------------------------------------------------------------------------
# 4. Label encoding
# ---------------------------------------------------------------------------
# Confirmed empirically in step 01 by matching each label segment's duration
# and ordering against S2_quest.csv (the study protocol timings):
#
#   0 = transient / not defined  -> drop
#   1 = baseline                 -> keep
#   2 = stress (TSST)            -> keep
#   3 = amusement (funny video)  -> keep
#   4 = meditation               -> drop (not part of the 3-class benchmark)
#   5, 6, 7 = recovery / reading tasks, flagged "to be ignored" by the
#             dataset authors -> drop
#
# We use the 3-class configuration (baseline / stress / amusement) because that
# is the standard benchmark reported in the original WESAD paper, which makes
# our numbers comparable to published results.

LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement"}

# The numeric labels we keep. Everything else is discarded.
TARGET_LABELS = [1, 2, 3]


# ---------------------------------------------------------------------------
# 5. Windowing
# ---------------------------------------------------------------------------
# Why 60 seconds:
#   EDA (skin conductance) responses and skin-temperature changes are slow -
#   they unfold over tens of seconds. A 60 s window is long enough to contain
#   several of them and to give a stable heart-rate estimate, and it is the
#   window length most commonly used in the WESAD literature.
#
# Why 50% overlap (30 s step):
#   We only have 15 subjects and a few tens of minutes of usable data each.
#   Overlapping windows roughly double the number of training rows without
#   inventing data. NOTE: overlapping windows are NOT statistically independent
#   (neighbouring windows share 30 s of signal). That is acceptable here ONLY
#   because we always split train/test by SUBJECT - two overlapping windows
#   always belong to the same subject, so they can never straddle the split.

WINDOW_SECONDS = 60
STEP_SECONDS = 30       # 30 s step = 50% overlap with a 60 s window

# A window is kept only if this fraction (or more) of its label samples carry
# the SAME label. 1.0 = the window must be 100% pure; any window that straddles
# a transition between conditions is thrown away rather than being given a
# possibly-wrong majority label. Lower this (e.g. 0.9) to keep more windows.
LABEL_PURITY_THRESHOLD = 1.0


# ---------------------------------------------------------------------------
# 6. Heart-rate extraction from BVP
# ---------------------------------------------------------------------------
# The synchronised S{n}.pkl file contains wrist ACC, BVP, EDA and TEMP - but
# NOT heart rate. (The raw S{n}_E4_Data/HR.csv exists, but it covers the full
# untrimmed recording and would have to be re-aligned by unix timestamp, which
# is an extra source of error.) So we compute heart rate ourselves from BVP.
# See features.py -> heart_rate_series() for the method.

BVP_BANDPASS_LOW_HZ = 0.7     # 0.7 Hz = 42 bpm  (below any plausible heart rate)
BVP_BANDPASS_HIGH_HZ = 3.7    # 3.7 Hz = 222 bpm (above any plausible heart rate)
BVP_FILTER_ORDER = 3

# Minimum time between two accepted pulse peaks. 0.4 s = 150 bpm ceiling.
# This stops the peak finder from counting the dicrotic notch (the small second
# bump in each pulse wave) as a separate heartbeat.
BVP_MIN_PEAK_DISTANCE_S = 0.4

# Width, in beats, of the median filter applied to the inter-beat-interval
# (IBI) series before it is converted to a heart rate. 0 disables the filter.
#
# WHY THIS EXISTS. Without it, the HR features were dominated by beat-detection
# error rather than by heart rate. Measured on the unfiltered feature table,
# the average WESAD window had HR_min 46 bpm and HR_max 136 bpm - a 90 bpm
# swing inside a single 60-second window. No heart does that. Those extremes
# come from a handful of misdetected beats per window: a missed beat makes one
# interval roughly twice as long (an implausibly low instantaneous rate), and a
# spurious extra peak splits one interval into two short ones (an implausibly
# high rate). The 40-180 bpm plausibility filter alone was too loose to catch
# them, and they inflated HR_std, HR_min and HR_max.
#
# WHY 5. A median filter of width w removes runs of up to floor(w/2)
# consecutive outliers while leaving a smooth underlying trend essentially
# untouched. Detection errors here are isolated or come in pairs (one missed
# beat often disturbs its neighbour), so w = 5 - which absorbs up to two
# consecutive bad intervals - is the smallest width that covers the observed
# failure mode. w = 3 handles only single isolated errors.
#
# WHY NOT LARGER. At a resting rate of about one beat per second, w = 5 spans
# roughly 5 seconds. Respiratory sinus arrhythmia - the genuine beat-to-beat
# variation in time with breathing - cycles about every 4 seconds, so a much
# wider window would start erasing real physiology along with the artifacts.
# w = 5 is the compromise: wide enough for the artifacts, narrow enough that
# breathing-driven variation survives, though some attenuation of it is
# accepted as the price of the fix.
IBI_MEDIAN_FILTER_BEATS = 5

# Plausible heart-rate range. Beat-to-beat intervals implying a rate outside
# this range are treated as detection errors and dropped.
HR_MIN_BPM = 40
HR_MAX_BPM = 180


# ---------------------------------------------------------------------------
# 7. Cross-validation / model settings
# ---------------------------------------------------------------------------

RANDOM_STATE = 42       # fixed so every run gives the same result
