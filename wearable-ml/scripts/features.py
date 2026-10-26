"""
features.py
-----------
The feature-extraction logic, kept in its own file for two reasons:

  1. 02_feature_extraction.py imports it, so the windowing/feature code is
     written once instead of being copy-pasted.
  2. It gets copied into models/ alongside the trained model, so that the
     EXACT same feature definitions can be re-applied later to personal
     wearable data.

Everything here is plain functions - no classes, no hidden state.
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks

import project_config as config


# ---------------------------------------------------------------------------
# Heart rate from BVP
# ---------------------------------------------------------------------------

def heart_rate_series(bvp, fs=config.FS_WRIST_BVP):
    """
    Estimate an instantaneous heart-rate series from a raw BVP (PPG) segment.

    Method, step by step:

      1. Band-pass filter the BVP with a 3rd-order Butterworth filter between
         0.7 Hz and 3.7 Hz (= 42 to 222 bpm). This removes two things:
           - baseline wander / slow drift below 0.7 Hz (arm movement, pressure
             changes of the strap against the skin),
           - high-frequency sensor noise above 3.7 Hz.
         We use filtfilt (forward-backward filtering) rather than lfilter so
         the filter introduces ZERO phase shift - important here, because we
         are about to measure the *timing* of peaks and a phase shift would
         move them.

      2. Find the peaks of the filtered signal with scipy.signal.find_peaks,
         using two constraints:
           - distance = 0.4 s * fs samples. Each pulse wave has a large
             systolic peak followed by a smaller "dicrotic notch" bump. A
             minimum spacing of 0.4 s (i.e. a 150 bpm ceiling) prevents that
             second bump from being counted as an extra heartbeat.
           - prominence = 0.5 * standard deviation of the filtered segment.
             Prominence measures how far a peak stands out from the
             surrounding signal. Scaling it by the segment's own standard
             deviation makes the threshold adapt to each subject and to each
             window's signal strength, instead of hard-coding an absolute
             amplitude that would fail on a weak signal.

      3. Take the gaps between consecutive peaks - the inter-beat intervals
         (IBIs), in seconds - and MEDIAN-FILTER them over
         config.IBI_MEDIAN_FILTER_BEATS consecutive intervals.

         This step was added after the first version of this pipeline was
         evaluated, because without it the resulting features described
         detection error more than they described heart rate: the average
         60-second WESAD window spanned HR_min 46 bpm to HR_max 136 bpm,
         which is not physiologically possible. See the long note on
         IBI_MEDIAN_FILTER_BEATS in project_config.py for the full reasoning
         and for why the width is 5.

         The filter is applied to the IBI series rather than to the bpm
         series because that is where the error physically lives. One missed
         beat merges two real intervals into a single long one; one spurious
         peak splits a real interval into two short ones. Both are local
         corruptions of the interval sequence, and a median over neighbouring
         intervals replaces a corrupted value with a typical nearby one. The
         median is used rather than a mean precisely because a mean would let
         a single doubled interval drag the whole neighbourhood with it.

         mode="nearest" repeats the edge values when the filter window runs
         off either end of the series. The default for this function pads
         with ZEROS, which here would manufacture a zero-length interval and
         hence an infinite heart rate at the first and last beats of every
         window.

      4. Convert the filtered intervals to beats per minute:
         bpm = 60 / interval_in_seconds.

      5. Discard any remaining value outside 40-180 bpm. After the median
         filter this should catch very little; it is kept as a final guard
         for windows so corrupted that even the local median is implausible
         (for example, the watch off the wrist entirely).

    Returns a 1-D array of instantaneous heart-rate values in bpm, one per
    detected beat-to-beat interval. Returns an empty array if too few peaks
    were found (e.g. a badly corrupted window).
    """
    bvp = np.asarray(bvp, dtype=float).ravel()

    # A Butterworth band-pass needs its cutoffs expressed as a fraction of the
    # Nyquist frequency (half the sampling rate).
    nyquist = fs / 2.0
    low = config.BVP_BANDPASS_LOW_HZ / nyquist
    high = config.BVP_BANDPASS_HIGH_HZ / nyquist
    b, a = butter(config.BVP_FILTER_ORDER, [low, high], btype="band")

    # filtfilt needs the segment to be longer than about 3x the filter order.
    if len(bvp) < 3 * (config.BVP_FILTER_ORDER + 1):
        return np.array([])

    filtered = filtfilt(b, a, bvp)

    # A flat segment (sensor detached) has no peaks worth finding.
    sd = float(np.std(filtered))
    if sd == 0.0:
        return np.array([])

    min_distance_samples = int(round(config.BVP_MIN_PEAK_DISTANCE_S * fs))
    peaks, _ = find_peaks(filtered,
                          distance=min_distance_samples,
                          prominence=0.5 * sd)

    # Need at least 2 peaks to have 1 interval.
    if len(peaks) < 2:
        return np.array([])

    ibi_seconds = np.diff(peaks) / float(fs)

    # Median-filter the interval series to remove missed/doubled-beat
    # artifacts before they become heart-rate values. See step 3 above.
    if config.IBI_MEDIAN_FILTER_BEATS > 1:
        ibi_seconds = median_filter(ibi_seconds,
                                    size=config.IBI_MEDIAN_FILTER_BEATS,
                                    mode="nearest")

    bpm = 60.0 / ibi_seconds

    # Keep only physiologically plausible values.
    plausible = (bpm >= config.HR_MIN_BPM) & (bpm <= config.HR_MAX_BPM)
    return bpm[plausible]


# ---------------------------------------------------------------------------
# The five per-signal statistics
# ---------------------------------------------------------------------------

def basic_features(x, fs):
    """
    Compute the five engineered features for one signal segment:
    mean, standard deviation, minimum, maximum, and slope.

    'slope' is the gradient of a straight line fitted through the segment by
    least squares, expressed in signal-units PER SECOND (not per sample). It
    captures the direction and speed of drift within the window - for example
    skin temperature falling during stress, or EDA ramping up.

    Returns a dict of {feature_name: value}. If the segment is empty we return
    NaN for every feature; 02_feature_extraction.py drops such rows.
    """
    x = np.asarray(x, dtype=float).ravel()

    if len(x) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan,
                "max": np.nan, "slope": np.nan}

    # Time axis in seconds, so the slope has units of "per second" and is
    # therefore comparable between signals sampled at different rates.
    t = np.arange(len(x)) / float(fs)

    if len(x) < 2:
        slope = 0.0
    else:
        # polyfit(t, x, 1) returns [slope, intercept]; we want the slope.
        slope = float(np.polyfit(t, x, 1)[0])

    return {"mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "slope": slope}


# ---------------------------------------------------------------------------
# All features for one 60-second window
# ---------------------------------------------------------------------------

def window_features(wrist_window):
    """
    Given the wrist signals for ONE window, return a flat dict of features.

    `wrist_window` is a dict with these keys (each a numpy array covering the
    same 60 seconds of time):
        'ACC'  : shape (n, 3)  @ 32 Hz
        'BVP'  : shape (n, 1)  @ 64 Hz
        'EDA'  : shape (n, 1)  @  4 Hz
        'TEMP' : shape (n, 1)  @  4 Hz

    We turn that into 8 "signal channels" and compute the same 5 statistics on
    each one, giving 8 x 5 = 40 features:

        EDA, TEMP, BVP                  - as recorded
        ACC_x, ACC_y, ACC_z             - the three accelerometer axes
        ACC_mag                         - sqrt(x^2 + y^2 + z^2), overall
                                          movement intensity regardless of how
                                          the watch happens to be oriented
        HR                              - instantaneous heart rate derived
                                          from BVP (see heart_rate_series)

    Feature names look like "EDA_mean", "ACC_mag_slope", "HR_std", ...
    """
    feats = {}

    acc = np.asarray(wrist_window["ACC"], dtype=float)
    acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))

    # (channel name, samples, sampling rate) for the 7 directly-measured ones.
    channels = [
        ("EDA",     wrist_window["EDA"],  config.FS_WRIST_EDA),
        ("TEMP",    wrist_window["TEMP"], config.FS_WRIST_TEMP),
        ("BVP",     wrist_window["BVP"],  config.FS_WRIST_BVP),
        ("ACC_x",   acc[:, 0],            config.FS_WRIST_ACC),
        ("ACC_y",   acc[:, 1],            config.FS_WRIST_ACC),
        ("ACC_z",   acc[:, 2],            config.FS_WRIST_ACC),
        ("ACC_mag", acc_mag,              config.FS_WRIST_ACC),
    ]

    for name, samples, fs in channels:
        for stat_name, value in basic_features(samples, fs).items():
            feats[name + "_" + stat_name] = value

    # Heart rate is a special case: it is not evenly sampled (there is one
    # value per detected heartbeat, and heartbeats are not equally spaced).
    # For the slope we therefore treat the beat index as the time axis at a
    # nominal 1 "sample" per beat, which makes HR_slope mean "bpm change per
    # beat" - a measure of whether the heart rate is rising or falling across
    # the window.
    hr = heart_rate_series(wrist_window["BVP"])
    for stat_name, value in basic_features(hr, fs=1.0).items():
        feats["HR_" + stat_name] = value

    return feats


# The order of channels and statistics above determines the column order of
# the feature table. This list is written to models/feature_config.json so the
# saved model can be fed columns in exactly the right order later.
CHANNEL_NAMES = ["EDA", "TEMP", "BVP", "ACC_x", "ACC_y", "ACC_z", "ACC_mag", "HR"]
STAT_NAMES = ["mean", "std", "min", "max", "slope"]
FEATURE_NAMES = [c + "_" + s for c in CHANNEL_NAMES for s in STAT_NAMES]


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
# We train on two different subsets of the same feature table:
#
#   "full"    - all 40 features from all 8 channels. This is what a
#               research-grade multi-sensor wristband (the Empatica E4 used in
#               WESAD) can provide: EDA, skin temperature, raw PPG and raw
#               accelerometry.
#
#   "hr_only" - the 5 heart-rate features only. This is what a mainstream
#               consumer wearable actually reports. A Fitbit-class device
#               gives heart rate (plus derived things like HRV and resting HR)
#               but does NOT expose EDA, raw BVP or raw accelerometer samples.
#
# Training both on the identical windows, with the identical cross-validation
# protocol, answers a specific question: how much classification performance
# is lost when you are restricted to wrist heart rate alone?
#
# Note the "hr_only" features are still derived from WESAD's BVP signal by our
# own peak detection - they are not Fitbit data. The point is that they use
# only the *kind* of information a consumer device makes available.

HR_ONLY_FEATURE_NAMES = [name for name in FEATURE_NAMES if name.startswith("HR_")]

FEATURE_SETS = {
    "full": FEATURE_NAMES,
    "hr_only": HR_ONLY_FEATURE_NAMES,
}
