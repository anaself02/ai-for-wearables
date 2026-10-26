"""
01_load_and_inspect.py
----------------------
STEP 1: Look at the raw data before trusting any assumption about it.

This script loads ONE subject's .pkl file (S2 by default) and prints:
  - the dictionary structure of the file,
  - the shape and sampling rate of every wrist signal, CHECKED against the
    length of the label array rather than assumed from the documentation,
  - which label values occur, how long each one lasts, and in what order,
  - the raw Empatica E4 CSV layout for the same subject.

It writes nothing. Run it, read the output, and confirm the schema is what
the later scripts expect.

Usage:
    python 01_load_and_inspect.py          # inspects S2
    python 01_load_and_inspect.py S5       # inspects S5
"""

import os
import pickle
import sys

import numpy as np

import project_config as config


def load_subject(subject):
    """
    Load one subject's synchronised .pkl file.

    encoding="latin1" is required because these pickles were written with
    Python 2; without it, Python 3 raises a UnicodeDecodeError on the byte
    strings inside.
    """
    path = os.path.join(config.WESAD_DIR, subject, subject + ".pkl")
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def describe_structure(data):
    """Print the nested dictionary layout of the loaded pickle."""
    print("Top-level keys:", list(data.keys()))
    print("'subject' field:", repr(data["subject"]))
    print("'label' array :", data["label"].shape, data["label"].dtype)
    print()
    print("'signal' contains these sensor groups:", list(data["signal"].keys()))
    for group in data["signal"]:
        print("  " + group + ":")
        for name, array in data["signal"][group].items():
            print("    {:6s} shape={:18s} dtype={}".format(
                name, str(array.shape), array.dtype))
    print()


def check_sampling_rates(data):
    """
    Verify the wrist sampling rates instead of assuming them.

    The label array is sampled at 700 Hz, so the recording lasts
    len(label) / 700 seconds. Dividing each signal's length by that duration
    gives its actual sampling rate. If these come out as clean integers that
    match config.py, the signals and the labels really do cover the same
    stretch of time and we can window them by wall-clock seconds.
    """
    label = data["label"]
    duration_s = len(label) / float(config.FS_LABEL)
    print("Recording duration from label array: {:.1f} s "
          "({:.1f} min)".format(duration_s, duration_s / 60.0))
    print()

    expected = {"ACC": config.FS_WRIST_ACC,
                "BVP": config.FS_WRIST_BVP,
                "EDA": config.FS_WRIST_EDA,
                "TEMP": config.FS_WRIST_TEMP}

    print("Wrist signal    samples   measured fs   expected fs   matches?   NaNs")
    for name, array in data["signal"]["wrist"].items():
        measured = array.shape[0] / duration_s
        ok = "yes" if abs(measured - expected[name]) < 0.001 else "NO <-- CHECK"
        print("  {:5s}  {:12d}   {:9.3f} Hz   {:9d} Hz   {:10s} {:5d}".format(
            name, array.shape[0], measured, expected[name], ok,
            int(np.isnan(array).sum())))
    print()

    # The pickle has no heart-rate channel - we derive HR from BVP later.
    print("Note: the wrist group has no 'HR' channel. Heart rate is computed")
    print("      from BVP in features.py (see heart_rate_series).")
    print()


def describe_labels(data):
    """
    Print which label values appear, how much data each one holds, and the
    order in which the conditions occurred.

    Comparing the durations and the order of these segments against the
    subject's S{n}_quest.csv (which lists the study protocol timings in
    minutes) is what confirms the label encoding empirically.
    """
    label = data["label"]

    print("Label values present (each sample is 1/700 s):")
    values, counts = np.unique(label, return_counts=True)
    for value, count in zip(values, counts):
        name = config.LABEL_NAMES.get(int(value), "(dropped)")
        print("  label {:d} = {:11s} {:9d} samples = {:7.1f} s = {:5.1f}% "
              "of recording".format(
                  int(value), name, int(count), count / float(config.FS_LABEL),
                  100.0 * count / len(label)))
    print()

    # A "run" is a stretch where the label does not change. np.diff is non-zero
    # exactly at the sample where the label switches, so flatnonzero gives us
    # the boundaries between conditions.
    print("Label segments in time order:")
    boundaries = np.concatenate(([0],
                                 np.flatnonzero(np.diff(label)) + 1,
                                 [len(label)]))
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        value = int(label[start])
        name = config.LABEL_NAMES.get(value, "(dropped)")
        print("  label {:d} = {:11s} from {:8.1f} s to {:8.1f} s  "
              "(lasts {:7.1f} s = {:5.2f} min)".format(
                  value, name,
                  start / float(config.FS_LABEL),
                  end / float(config.FS_LABEL),
                  (end - start) / float(config.FS_LABEL),
                  (end - start) / float(config.FS_LABEL) / 60.0))
    print()


def show_quest_file(subject):
    """Print the protocol timings so the labels above can be cross-checked."""
    path = os.path.join(config.WESAD_DIR, subject, subject + "_quest.csv")
    print("--- " + subject + "_quest.csv (protocol timings, in minutes) ---")
    with open(path, "r") as f:
        for line in f:
            # Only the ORDER/START/END rows describe the protocol; the rest are
            # questionnaire answers (PANAS, STAI, ...) which we do not use.
            if line.startswith(("# ORDER", "# START", "# END", "# Subj")):
                print("  " + line.rstrip())
    print()


def show_e4_layout(subject):
    """
    Print the raw Empatica E4 folder contents and the header of each CSV.

    Per the E4 documentation, each CSV starts with:
        row 1 = session start time as a UTC unix timestamp
        row 2 = sampling rate in Hz
        row 3+ = the samples
    (IBI.csv is the exception: no rate row, and it stores
     "seconds since start, interval duration".)

    We print these only to CONFIRM the layout. The pipeline itself does not
    use these files - it uses the synchronised .pkl, because the raw CSVs
    cover a longer, untrimmed recording that would need re-aligning by
    timestamp before the labels applied to it.
    """
    folder = os.path.join(config.WESAD_DIR, subject, subject + "_E4_Data")
    print("--- " + subject + "_E4_Data (raw Empatica E4 files) ---")
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        size = os.path.getsize(path)
        print("  {:10s} {:12d} bytes".format(filename, size))

    print()
    print("  First 3 rows of each sensor CSV:")
    for filename in ["ACC.csv", "BVP.csv", "EDA.csv", "HR.csv", "TEMP.csv", "IBI.csv"]:
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            rows = [next(f, "").rstrip() for _ in range(3)]
        print("    {:8s} | {}".format(filename, " | ".join(rows)))
    print()


def main():
    # Allow "python 01_load_and_inspect.py S5" to inspect a different subject.
    subject = sys.argv[1] if len(sys.argv) > 1 else "S2"

    print("=" * 70)
    print("Inspecting subject " + subject)
    print("=" * 70)
    print()

    print("Loading " + subject + ".pkl (about 1 GB, this takes a moment)...")
    data = load_subject(subject)
    print("Loaded.")
    print()

    describe_structure(data)
    check_sampling_rates(data)
    describe_labels(data)
    show_quest_file(subject)
    show_e4_layout(subject)

    print("Done. If the 'matches?' column above says yes everywhere and the")
    print("label segments line up with the protocol timings, the schema is as")
    print("the rest of the pipeline expects.")


if __name__ == "__main__":
    main()
