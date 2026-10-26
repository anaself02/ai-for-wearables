# wearable-ml

Stress / affect classification from **wrist-worn sensor data**, trained on the
WESAD dataset. The pipeline compares two feature sets:

| Feature set | Features | Represents |
|---|---|---|
| `full` | 40 | Research wristband (Empatica E4): EDA, skin temperature, PPG, accelerometer, heart rate |
| `hr_only` | 5 | Consumer wearable (Fitbit-class): heart rate only |

Headline result (SVM, leave-one-subject-out cross-validation, 3 classes:
baseline / stress / amusement):

| Feature set | Accuracy | Macro-F1 |
|---|---|---|
| `full` | 0.763 | **0.717** |
| `hr_only` | 0.549 | **0.499** |

Full write-up, figures and per-subject scores: [RESULTS.md](RESULTS.md).

---

## 1. Get the dataset

The raw WESAD data (~17 GB) is **not included in this repository**. It is
third-party research data released under its own licence, and it is far too
large for git. Download it yourself:

- **Kaggle mirror (used for this project):**
  https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset
- Original source (UCI Machine Learning Repository):
  https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection

### Option A: browser

1. Open the Kaggle link above and sign in.
2. Click **Download**. You get a zip archive.
3. Extract it (see the folder layout below).

### Option B: Kaggle CLI

```bash
pip install kaggle
# Put your API token at ~/.kaggle/kaggle.json
# (Kaggle -> Settings -> API -> Create New Token)
kaggle datasets download -d orvile/wesad-wearable-stress-affect-detection-dataset -p data/
cd data && unzip wesad-wearable-stress-affect-detection-dataset.zip
```

### Expected folder layout

The scripts read from `wearable-ml/data/WESAD/` (set in
[`scripts/project_config.py`](scripts/project_config.py)). After extracting,
move the folders around if you need to so the layout matches this:

```
wearable-ml/
└── data/
    └── WESAD/
        ├── wesad_readme.pdf
        ├── S2/
        │   ├── S2.pkl          <- the only file the pipeline needs
        │   ├── S2_E4_Data/
        │   ├── S2_quest.csv
        │   ├── S2_readme.txt
        │   └── S2_respiban.txt
        ├── S3/
        ├── ...
        └── S17/
```

There are 15 subjects: S2–S17. **S1 and S12 do not exist.** The dataset
authors removed them because of sensor malfunction, so their absence is
expected.

`data/` is listed in `.gitignore`, so the dataset will never be committed by
accident.

---

## 2. Set up the environment

Python 3.12.

```bash
cd wearable-ml
python -m venv .venv

# Windows
.venv\Scripts\pip install -r requirements.txt
# macOS / Linux
.venv/bin/pip install -r requirements.txt
```

The Django / Google packages at the bottom of `requirements.txt` are only
needed for step 5 (personal data). Steps 1–4 only need numpy, pandas,
scikit-learn, scipy, matplotlib and joblib.

---

## 3. Run the pipeline

Run the scripts in order, from inside `wearable-ml/` with the venv activated:

| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `python scripts/01_load_and_inspect.py [S2]` | Inspects one subject's `.pkl`: structure, sampling rates, label timeline. Writes nothing. | console only |
| 2 | `python scripts/02_feature_extraction.py` | Slides 60 s windows (30 s step) over every subject and computes 40 features per window | `results/features.csv` |
| 3 | `python scripts/03_train_model.py` | Random Forest + SVM, leave-one-subject-out CV, on both feature sets. Saves the SVMs. | `results/cv_predictions_*.csv`, `results/scores_*.json`, `models/*.joblib` |
| 4 | `python scripts/04_evaluate.py` | Confusion matrices, per-subject F1 plots, classification reports | `results/*.png`, `results/*.txt`, `RESULTS.md` |
| 5 | `python scripts/05_apply_to_personal_data.py inspect` then `... predict` | *Optional.* Runs the heart-rate-only model on your own heart-rate data from Google Health | `results/personal/` (git-ignored) |

`results/features.csv` and the trained models are already committed. To
explore the results or retrain without downloading 17 GB, skip steps 1–2 and
start at step 3.

### Step 5: personal data

Script 05 reuses the OAuth credentials and Google Health API client from the
sibling [`health dashboard`](../health%20dashboard/) project, so that project
must be set up first (its `.env`, and a Google account connected through the
dashboard). Its output is a **pipeline demo, not a validated stress detector**:
WESAD's lab-induced stress does not transfer cleanly to everyday heart rate.
The docstring at the top of the script explains why.

---

## Project layout

```
wearable-ml/
├── data/              WESAD dataset (download yourself, git-ignored)
├── models/            trained SVMs + feature configs + model card
├── results/           features table, CV predictions, plots, reports
├── scripts/
│   ├── project_config.py   all paths and constants in one place
│   ├── features.py         feature definitions (full / hr_only)
│   └── 01_ ... 05_         pipeline steps
├── requirements.txt
└── RESULTS.md         generated report
```

---

## Citation

If you use WESAD, cite the original paper:

> Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof
> Van Laerhoven. 2018. *Introducing WESAD, a Multimodal Dataset for Wearable
> Stress and Affect Detection.* In Proceedings of the 20th ACM International
> Conference on Multimodal Interaction (ICMI '18).
