# Health Dashboard (course project viewer)

A local, file-based viewer for two people's wearable heart-rate data, built for
COEN 6322 (AI for Wearable Health Applications).

- Two participants, each collecting only on themselves, on different devices:
  **Anas** (Fitbit Air, Google Takeout export) and **Rahil** (EmotiBit, data
  expected after the mid-project presentation).
- Data comes from **uploaded files**. There is no Google sign-in, no API and no
  network access to any health service.
- The two people's data is **never pooled**. Every page is about one named
  person and shows their name and device in the top bar, because the devices
  differ and that difference is a confound the project keeps visible.
- A model trained on the public **WESAD** dataset is applied to a chosen day of
  a person's data. **The model is trained on WESAD only; uploaded data is only
  scored**, never used to train, tune or fine-tune anything.

Django, server-rendered templates, SQLite, no npm and no build step. Local only:
`ALLOWED_HOSTS` is `localhost` and `127.0.0.1`.

## Run it

Windows, from this folder:

```
.venv\Scripts\python.exe manage.py migrate
.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8000
```

Open http://127.0.0.1:8000. The only value `.env` needs is `DJANGO_SECRET_KEY`
(see `.env.example`).

## Add a person's data

From a Google Takeout, take the `heart_rate_YYYY-MM-DD.csv` files from
`Takeout/Google Health/Physical Activity_GoogleData/Heart Rate/`.

- **Command line:** `manage.py import_heart_rate anas "<path to that folder>"`
  (add `--dry-run` to see the report without storing anything).
- **Browser:** open the person's page and use the import page.

Both use the same code: files are parsed into a plan, a validation report is
shown (what was read, what was skipped and why, row counts, UTC and local date
range, timezone assumed, device names found), and nothing is stored until it is
confirmed. Loading the same file again changes nothing (same SHA-256), and
overlapping files are merged by timestamp. Only files named
`heart_rate_YYYY-MM-DD.csv` are opened; every other file in a Takeout (profile,
location, security) is skipped without being read. Original files are not kept.

Rahil's reader is a placeholder: it needs a sample export before a parser can be
written, so nothing can be loaded for him yet.

## Pages

| URL | What |
|---|---|
| `/` | Choose whose data to view |
| `/p/<slug>/` | One person's overview |
| `/p/<slug>/heart-rate/` | Daily average, minimum and maximum over 7, 30 or 90 days |
| `/p/<slug>/heart-rate/day/<date>/` | One local day as per-minute means |
| `/p/<slug>/raw/` and `/raw/<date>/` | The files loaded and the rows as loaded |
| `/p/<slug>/arousal/` | Score a chosen day with the WESAD-trained model |

Values exactly as in the uploaded file are drawn solid. Values the app computes
(daily averages, per-minute means, spacing, shares) are drawn **dashed**.

## The model

`health/arousal.py` loads a scikit-learn pipeline from the sibling `wearable-ml`
project (`WEARABLE_ML_DIR`, default `../wearable-ml`), reading four files and
writing none. It needs scikit-learn exactly 1.5.2 and refuses anything else.
Results always carry a caveat: the model was trained on laboratory stress, has no
"none of these" class, and a consumer device's smoothed readings are measurably
out of distribution. Each result prints the device and the spacing between its
readings.

## Tests

```
.venv\Scripts\python.exe manage.py test
```

All readings in tests are synthetic.
