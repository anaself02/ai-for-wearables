# Health Dashboard: project context

A handoff document. Read it before changing anything: it records the decisions
the project has already made and why. It was rewritten for the file-based,
two-person version; an older version described a Google-API dashboard that no
longer exists.

## What this is

A local Django app for a university course (COEN 6322, AI for Wearable Health
Applications). It lets two teammates look at their **own** wearable heart-rate
data from **uploaded files**, and scores a chosen day with a model trained on the
public **WESAD** dataset (15 subjects).

- **Anas**: Fitbit Air, loaded from a Google Takeout export.
- **Rahil**: EmotiBit; data expected after the mid-project presentation. He
  exists as a profile with no data and an honest empty state.

Nothing is live. There is no Google sign-in, API, or network access. Everything is
behind an email-and-password login whose only job is to keep the two people's
profiles apart on one machine.

## The principles that define it

1. **Never pool the two people.** The devices differ and that is a confound. Every
   page is about one named person and shows their name and device in the top bar,
   with a Switch link. `people/repository.py` functions require a `Person` and
   raise on anything else; tests assert one person's page never contains
   another's numbers, name, device or dates.
2. **The model is trained on WESAD only; uploaded data is only scored.** No code
   path fits, tunes or saves a model on uploaded data. Guard tests
   (`people/fitguard.py`) patch every fit and save to raise while the upload and
   scoring flows run, and a source scan fails if an application module fits.
3. **Refuse to overclaim.** The arousal result always carries a caveat in the same
   template branch as its headline (a test asserts it for every label), plus the
   out-of-distribution evidence, the device and the reading spacing. There is no
   ground truth for personal days, so no accuracy is claimed for them.
4. **Solid vs dashed.** Values exactly as in the uploaded file are solid; values
   the app computes from the uploaded rows (daily averages, per-minute means,
   spacing, class shares) carry the `derived` class and are drawn dashed.
5. **No invented data.** A day or person with no readings says so; minutes with no
   readings are chart gaps, never bridged. Rahil's pages show no numbers.

## Accounts and access (members app)

- Django's built-in auth only; email is the login, stored lowercased in `username`
  and `email`. No custom user model, no extra auth libraries.
- `Person.owner` (one-to-one, SET_NULL) is the link; `ProfileClaim` is a request
  (pending, approved, rejected). **Sign-up only ever creates a pending claim** and
  never links; `people/links.py` is the only code that sets `owner`, and it refuses
  to replace an existing link or to link an already-linked account.
- Access (`members/access.py`, also enforced where a person is resolved in
  `people.views.person_page`): anonymous -> `/login/` except `/signup/` and
  `/login/`; staff see the chooser and every profile; a linked account is sent from
  `/` to its own profile and any other slug is a 404; an account with no link (or a
  pending claim) sees only `/waiting/`. Non-staff get 404 for `/admin/`.
- Login throttle (`members/throttle.py`): 5 failures per email then a 5-minute lock,
  cached under a hash of the email. Cookies HttpOnly + SameSite=Lax, CSRF on every
  POST, signed-in responses no-store, dummy email backend. Emails are never logged
  or shown to another account.
- Admin (`/admin/`): profiles (name, device, status, readings, linked email) and
  accounts (linked profile, pending claim, active, last login, filter, approve /
  reject / unlink actions). No password hashes, no heart-rate data, accounts cannot
  be created there. The first administrator is made with `createsuperuser`.

## Stack and constraints

Django, server-rendered templates, SQLite, no npm or build step, no webfonts,
Chart.js pinned from a CDN. Windows host; use `.venv\Scripts\python.exe`. Local
only: `ALLOWED_HOSTS` is `localhost` and `127.0.0.1`. Conventional Commits; every
commit leaves the suite green; never weaken a test to make a change pass (move or
delete the assertion and say so).

## Layout

```
people/            who the data belongs to, and everything about their readings
  models.py        Person, Recording, HeartRateDay (packed per UTC day)
  sources/         device adapters: fitbit_takeout.py, emotibit.py (stub)
  ingest.py        parse -> report -> dedupe -> store (command and page share it)
  spool.py         short-lived holding area between report and confirm
  repository.py    the only way to read readings; requires a Person
  links.py         approve / reject / unlink claims (the only writer of Person.owner)
  views.py         chooser, person pages, arousal; views_import.py: Import data tab
  xlsx.py          first sheet of a workbook -> the same CSV text, header-checked
  management/      import_heart_rate
members/           signup, login, logout, waiting page, throttle, access rules, admin
health/            arousal.py: the WESAD model runner (and its tests)
config/            settings, urls
templates/         base.html, people/*, health/_chart_scripts.html
static/            app.css, charts.js, theme.js
docs/              ETHICS_AND_PRIVACY.md
```

## Data model and decisions

- Readings are stored **per UTC day** as packed arrays (uint32 second offsets,
  uint16 bpm x 10), because the source files are cut on UTC and a day at ~3 s is
  ~30,000 rows. Local days are assembled at read time in the person's timezone
  (default America/Toronto) from real midnights, so a daylight-saving day is 23
  or 25 hours.
- Re-importing is a no-op (SHA-256 per file per person) and overlapping files
  merge by timestamp. **Uploaded files are not retained**: they are parsed in the
  first request, and only the parsed readings wait (deleted on confirm, cancel or
  after an hour) for the Confirm click.
- The Import data tab accepts `.csv` / `.xlsx` (first sheet) by HEADER (exactly
  `timestamp`, `beats per minute`, `data source`; any other column refuses the file
  unread) and a `.zip` whose members are named `heart_rate_YYYY-MM-DD.csv` (others
  skipped unopened, nothing extracted to disk). There are caps on file size, file
  count and total uncompressed size. `DATA_UPLOAD_MAX_NUMBER_FILES` is raised
  globally (a global setting; this is the only route that takes files).
- A Takeout holds hundreds of unrelated files, some personal. The Fitbit reader
  accepts **by name** (`heart_rate_YYYY-MM-DD.csv`); everything else is skipped
  unopened. It refuses timestamps without a timezone rather than guessing.
- Rahil's reader is a stub: any import shows "EmotiBit import is not available
  yet: it needs a sample export from Rahil." and saves nothing. Its docstring and
  his Import data page list what is needed from his export (layout, whether
  heart rate is present or only raw PPG, sampling rate, timestamp format and
  timezone, units, markers). Deriving heart rate from raw PPG would be a separate,
  declared decision.

## The arousal model

`health/arousal.py` loads the pipeline from the sibling **wearable-ml** project
(`WEARABLE_ML_DIR`), reading four files and writing none, and refuses any
scikit-learn other than 1.5.2. Windows are 60 s with a 30 s step, at least 10
readings each, anchored to local midnight; a day needs at least 60 such windows
(a constant the page prints). The pre-declared display rules (out-of-distribution
flag at 25% of windows outside the training range; class-dominance flag at 60%)
are in `wearable-ml/PRE_DECLARED.md`. The model is the saved 3-class
heart-rate-only SVM; its macro-F1 on WESAD under leave-one-subject-out is 0.499,
weak but above shuffled labels (0.332). WESAD has **15** subjects.

## Known gaps

- Only Anas's data exists. Rahil's depends on a sample export.
- The Fitbit's readings are device-smoothed (~2-3 s), the training data is
  beat-level (~77 values per window): the same feature names describe different
  measurements, and Anas's days are typically flagged out of distribution.
- `wearable-ml/scripts/05_apply_to_personal_data.py` (Pass 1) imported the old
  Google client and no longer works; its Pass 2 and the demo do not depend on it.
- The charts, pages, forms and the chooser's responsive layout were checked
  through the test client, the markup and the stylesheet, not in a browser.
- The admin actions and the access rules are tested through the test client; the
  first real login and approval have not been exercised by hand.

## Working on it

```
.venv\Scripts\python.exe manage.py createsuperuser        # once, with your email
.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8000
.venv\Scripts\python.exe manage.py test
.venv\Scripts\python.exe manage.py import_heart_rate anas "<Heart Rate folder>" --dry-run
```

Back up `db.sqlite3` before migrating a real database. Sample data lives in
`data-samples/` (gitignored); never copy real files into tests or fixtures: tests
build synthetic files with the real column names in `people/testing.py`.
