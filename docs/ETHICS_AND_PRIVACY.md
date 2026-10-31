# Ethics and privacy

Covers the dashboard in this repository, which holds the data, and the
`wearable-ml` project, which trains the model on public data only.

## 1. Who is involved, and why

Two people, both course teammates, each the subject of **their own** data only:

- **Anas**, who wears a Fitbit Air.
- **Rahil**, who will wear an EmotiBit. His data has not been received and
  nothing is stored for him.

Nobody's data is collected by anyone else, and nobody else's data is in the app.
Each teammate has agreed to their own data being used for this graded course
project. Data is used to look at the structure of consumer-wearable heart rate,
and to *score* it with a model trained elsewhere. It is not used for anything
else.

Labels are first names only. No surname, email, birthday, location or account
identifier is stored.

## 2. What is stored

For each person, the **heart-rate readings** from the uploaded files (time and
beats per minute) packed per UTC day, plus a record per file: its name, SHA-256,
row count, time range, the device name found in it, and the timezone assumed.
**The uploaded files themselves are not kept**; the owner keeps the originals.

A short-lived holding area (`uploads_tmp/`, gitignored) holds a file between the
validation report and the confirmation. It is deleted on confirm, on cancel, or
when an hour old.

The database (`db.sqlite3`) and `data-samples/` are gitignored. Neither is ever
committed.

## 3. What is not read

A Google Takeout holds hundreds of unrelated files, including a profile (name,
email, birthday), device lists, location folders and security data. The loader
accepts files **by name** (`heart_rate_YYYY-MM-DD.csv`) and every other file is
skipped without being opened, including inside a zip. Tests assert that decoy
files are never opened.

## 4. No Google connection

There is no Google sign-in, OAuth client, token storage or API access. An earlier
version signed in with Google and stored encrypted tokens; that code, its tables
and the user rows it created were removed, and a migration drops the old tables
so tokens do not linger on an existing database. The app makes no outbound
network requests.

## 5. Local only

The app runs on `localhost` / `127.0.0.1` (`ALLOWED_HOSTS` is those two names).
It is not deployed and has no remote-access features. Nothing is sent to any
external service.

## 6. Machine learning

The model is trained **only on the public WESAD dataset** (15 subjects, in a
laboratory). Uploaded data is **only scored**: no code path fits, tunes or
fine-tunes on it. Tests run the upload and scoring flows with every fit and save
patched to raise, and a source scan fails if an application module fits or saves
a model.

The model is **not a medical device and not a validated stress detector**. It
was trained on laboratory-induced stress, has no "none of these" class, and a
consumer device's smoothed readings sit measurably outside its training
distribution. Every result prints a caveat and the device and spacing of the
readings; where the day is out of distribution or the predictions are degenerate
the page says the class shares are not interpretable. There is no ground truth
for anyone's real days, so no accuracy is claimed for uploaded data.

## 7. Why the two people are never combined

The devices differ (a Fitbit's smoothed readings every few seconds, an
EmotiBit's different export), and that difference is a confound. Every page is
scoped to one named person, shows their name and device, and the code that reads
readings refuses to run without naming the person. Tests assert that one
person's page never contains another's numbers, name, device or dates.

## 8. What this is not

- Not a commercial product, not deployed, not handling anyone else's data.
- PIPEDA and Quebec Law 25 do not bind a personal course project, but their
  principles shape it: purpose limitation (course use only), data minimisation
  (heart rate only, nothing else read), consent (between teammates), and
  erasure (delete `db.sqlite3` and `uploads_tmp/`).
